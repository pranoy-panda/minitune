# minitune/autotuner/strategies.py
import math
import torch
from dataclasses import dataclass
from typing import Dict, Optional, List
from abc import ABC, abstractmethod

from skopt import Optimizer
from skopt.space import Integer, Categorical
SKOPT_AVAILABLE = True

@dataclass
class TrialConfig:
    micro_batch_size: int
    grad_accum_steps: int
    use_flash_attn: bool = True

class TuningStrategy(ABC):
    @abstractmethod
    def next_trial(self) -> Optional[TrialConfig]:
        pass

    @abstractmethod
    def report_result(self, config: TrialConfig, result: Dict):
        pass

class HeuristicSearch(TuningStrategy):
    """
    Ramps up Micro-Batch Size until memory threshold is hit or OOM occurs.
    """
    def __init__(
        self, 
        target_global_batch_size: int = 32, 
        num_gpus: int = 1,
        memory_threshold_pct: float = 0.90,  # Safety margin (stop if >90%)
        start_micro_batch: int = 1
    ):
        self.target_gbs = target_global_batch_size
        self.num_gpus = num_gpus
        self.memory_threshold_pct = memory_threshold_pct
        
        # --- ROBUST VRAM DETECTION ---
        if torch.cuda.is_available():
            # Use current_device() to be safe in distributed contexts
            device_id = torch.cuda.current_device()
            # total_memory is in Bytes. Convert to MB.
            self.total_mem_mb = torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024
        else:
            print("No GPU detected. Exiting tuner.")
            exit()
            
        print(f"[AutoTuner] Detected VRAM: {self.total_mem_mb/1024:.2f} GB per GPU")
        print(f"[AutoTuner] Safety Limit: {self.total_mem_mb * self.memory_threshold_pct / 1024:.2f} GB")

        # Define search space: Powers of 2 starting from 'start_micro_batch'
        self.micro_batch_candidates = []
        bs = start_micro_batch
        while bs * num_gpus <= target_global_batch_size:
            self.micro_batch_candidates.append(bs)
            bs *= 2
            
        self.current_idx = 0
        self.best_config = None
        self.best_mfu = 0.0
        self.stop_search = False
        self.history = []

    def next_trial(self) -> Optional[TrialConfig]:
        if self.stop_search or self.current_idx >= len(self.micro_batch_candidates):
            return None

        mb = self.micro_batch_candidates[self.current_idx]
        
        # Calculate Accumulation to maintain Global Batch Size constant
        # GBS = Micro * GPUs * Accum
        # Accum = GBS / (Micro * GPUs)
        accum = max(1, self.target_gbs // (mb * self.num_gpus))
        
        return TrialConfig(
            micro_batch_size=mb,
            grad_accum_steps=accum
        )

    def report_result(self, config: TrialConfig, result: Dict):
        self.history.append({"config": config, "result": result})
        
        status = result.get("status", "error")
        
        # 1. Handle OOM (Immediate Stop)
        if status == "oom":
            print(f"OOM detected at Batch Size {config.micro_batch_size}. Reverting to previous best.")
            self.stop_search = True
            return

        # 2. Handle Success
        if status == "success":
            mfu = result.get("mfu", 0.0)
            mem_mb = result.get("peak_memory_mb", 0.0)
            
            # Update Best
            if mfu > self.best_mfu:
                self.best_mfu = mfu
                self.best_config = config
            
            # 3. Check Safety Threshold
            limit_mb = self.total_mem_mb * self.memory_threshold_pct
            
            if mem_mb > limit_mb: 
                print(f"Memory usage ({mem_mb:.0f} MB) exceeds safety limit. Stopping search.")
                self.stop_search = True
            else:
                # If safe, keep pushing!
                self.current_idx += 1
                

class BayesianSearch(TuningStrategy):
    """
    Bayesian Optimization using Gaussian Processes (GP-UCB).
    
    ALGORITHM DESCRIPTION:
    ----------------------
    1. Search Space Modeling:
       We model the search space as 'Batch Size Power' (integer x where BS=2^x).
       This linearizes the exponential nature of memory usage, making it easier
       for the Gaussian Process to model.
       
    2. Surrogate Model (Gaussian Process):
       We maintain a probabilistic model P(y|x) where:
       - x: Configuration (Batch Size Power)
       - y: Score (MFU - MemoryPenalty)
       The GP predicts the expected performance (mean) and uncertainty (variance)
       for unobserved configurations.
       
    3. Acquisition Function (UCB - Upper Confidence Bound):
       Next Config = argmax( Mean(x) + kappa * StdDev(x) )
       - Exploitation: High Mean (try what we think is good)
       - Exploration:  High StdDev (try what we haven't seen yet)
       
    4. OOM Handling:
       If a configuration crashes (OOM), we assign it a severe penalty score (-1.0).
       The GP learns the 'OOM Cliff' and naturally avoids that region of the space.
       
    5. Constraint Satisfaction:
       Gradient Accumulation is derived deterministically from the suggested 
       Batch Size to maintain the Target Global Batch Size constant.
    """

    def __init__(
        self, 
        target_global_batch_size: int = 32, 
        num_gpus: int = 1,
        n_calls: int = 15,            # Total trials to run
        n_random_starts: int = 3      # Initial random exploration steps
    ):
        if not SKOPT_AVAILABLE:
            raise ImportError("BayesianSearch requires 'scikit-optimize'. Install with `pip install scikit-optimize`")

        self.target_gbs = target_global_batch_size
        self.num_gpus = num_gpus
        self.n_calls = n_calls
        self.current_call = 0
        self.best_config = None
        self.best_mfu = 0.0

        # Define Search Space
        # We tune the EXPONENT of the batch size: 2^0 (1) to 2^5 (32)
        # We assume 32 is the max reasonable batch size for a single GPU in this context
        self.search_space = [
            Integer(0, 5, name='batch_power') 
        ]

        # Initialize Optimizer
        # base_estimator="gp": Uses Gaussian Process
        # acq_func="LCB": Lower Confidence Bound (skopt minimizes by default, we negate MFU)
        self.optimizer = Optimizer(
            dimensions=self.search_space,
            base_estimator="gp",
            n_initial_points=n_random_starts,
            acq_func="LCB", 
            acq_optimizer="auto",
            random_state=42
        )
        
        self.history = []

    def next_trial(self) -> Optional[TrialConfig]:
        if self.current_call >= self.n_calls:
            return None

        # Ask optimizer for next parameters
        # Returns a list of values, e.g., [3] for 2^3=8
        suggested_params = self.optimizer.ask()
        batch_power = suggested_params[0]
        
        # Convert Power -> Actual Batch Size
        mb = 2 ** batch_power
        
        # Calculate Accumulation to maintain Global Batch Size
        accum = max(1, self.target_gbs // (mb * self.num_gpus))
        
        self.current_config_params = suggested_params # Store for reporting
        self.current_call += 1

        return TrialConfig(
            micro_batch_size=mb,
            grad_accum_steps=accum
        )

    def report_result(self, config: TrialConfig, result: Dict):
        status = result.get("status", "error")
        mfu = result.get("mfu", 0.0)
        
        # Calculate Score for the Optimizer
        # skopt minimizes objectives, so we return NEGATIVE MFU.
        
        if status == "oom":
            # Penalize OOM heavily so the GP learns the cliff
            score = 0.0 # equivalent to 0% MFU
        elif status == "success":
            score = -mfu # Negative because we minimize
            
            # Update Best Config Logic
            if mfu > self.best_mfu:
                self.best_mfu = mfu
                self.best_config = config
        else:
            score = 0.0 # Error penalty

        # Tell the optimizer the result (x, y)
        # x must be the list of params we got from .ask() earlier
        # In a robust system, we ensure config maps back to params
        
        # Reconstruct params from config for safety
        # log2(batch_size)
        batch_power = int(math.log2(config.micro_batch_size))
        
        self.optimizer.tell([batch_power], score)
        
        self.history.append({
            "config": config,
            "result": result,
            "score": score
        })