# FILE: minitune/utils/profiling.py
import torch
import time
from collections import deque

# Hardware specs for common GPUs (Theoretical Peak TFLOPS for BF16/FP16)
# Sources: NVIDIA Whitepapers
GPU_SPECS = {
    "NVIDIA A40": 149.7e12,  # ~150 TFLOPS (Tensor Float 32 / BF16)
    "NVIDIA A100-SXM4-40GB": 312e12,
    "NVIDIA A100-PCIE-40GB": 312e12,
    "NVIDIA H100": 989e12,   # FP16 Tensor Core
    "Tesla T4": 65e12,
    "Unknown": 100e12        # Fallback
}

def get_gpu_peak_flops():
    """
    Returns the theoretical peak FLOPS for the current GPU.
    """
    if not torch.cuda.is_available():
        return 1.0 # CPU fallback
    
    device_name = torch.cuda.get_device_name(0)
    
    # Simple substring matching
    for name, flops in GPU_SPECS.items():
        if name in device_name:
            return flops
            
    # Heuristic for A40 if exact string match fails
    if "A40" in device_name:
        return 149.7e12
        
    print(f"Warning: GPU {device_name} not found in specs. Using default.")
    return GPU_SPECS["Unknown"]

class MFUCalculator:
    """
    Calculates Model Flops Utilization (MFU) in real-time.
    
    THEORY:
    -------
    1. FLOPs (Floating Point Operations): 
       The number of math operations (multiplications/additions) required.
       For Transformers, a standard approximation is:
       
       FLOPs_per_token ≈ 6 * N (Kaplan et al., PaLM paper)
       
       Where N is the number of parameters.
       - 2N for the Forward Pass (Matrix Multiply + Bias)
       - 4N for the Backward Pass (2N for gradients w.r.t weights, 2N for input)
       
    2. MFU (Model Flops Utilization):
       A percentage representing how efficiently we use the hardware.
       
       MFU = (Achieved_FLOPs_per_second) / (Theoretical_Peak_FLOPs_per_second)
       
       - 30-40%: Standard unoptimized PyTorch.
       - 50-60%: Good (FlashAttention, FSDP).
       - 70%+: State of the Art (Custom CUDA kernels).
    """
    def __init__(self, model, avg_window=20):
        self.peak_flops = get_gpu_peak_flops()
        self.param_count = sum(p.numel() for p in model.parameters())
        
        # 6N formula is standard for Decoder-only Transformers (PaLM paper)
        self.flops_per_token = 6 * self.param_count
        
        self.avg_window = avg_window
        self.step_times = deque(maxlen=avg_window)
        self.last_time = None

    def step(self, batch_total_tokens):
        """
        Call this after optimizer.step().
        Returns: current_mfu (float)
        """
        # Synchronize CPU and GPU for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        current_time = time.time()
        
        if self.last_time is None:
            self.last_time = current_time
            return 0.0, 0.0  # No MFU on first call
            
        step_time = current_time - self.last_time
        self.step_times.append(step_time)
        self.last_time = current_time
        
        # Calculate Moving Average
        avg_time = sum(self.step_times) / len(self.step_times)
        
        # Throughput
        tokens_per_sec = batch_total_tokens / avg_time
        
        # MFU Calculation
        achieved_flops = tokens_per_sec * self.flops_per_token
        mfu = achieved_flops / self.peak_flops
        
        return mfu, tokens_per_sec