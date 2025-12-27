import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
from omegaconf import OmegaConf

from minitune.sft import SFTTrainer
from minitune.config import TrainConfig, PeftConfig
# from minitune.losses import FocalLoss

# --- Fixtures ---
@pytest.fixture
def mock_config():
    """Returns a valid TrainConfig object."""
    dummy_config_path = "tests/dummy_config.yaml"
    dummy_config_obj = OmegaConf.load(dummy_config_path)
    return TrainConfig(**dummy_config_obj)

@pytest.fixture
def mock_accelerator():
    """Mocks the Accelerate library to avoid hardware checks."""
    with patch("minitune.sft.Accelerator") as mock_acc_cls:
        mock_acc = mock_acc_cls.return_value
        # Mock device
        mock_acc.device = torch.device("cpu")
        # Mock prepare to just return the objects as-is
        mock_acc.prepare.side_effect = lambda m, o, d: (m, o, d)
        # Mock is_main_process to True so logging logic runs
        mock_acc.is_main_process = True
        # Mock the accumulate context manager
        mock_acc.accumulate.return_value.__enter__ = MagicMock()
        mock_acc.accumulate.return_value.__exit__ = MagicMock()
        yield mock_acc

# --- Tests ---

def test_standard_initialization(mock_config, mock_accelerator):
    """Test standard initialization loads components correctly."""
    with patch("minitune.sft.AutoModelForCausalLM") as mock_model_cls, \
         patch("minitune.sft.AutoTokenizer") as mock_tok_cls, \
         patch("minitune.sft.load_and_prepare_dataset") as mock_data_fn, \
         patch("minitune.sft.SummaryWriter"):
        
        mock_data_fn.return_value.__len__.return_value = 100

        mock_model = mock_model_cls.from_pretrained.return_value
        mock_tokenizer = mock_tok_cls.from_pretrained.return_value
        
        trainer = SFTTrainer(mock_config)
        
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.train_dataloader is not None

def test_peft_initialization(mock_config, mock_accelerator):
    """
    Test that if PEFT config is provided, the model is wrapped using get_peft_model.
    """
    # Add PEFT config
    mock_config.peft = PeftConfig(r=8, lora_alpha=16)

    with patch("minitune.sft.AutoModelForCausalLM"), \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset"), \
         patch("minitune.sft.get_peft_model") as mock_get_peft, \
         patch("minitune.sft.LoraConfig") as mock_lora_config, \
         patch("minitune.sft.SummaryWriter"):
        
        # Giving the mock model some dummy parameters ---
        # The optimizer needs to iterate over parameters. We create a fake tensor.
        mock_model = mock_get_peft.return_value
        dummy_param = torch.tensor([1.0], requires_grad=True)
        # When .parameters() is called, return an iterator containing our dummy param
        mock_model.parameters.return_value = iter([dummy_param])

        # ACT
        trainer = SFTTrainer(mock_config)
        
        # ASSERT
        mock_lora_config.assert_called()
        mock_get_peft.assert_called_once()
        # The trainer model should now be the result of get_peft_model
        assert trainer.model == mock_get_peft.return_value

def test_training_loop_execution(mock_config, mock_accelerator):
    """
    Test the train() method loops through data, calls backward, and steps optimizer.
    This simulates a training run without actual computation.
    """
    # Create a dummy batch
    dummy_batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}
    
    with patch("minitune.sft.AutoModelForCausalLM") as mock_model_cls, \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset"), \
         patch("minitune.sft.DataLoader") as mock_loader_cls, \
         patch("minitune.sft.SummaryWriter"):

        # Setup the mock model to return a mock loss
        mock_model = mock_model_cls.from_pretrained.return_value
        mock_model.return_value.loss = torch.tensor(0.5, requires_grad=True)
        
        # Setup the DataLoader to return our dummy batch once, then stop
        mock_loader = MagicMock()
        mock_loader.__len__.return_value = 1
        mock_loader.__iter__.return_value = iter([dummy_batch])
        mock_loader_cls.return_value = mock_loader
        
        # Initialize
        trainer = SFTTrainer(mock_config)
        
        # Spy on the optimizer (created inside init)
        trainer.optimizer = MagicMock()
        
        # ACT
        trainer.train()
        
        # ASSERT
        # 1. Did we forward pass?
        mock_model.assert_called_with(**dummy_batch)
        
        # 2. Did we backward pass?
        # Note: Accelerator wraps backward, so we check accelerator
        mock_accelerator.backward.assert_called()
        
        # 3. Did we step optimizer?
        trainer.optimizer.step.assert_called_once()
        trainer.optimizer.zero_grad.assert_called_once()
        
        # 4. Did we try to save?
        # (Since we mocked prepare, the model is still the raw mock, so save_pretrained should be called on it)
        mock_model.save_pretrained.assert_called()

# --- Tests for Custom Loss & Data Injection ---

def test_custom_loss_initialization(mock_config, mock_accelerator):
    """Test that SFTTrainer accepts a custom loss function."""
    custom_loss = FocalLoss(gamma=2.0)
    
    with patch("minitune.sft.AutoModelForCausalLM"), \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset"), \
         patch("minitune.sft.SummaryWriter"):
        
        trainer = SFTTrainer(mock_config, loss_fn=custom_loss)
        assert trainer.loss_fn == custom_loss

def test_programmatic_dataset_injection(mock_config, mock_accelerator):
    """Test that we can pass a dataset object directly, overriding config."""
    dummy_dataset = MagicMock()
    
    with patch("minitune.sft.AutoModelForCausalLM"), \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset") as mock_load_data, \
         patch("minitune.sft.SummaryWriter"):
             
        trainer = SFTTrainer(mock_config, train_dataset=dummy_dataset)
        
        # The internal loader logic should NOT be called
        mock_load_data.assert_not_called()
        # The internal dataloader should be using our injected dataset
        # (Note: DataLoader wraps the dataset, so we check the dataset attr)
        # Since we mocked DataLoader class in the other test, here it might be real or implicit.
        # But we can check the trainer attribute directly.
        assert trainer.train_dataset == dummy_dataset

# --- Tests for Unit Logic (Loss Function) ---

def test_focal_loss_math():
    """Test the calculation of Focal Loss on CPU."""
    loss_fn = FocalLoss(gamma=2.0, reduction='mean')
    
    # Simple logits: Batch 1, Seq 1, Vocab 2
    # Logits: Class 0 is high, Class 1 is low
    logits = torch.tensor([[[10.0, -10.0]]], requires_grad=True)
    # Label is 0
    labels = torch.tensor([[0]])
    
    # CE Loss would be near 0. Focal loss should be even closer to 0 (easy example)
    loss_easy = loss_fn(logits, labels)
    
    # Hard example: Label is 1 (but model predicts 0 strongly)
    labels_hard = torch.tensor([[1]])
    loss_hard = loss_fn(logits, labels_hard)
    
    assert loss_hard > loss_easy
    assert loss_hard.requires_grad