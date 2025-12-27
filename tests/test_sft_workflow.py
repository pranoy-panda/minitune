# tests/test_sft_workflow.py
import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
from minitune.sft import SFTTrainer
from minitune.config import TrainConfig, PeftConfig
from minitune.losses import FocalLoss
from omegaconf import OmegaConf

# --- Fixtures ---

DUMMY_CONFIG = {
    "model": {"name_or_path": "gpt2", "use_flash_attention_2": False},
    "data": {"path": "dummy", "prompt_column": "p"},
    "sft": {
        "output_dir": "tmp_test_output", 
        "batch_size": 1,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "learning_rate": 1e-4
    }
}

@pytest.fixture
def mock_config():
    """Returns a valid TrainConfig object."""
    conf = OmegaConf.create(DUMMY_CONFIG)
    return TrainConfig(**conf)

@pytest.fixture
def mock_accelerator():
    """Mocks the Accelerate library."""
    with patch("minitune.sft.Accelerator") as mock_acc_cls:
        mock_acc = mock_acc_cls.return_value
        mock_acc.device = torch.device("cpu")
        mock_acc.prepare.side_effect = lambda m, o, d: (m, o, d)
        mock_acc.is_main_process = True
        mock_acc.accumulate.return_value.__enter__ = MagicMock()
        mock_acc.accumulate.return_value.__exit__ = MagicMock()
        yield mock_acc

# --- Tests for Existing Functionality ---

def test_standard_initialization(mock_config, mock_accelerator):
    """Test standard initialization loads components correctly."""
    # PATCH EVERYTHING that interacts with hardware or internet
    with patch("minitune.sft.AutoModelForCausalLM") as mock_model_cls, \
         patch("minitune.sft.AutoTokenizer") as mock_tok_cls, \
         patch("minitune.sft.load_and_prepare_dataset") as mock_data_fn, \
         patch("minitune.sft.DataLoader"), \
         patch("minitune.sft.DataCollatorForLanguageModeling"), \
         patch("minitune.sft.AdamW"), \
         patch("minitune.sft.SummaryWriter"):
        
        # Setup specific mocks
        mock_data_fn.return_value.__len__.return_value = 100
        mock_model = mock_model_cls.from_pretrained.return_value
        mock_tokenizer = mock_tok_cls.from_pretrained.return_value
        
        # ACT
        trainer = SFTTrainer(mock_config)
        
        # ASSERT
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.train_dataloader is not None

def test_peft_initialization(mock_config, mock_accelerator):
    """Test PEFT wrapping."""
    mock_config.peft = PeftConfig(r=8, lora_alpha=16)

    with patch("minitune.sft.AutoModelForCausalLM"), \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset") as mock_data_fn, \
         patch("minitune.sft.DataLoader"), \
         patch("minitune.sft.DataCollatorForLanguageModeling"), \
         patch("minitune.sft.AdamW"), \
         patch("minitune.sft.get_peft_model") as mock_get_peft, \
         patch("minitune.sft.LoraConfig") as mock_lora_config, \
         patch("minitune.sft.SummaryWriter"):
        
        mock_data_fn.return_value.__len__.return_value = 100
        
        # We don't need to mock parameters anymore because AdamW is mocked!
        
        trainer = SFTTrainer(mock_config)
        
        mock_lora_config.assert_called()
        mock_get_peft.assert_called_once()
        assert trainer.model == mock_get_peft.return_value.to(torch.bfloat16) # we do a typecast so that the base model and lora adapters have same precision. ideally we should not hard code it.

def test_training_loop_execution(mock_config, mock_accelerator):
    """Test the training loop steps."""
    dummy_batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}
    
    with patch("minitune.sft.AutoModelForCausalLM") as mock_model_cls, \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset") as mock_data_fn, \
         patch("minitune.sft.DataLoader") as mock_loader_cls, \
         patch("minitune.sft.DataCollatorForLanguageModeling"), \
         patch("minitune.sft.AdamW"), \
         patch("minitune.sft.SummaryWriter"):

        mock_data_fn.return_value.__len__.return_value = 100

        mock_model = mock_model_cls.from_pretrained.return_value
        mock_model.return_value.loss = torch.tensor(0.5, requires_grad=True)
        
        # Mock DataLoader to yield one batch
        mock_loader = MagicMock()
        mock_loader.__len__.return_value = 1
        mock_loader.__iter__.return_value = iter([dummy_batch])
        mock_loader_cls.return_value = mock_loader
        
        trainer = SFTTrainer(mock_config)
        
        # ACT
        trainer.train()
        
        # ASSERT
        mock_accelerator.backward.assert_called()
        # Since AdamW is mocked, we check the mock instance
        trainer.optimizer.step.assert_called_once()

# --- Tests for New Functionality ---

def test_custom_loss_initialization(mock_config, mock_accelerator):
    """Test custom loss injection."""
    custom_loss = FocalLoss(gamma=2.0)
    
    with patch("minitune.sft.AutoModelForCausalLM"), \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset") as mock_data_fn, \
         patch("minitune.sft.DataLoader"), \
         patch("minitune.sft.DataCollatorForLanguageModeling"), \
         patch("minitune.sft.AdamW"), \
         patch("minitune.sft.SummaryWriter"):
        
        mock_data_fn.return_value.__len__.return_value = 100
        
        trainer = SFTTrainer(mock_config, loss_fn=custom_loss)
        assert trainer.loss_fn == custom_loss

def test_programmatic_dataset_injection(mock_config, mock_accelerator):
    """Test injecting a dataset directly."""
    dummy_dataset = MagicMock()
    dummy_dataset.__len__.return_value = 50
    
    with patch("minitune.sft.AutoModelForCausalLM"), \
         patch("minitune.sft.AutoTokenizer"), \
         patch("minitune.sft.load_and_prepare_dataset") as mock_load_data, \
         patch("minitune.sft.DataLoader"), \
         patch("minitune.sft.DataCollatorForLanguageModeling"), \
         patch("minitune.sft.AdamW"), \
         patch("minitune.sft.SummaryWriter"):
             
        trainer = SFTTrainer(mock_config, train_dataset=dummy_dataset)
        
        mock_load_data.assert_not_called()
        assert trainer.train_dataset == dummy_dataset

def test_focal_loss_math():
    """Test Focal Loss calculation."""
    loss_fn = FocalLoss(gamma=2.0, reduction='mean')
    
    # Simple logits: Batch 1, Seq 1, Vocab 2
    logits = torch.tensor([[[10.0, -10.0]]], requires_grad=True)
    labels = torch.tensor([[0]])
    
    loss_easy = loss_fn(logits, labels)
    
    # Hard example: Label is 1 (but model predicts 0 strongly)
    labels_hard = torch.tensor([[1]])
    loss_hard = loss_fn(logits, labels_hard)
    
    assert loss_hard > loss_easy
    assert loss_hard.requires_grad