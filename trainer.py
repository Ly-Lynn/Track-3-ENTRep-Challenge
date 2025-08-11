import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from constants import (
    DEFAULT_BETAS, DEFAULT_EMA_DECAY, DEFAULT_LEARNING_RATE, 
    DEFAULT_PATIENCE, DEFAULT_START_FACTOR, DEFAULT_WEIGHT_DECAY,
    LOG_FREQUENCY, MAX_GRAD_NORM, WARMUP_RATIO
)
from loss import InfoNCELoss
from visualizer import LossVisualizer
class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = DEFAULT_EMA_DECAY):
        """
        Initialize EMA.
        
        Args:
            model: PyTorch model
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backup = {}
        self.register()

    def register(self) -> None:
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self) -> None:
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class MedicalVLMTrainer:
    """Trainer class for Medical Vision-Language Models."""
    
    def __init__(
        self, 
        model: nn.Module, 
        train_loader, 
        val_loader, 
        config: Dict[str, Any]
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup model and device
        self.model, self.is_parallel = self._setup_model(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup EMA
        self.ema = EMA(self.model, decay=DEFAULT_EMA_DECAY)
        
        # Setup loss function and metrics
        self.loss_fn = InfoNCELoss()
        self._reset_training_state()
        
        # Setup output directory
        self.model_save_path = os.path.join(
            self.config['output_path'], 
            self.config['model_name']
        )
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def _setup_model(self, model: nn.Module) -> tuple:
        """Setup model for training with multi-GPU support."""
        num_gpus = torch.cuda.device_count()
        print(f"🔍 Detected {num_gpus} GPU(s)")
        
        if num_gpus > 1:
            print(f"🚀 Using DataParallel with {num_gpus} GPUs")
            model = nn.DataParallel(model)
            is_parallel = True
        else:
            print(f"📱 Using single GPU/CPU")
            is_parallel = False
            
        return model.to(self.device), is_parallel
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with configuration."""
        return optim.AdamW(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', DEFAULT_LEARNING_RATE),
            weight_decay=self.config.get('weight_decay', DEFAULT_WEIGHT_DECAY),
            betas=DEFAULT_BETAS
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        steps_per_epoch = len(self.train_loader)
        warmup_steps = int(WARMUP_RATIO * steps_per_epoch)
        
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=DEFAULT_START_FACTOR, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=steps_per_epoch * 100 - warmup_steps
        )
        
        return SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    
    def _reset_training_state(self) -> None:
        """Reset training state for new training run."""
        self.best_val_loss = float('inf')
        self.patience = DEFAULT_PATIENCE
        self.patience_counter = 0
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        self.visualizer = LossVisualizer(self.history)

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images, input_ids, attention_mask)
                
                loss = self.loss_fn(
                    outputs["image_embeds"],
                    outputs["text_embeds"]
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                
                self.optimizer.step()
                self.scheduler.step()
                
                self.ema.update()
                
                train_loss += loss.item()
                
                if batch_idx % LOG_FREQUENCY == 0:
                    print(f"\nBatch {batch_idx}, Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.history["train_loss"].append(avg_train_loss)
            
            self.ema.apply_shadow()
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validation"):
                    images = batch["image"].to(self.device)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    outputs = self.model(images, input_ids, attention_mask)
                    
                    loss = self.loss_fn(
                        outputs["image_embeds"],
                        outputs["text_embeds"]
                    )
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            self.history["val_loss"].append(avg_val_loss)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model(os.path.join(self.model_save_path, 'best.pt'))
                self.patience_counter = 0
                print(f">>> Model best saved (val_loss: {avg_val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f">>> Early stopping triggered after {epoch+1} epochs")
                break
                
            self.ema.restore()
            
        self.visualizer.plot(save_path=os.path.join(self.model_save_path, 'infoNCE.png'))
        return self.history


    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluation"):
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(images, input_ids, attention_mask)
                
                loss = self.loss_fn(
                    outputs["image_embeds"],
                    outputs["text_embeds"]
                )

                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Handle DataParallel model saving
        if self.is_parallel:
            # Save the module inside DataParallel
            torch.save(self.model.module.state_dict(), path)
        else:
            # Save normal model
            torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        
        # Handle DataParallel model loading
        if self.is_parallel:
            # Load into the module inside DataParallel
            self.model.module.load_state_dict(state_dict)
        else:
            # Load into normal model
            self.model.load_state_dict(state_dict)
            
        self.model = self.model.to(self.device)
        
    def update_model(self, model):
        self.model = model
        self.ema = EMA(model)
        self.optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config['num_epochs'])
