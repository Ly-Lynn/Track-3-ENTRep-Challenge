import torch
import torch.optim as optim
from tqdm import tqdm
from visualizer import LossVisualizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
import numpy as np
from loss import InfoNCELoss
import os
import torch.nn as nn
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class MedicalVLMTrainer:
    def __init__(self, model, 
                 train_loader, 
                 val_loader, 
                 config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU Detection and DataParallel setup
        self.num_gpus = torch.cuda.device_count()
        print(f"🔍 Detected {self.num_gpus} GPU(s)")
        
        if self.num_gpus > 1:
            print(f"🚀 Using DataParallel with {self.num_gpus} GPUs")
            model = nn.DataParallel(model)
            self.is_parallel = True
        else:
            print(f"📱 Using single GPU/CPU")
            self.is_parallel = False
            
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'], 
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.98)
        )
        
        self.ema = EMA(model, decay=0.995)
        
        steps_per_epoch = len(train_loader)
        
        warmup_steps = int(0.1 * steps_per_epoch)
        
        self.warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        self.cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=steps_per_epoch * 100 - warmup_steps)
        
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[warmup_steps]
        )
        self.loss_fn = InfoNCELoss()
        self.best_val_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        self.visualizer = LossVisualizer(self.history)
        
        self.model = self.model.to(self.device)
        self.model_save_path = os.path.join(self.config['output_path'], self.config['model_name'])
        os.makedirs(self.model_save_path, exist_ok=True)

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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                self.ema.update()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
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
