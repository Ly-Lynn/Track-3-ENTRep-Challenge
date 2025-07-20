from models.model import create_medical_vlm
from trainer import MedicalVLMTrainer
from utils import load_json, validate_json, create_df_from_json
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import argparse
import yaml
import os
from utils import create_df_from_json
from torchvision import transforms
from dataset import MedicalDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train a MedicalVLM model')
    parser.add_argument('--file_config', type=str, default='endovit.yaml', help='File config')
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_config(file_config):
    with open(os.path.join('Config/train', file_config), 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  
        # transforms.RandomRotation(15), 
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.1), 
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        # transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return train_transform, val_transform

def main():
    args = parse_args()
    set_seed()
    config = load_config(args.file_config)
    data_config = config['data']
    model_config = config['model']
    trainer_config = config['trainer']

    print("🔍 Data Config:")
    print(data_config)
    print("🔍 Model Config:")
    print(model_config)
    print("🔍 Trainer Config:")
    print(trainer_config)

    # Create dataset
    json_path = os.path.join(data_config['path'], data_config['json_path'])
    train_transform, val_transform = get_transforms()
    all_df, train_df, val_df, test_df = create_df_from_json(json_path)
    train_dataset = MedicalDataset(train_df, os.path.join(data_config['path'], "train"), train_transform)
    val_dataset = MedicalDataset(val_df, os.path.join(data_config['path'], "val"), val_transform)
    test_dataset = MedicalDataset(test_df, os.path.join(data_config['path'], "test"), val_transform)

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=trainer_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=trainer_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=trainer_config['batch_size'], shuffle=False)

    # Create model
    model = create_medical_vlm(**model_config)

    # Create trainer
    
    trainer = MedicalVLMTrainer(model, 
                                train_loader, 
                                val_loader, 
                                trainer_config)
    
    # Train model
    print("🔍 Training model...")
    history = trainer.train(
        num_epochs=trainer_config['num_epochs']
    )
    trainer.ema.apply_shadow()
    trainer.ema.restore()
    
    print("🔍 Training model done")
    print(f"🔍 History: {history}")

    # Test model
    print("🔍 Evaluating model...")
    eval_loss = trainer.evaluate(
        data_loader=test_loader
    )
    print(f"🔍 Evaluating model done, loss: {eval_loss:.4f}")

if __name__ == "__main__":
    main()