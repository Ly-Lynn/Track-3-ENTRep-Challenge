import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from constants import (
    CLIP_MEAN, CLIP_STD, DEFAULT_BATCH_SIZE, DEFAULT_SEED, 
    IMAGE_SIZE, CLIP_MODEL_NAME
)
from dataset import MedicalDataset
from models.model import create_medical_vlm
from trainer import MedicalVLMTrainer
from utils import load_json, validate_json, create_df_from_json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a MedicalVLM model')
    parser.add_argument('--file_config', type=str, default='endovit.yaml', help='File config')
    return parser.parse_args()

def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_config(file_config: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        file_config: Configuration file name
        
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join('Config/train', file_config)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")

def get_transforms() -> tuple:
    """
    Get image transforms for training and validation.
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        # TODO: Add data augmentation for better training
        # transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),  
        # transforms.RandomRotation(15), 
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.1), 
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        # transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    
    return train_transform, val_transform

def create_datasets(data_config: dict) -> tuple:
    """
    Create train, validation and test datasets.
    
    Args:
        data_config: Data configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    json_path = os.path.join(data_config['path'], data_config['json_path'])
    train_transform, val_transform = get_transforms()
    
    # Load data splits
    all_df, train_df, val_df, test_df = create_df_from_json(json_path)
    
    # Create datasets
    train_dataset = MedicalDataset(
        train_df, 
        os.path.join(data_config['path'], "train"), 
        train_transform
    )
    val_dataset = MedicalDataset(
        val_df, 
        os.path.join(data_config['path'], "val"), 
        val_transform
    )
    test_dataset = MedicalDataset(
        test_df, 
        os.path.join(data_config['path'], "test"), 
        val_transform
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(datasets: tuple, batch_size: int) -> tuple:
    """
    Create data loaders from datasets.
    
    Args:
        datasets: Tuple of (train_dataset, val_dataset, test_dataset)
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = datasets
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def print_configs(data_config: dict, model_config: dict, trainer_config: dict) -> None:
    """Print configuration information."""
    print("🔍 Data Config:")
    print(data_config)
    print("🔍 Model Config:")
    print(model_config)
    print("🔍 Trainer Config:")
    print(trainer_config)


def main():
    """Main training function."""
    # Parse arguments and setup
    args = parse_args()
    set_seed()
    config = load_config(args.file_config)
    
    # Extract configurations
    data_config = config['data']
    model_config = config['model']
    trainer_config = config['trainer']
    
    print_configs(data_config, model_config, trainer_config)

    # Create datasets and data loaders
    datasets = create_datasets(data_config)
    train_loader, val_loader, test_loader = create_dataloaders(
        datasets, 
        trainer_config['batch_size']
    )

    # Create model and trainer
    model = create_medical_vlm(**model_config)
    trainer = MedicalVLMTrainer(
        model, 
        train_loader, 
        val_loader, 
        trainer_config
    )
    
    # Train model
    print("🔍 Training model...")
    history = trainer.train(num_epochs=trainer_config['num_epochs'])
    
    # Apply EMA weights
    trainer.ema.apply_shadow()
    trainer.ema.restore()
    
    print("🔍 Training completed successfully")
    print(f"🔍 Training history: {history}")

    # Evaluate model
    print("🔍 Evaluating model on test set...")
    eval_loss = trainer.evaluate(data_loader=test_loader)
    print(f"🔍 Evaluation completed - Test loss: {eval_loss:.4f}")

if __name__ == "__main__":
    main()