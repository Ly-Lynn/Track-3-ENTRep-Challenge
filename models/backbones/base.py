import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# Abstract Base Classes
class VisionEncoder(ABC, nn.Module):
    """Abstract base class for vision encoders"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized embeddings"""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimension of features before projection"""
        pass

    @abstractmethod
    def load_pretrained(self, model_path: str):
        print(f"🔍 Loading pretrained model from {model_path}")
        pass

class TextEncoder(ABC, nn.Module):
    """Abstract base class for text encoders"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized embeddings"""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimension of features before projection"""
        pass
    
    @abstractmethod
    def load_pretrained(self, model_path: str):
        print(f"🔍 Loading pretrained model from {model_path}")
        pass
