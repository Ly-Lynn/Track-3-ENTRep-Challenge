import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from .base import TextEncoder, VisionEncoder
from typing import Optional
class CLIPTextEncoder(TextEncoder):
    """CLIP Text Encoder"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 feature_dim: int = 768, dropout_rate: float = 0.3, ckp_path: Optional[str] = None):
        super().__init__(feature_dim)
        self.feature_dim = feature_dim
        self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.text_model = self.clip_model.text_model
        
        # Batch normalization and dropout
        self.bn = nn.BatchNorm1d(self.get_feature_dim())
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.get_feature_dim(), self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        if ckp_path is not None:
            self.load_pretrained(ckp_path)

    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        self.text_model.load_state_dict(torch.load(model_path))

    def get_feature_dim(self) -> int:
        return self.text_model.config.hidden_size
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = text_outputs.pooler_output
        embeddings = self.bn(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.projection(embeddings)
        if return_features:
            return embeddings
        return F.normalize(embeddings, dim=-1)

# Vision Encoder Implementations
class CLIPVisionEncoder(VisionEncoder):
    """CLIP Vision Encoder"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 feature_dim: int = 768, dropout_rate: float = 0.3, ckp_path: Optional[str] = None):
        super().__init__(feature_dim)
        
        self.clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.vision_model = self.clip_model.vision_model
        
        # Batch normalization and dropout
        self.bn = nn.BatchNorm1d(self.get_feature_dim())
        self.dropout = nn.Dropout(dropout_rate)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.get_feature_dim(), self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        if ckp_path is not None:
            self.load_pretrained(ckp_path)

    def get_feature_dim(self) -> int:
        return self.vision_model.config.hidden_size
    
    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=images)
        embeddings = vision_outputs.pooler_output
        embeddings = self.bn(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.projection(embeddings)
        if return_features:
            return embeddings
        return F.normalize(embeddings, dim=-1)

    
    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        self.clip_model.load_state_dict(torch.load(model_path))
