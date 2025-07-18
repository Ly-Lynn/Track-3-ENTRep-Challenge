import torch
import torch.nn as nn
from .base import VisionEncoder
from typing import Optional

class DinoV2VisionEncoder(VisionEncoder):
    """Complete DinoV2 model"""
    def __init__(self, 
                 model_name: str = 'dinov2_vitb14',
                 feature_dim: int = 768,
                 num_classes: int = 7,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False,
                 ckp_path: Optional[str] = None):
        super().__init__()
        
        # Initialize backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Initialize feature projection layers (formerly DinoV2Head)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone.num_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Store config
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        if ckp_path is not None:
            self.load_pretrained(ckp_path)
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Project features
        features = self.feature_projection(backbone_features)
        
        if return_features:
            return features
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)
    
    def get_backbone_features(self, x):
        """Get raw backbone features"""
        return self.backbone(x)

    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        checkpoint = torch.load(model_path, map_location='cpu')
            
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        self.backbone.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint loaded successfully")