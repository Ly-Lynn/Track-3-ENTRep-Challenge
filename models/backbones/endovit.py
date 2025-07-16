import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from .base import VisionEncoder
from pathlib import Path
from functools import partial
from huggingface_hub import snapshot_download
from typing import Optional
class EndoViTVisionEncoder(VisionEncoder):
    """Complete EndoViT Vision Encoder with integrated backbone and head"""
    
    def __init__(self, 
                 model_name: str = "egeozsoy/EndoViT",
                 feature_dim: int = 768,
                 num_classes: int = 7,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False,
                 ckp_path: Optional[str] = None):
        super().__init__(feature_dim)
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Initialize backbone with EndoViT
        self._init_backbone(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Initialize feature projection layers (head functionality)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.backbone_feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        if ckp_path is not None:
            self.load_pretrained(ckp_path)

    def _init_backbone(self, model_name: str):
        """Initialize the backbone model"""
        model_path = snapshot_download(repo_id=model_name, revision="main")
        model_weights_path = Path(model_path) / "pytorch_model.bin"
        
        if model_weights_path.exists():
            self.backbone = VisionTransformer(
                patch_size=16, 
                embed_dim=self.feature_dim, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4, 
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ).eval()
            
            model_weights = torch.load(model_weights_path, map_location='cpu', weights_only=False)
            if 'model' in model_weights:
                model_weights = model_weights['model']
                
            loading_info = self.backbone.load_state_dict(model_weights, strict=False)
            self.backbone_feature_dim = self.feature_dim
            print(f"✅ Successfully loaded pretrained EndoViT: {loading_info}")
        else:
            raise FileNotFoundError("EndoViT weights not found")
            
    def load_pretrained(self, model_path: str):
        """Load pretrained model from path"""
        self.backbone.load_state_dict(torch.load(model_path))

    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        return self.feature_dim
    
    def _extract_backbone_features(self, x):
        """Extract features from backbone"""
        try:
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
                if len(features.shape) == 3:  # [batch, seq_len, embed_dim]
                    return features[:, 0]  # CLS token
                else:
                    return features
            else:
                return self.backbone(x)
        except Exception as e:
            print(f"⚠️ Error in backbone forward pass: {e}")
            return self.backbone(x)
    
    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass"""
        x = images
        backbone_features = self._extract_backbone_features(x)
        
        features = self.feature_projection(backbone_features)
        
        if return_features:
            return F.normalize(features, dim=-1)
        
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)
    
    def get_backbone_features(self, x):
        """Get raw backbone features"""
        return self._extract_backbone_features(x)