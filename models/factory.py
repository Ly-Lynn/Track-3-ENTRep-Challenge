from .backbones.base import VisionEncoder, TextEncoder
from .backbones.dino import DinoV2VisionEncoder
from .backbones.clip import CLIPVisionEncoder, CLIPTextEncoder
from .backbones.bert import BERTTextEncoder
from .backbones.endovit import EndoViTVisionEncoder
from .backbones.endovit import OldEndoViTVisionEncoder

class VisionEncoderFactory:
    """Factory for creating vision encoders"""
    
    @staticmethod
    def create_encoder(encoder_type: str, **kwargs) -> VisionEncoder:
        if encoder_type.lower() == "clip":
            return CLIPVisionEncoder(**kwargs)
        elif encoder_type.lower() == "endovit":
            return EndoViTVisionEncoder(**kwargs)
        elif encoder_type.lower() == "dinov2":
            return DinoV2VisionEncoder(**kwargs)
        elif encoder_type.lower() == "old_endovit":
            return OldEndoViTVisionEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown vision encoder type: {encoder_type}")

class TextEncoderFactory:
    """Factory for creating text encoders"""
    
    @staticmethod
    def create_encoder(encoder_type: str, **kwargs) -> TextEncoder:
        if encoder_type.lower() == "clip":
            return CLIPTextEncoder(**kwargs)
        elif encoder_type.lower() == "bert":
            return BERTTextEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown text encoder type: {encoder_type}")


