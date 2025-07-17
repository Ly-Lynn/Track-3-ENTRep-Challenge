import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from factory import VisionEncoderFactory, TextEncoderFactory
import os

class MedicalVLM(nn.Module):
    """Flexible Medical Vision-Language Model with swappable encoders"""
    
    def __init__(self, 
                 vision_encoder_type: str = "clip",
                 text_encoder_type: str = "clip",
                 temperature: float = 0.07,
                 vision_encoder_kwargs: Optional[Dict[str, Any]] = None,
                 text_encoder_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.temperature = temperature
        
        # Set default kwargs
        vision_encoder_kwargs = vision_encoder_kwargs or {}
        text_encoder_kwargs = text_encoder_kwargs or {}
        
        # Create encoders
        self.vision_encoder = VisionEncoderFactory.create_encoder(
            vision_encoder_type, **vision_encoder_kwargs
        )
        self.text_encoder = TextEncoderFactory.create_encoder(
            text_encoder_type, **text_encoder_kwargs
        )
    
    def forward(self, images: torch.Tensor, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Get main embeddings (ensure we get features, not logits)
        image_embeds = self.vision_encoder(images, return_features=True)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        
        result = {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds
        }
        
        return result
    def load_from_path(self, model_path: str):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        model_dict = self.state_dict()
        filtered_checkpoint = {}
        
        for k, v in checkpoint.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_checkpoint[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch: {model_dict[k].shape} vs {v.shape}")
            else:
                print(f"Skipping unexpected key: {k}")
        
        self.load_state_dict(filtered_checkpoint, strict=False)
        print(f"Loaded {len(filtered_checkpoint)}/{len(checkpoint)} parameters from {model_path}")
        return self

def create_medical_vlm(vision_encoder: Dict[str, Any], 
                      text_encoder: Dict[str, Any],
                      temperature: float = 0.07,
                      **kwargs) -> MedicalVLM:
    """
    Convenience function to create a MedicalVLM from config format
    
    Args:
        vision_encoder: Dict with 'type' and encoder parameters
        text_encoder: Dict with 'type' and encoder parameters  
        temperature: Temperature for contrastive learning
        **kwargs: Additional parameters for MedicalVLM
    
    Example:
        model = create_medical_vlm(
            vision_encoder={'type': 'endovit', 'feature_dim': 768, 'model_name': 'egeozsoy/EndoViT'},
            text_encoder={'type': 'clip', 'feature_dim': 768, 'model_name': 'openai/clip-vit-base-patch32'},
            temperature=0.07
        )
    """
    # Extract type and use remaining as kwargs
    vision_encoder_config = vision_encoder.copy()
    vision_encoder_type = vision_encoder_config.pop('type')
    
    text_encoder_config = text_encoder.copy()  
    text_encoder_type = text_encoder_config.pop('type')
    
    return MedicalVLM(
        vision_encoder_type=vision_encoder_type,
        text_encoder_type=text_encoder_type,
        temperature=temperature,
        vision_encoder_kwargs=vision_encoder_config,
        text_encoder_kwargs=text_encoder_config,
        **kwargs
    )

def debug_state_dict_mismatch(model_path, model_class):
    """Debug function for checking state dict mismatches"""
    print(f"\nDebugging state_dict for {model_class.__name__}")
    
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_keys = set(checkpoint.keys())
    
    model = model_class()
    
    model_keys = set(model.state_dict().keys())
    
    print(f"Checkpoint keys: {len(checkpoint_keys)}")
    print(f"Model keys: {len(model_keys)}")
    
    only_in_checkpoint = checkpoint_keys - model_keys
    if only_in_checkpoint:
        print(f"\nKeys only in checkpoint ({len(only_in_checkpoint)}):")
        for key in sorted(only_in_checkpoint):
            print(f"  - {key}")
    
    only_in_model = model_keys - checkpoint_keys
    if only_in_model:
        print(f"\nKeys only in model ({len(only_in_model)}):")
        for key in sorted(only_in_model):
            print(f"  - {key}")
    
    common_keys = checkpoint_keys & model_keys
    shape_mismatches = []
    for key in common_keys:
        if checkpoint[key].shape != model.state_dict()[key].shape:
            shape_mismatches.append((key, checkpoint[key].shape, model.state_dict()[key].shape))
    
    if shape_mismatches:
        print(f"\nShape mismatches ({len(shape_mismatches)}):")
        for key, ckpt_shape, model_shape in shape_mismatches:
            print(f"  - {key}: {ckpt_shape} vs {model_shape}")
    
    print(f"\nMatching keys: {len(common_keys) - len(shape_mismatches)}")


if __name__ == "__main__":
    def create_test_data():
        batch_size = 2
        seq_length = 77
        vocab_size = 1000

        test_inputs = {
            "images": torch.randn(batch_size, 3, 224, 224),
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
        }
        return test_inputs

    # # Test different configurations
    # print("Testing MedicalVLM with different configurations...")
    
    # # Test 1: CLIP vision + CLIP text
    # print("\n1. Testing CLIP vision + CLIP text:")
    # model1 = create_medical_vlm(vision_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"}, text_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"})
    # model1.eval()
    
    # # Test 2: EndoViT vision + CLIP text
    # print("\n2. Testing EndoViT vision + CLIP text:")
    # model2 = create_medical_vlm(vision_encoder={"type": "endovit", "model_name": "egeozsoy/EndoViT"}, text_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"})
    # model2.eval()
    
    # # Test 3: CLIP vision + BERT text
    # print("\n3. Testing CLIP vision + BERT text:")
    # model3 = create_medical_vlm(vision_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"}, text_encoder={"type": "bert", "model_name": "bert-base-uncased"})
    # model3.eval()
    
    # # Test 4: DINOv2 vision + CLIP text
    # print("\n4. Testing DINOv2 vision + CLIP text:")
    # model4 = create_medical_vlm(vision_encoder={"type": "dinov2", "model_name": "dinov2_vitb14"}, text_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"})
    # model4.eval()

    test_inputs = create_test_data()
    
    # # Test forward pass for each model
    # for i, model in enumerate([model1, model2, model3, model4], 1):
    #     print(f"\nTesting Model {i}:")
    #     with torch.no_grad():
    #         try:
    #             outputs = model(
    #                 images=test_inputs["images"],
    #                 input_ids=test_inputs["input_ids"],
    #                 attention_mask=test_inputs["attention_mask"]
    #             )
    #             print(f"  Image embeds: {outputs['image_embeds'].shape}")
    #             print(f"  Text embeds: {outputs['text_embeds'].shape}")
    #             print(f"  Image embeds norm: {outputs['image_embeds'].norm(dim=-1).mean():.4f}")
    #             print(f"  Text embeds norm: {outputs['text_embeds'].norm(dim=-1).mean():.4f}")
    #         except Exception as e:
    #             print(f"  Error: {e}")

    # Test pretrained model loading
    print("\n" + "="*60)
    print("TESTING PRETRAINED MODEL LOADING")
    print("="*60)
    
    def test_pretrained_loading():
        """Test loading pretrained models from Pretrained folder"""
        import os
        import glob
        
        pretrained_folder = "Pretrained"
        
        pretrained_dinob = '/Users/thuylinh.lynguyen/Documents/Code/Track-3-ENTRep-Challenge/Pretrained/dinov2_vitb14/best_model.pth'
        pretrained_dinos = '/Users/thuylinh.lynguyen/Documents/Code/Track-3-ENTRep-Challenge/Pretrained/dinov2_vits14/best_model.pth'
        pretrained_dinol = '/Users/thuylinh.lynguyen/Documents/Code/Track-3-ENTRep-Challenge/Pretrained/dinov2_vitl14/best_model-001.pth'
        pretrained_endovit = '/Users/thuylinh.lynguyen/Documents/Code/Track-3-ENTRep-Challenge/Pretrained/ent_vit/best_model.pth'

        
        
        # Test configurations that might match pretrained models
        test_configs = [
            {
                "name": "EndoViT-CLIP", 
                "vision_encoder": {"type": "endovit", 
                "model_name": "egeozsoy/EndoViT",
                "ckp_path": pretrained_endovit},
                "text_encoder": {"type": "clip", "model_name": "openai/clip-vit-base-patch32"}
            },
            {
                "name": "DINOv2s-CLIP",
                "vision_encoder": {"type": "dinov2", 
                "model_name": "dinov2_vits14",
                "ckp_path": pretrained_dinos},
                "text_encoder": {"type": "clip", "model_name": "openai/clip-vit-base-patch32"}
            },
            {
                "name": "DINOv2b-CLIP",
                "vision_encoder": {"type": "dinov2", 
                "model_name": "dinov2_vitb14",
                "ckp_path": pretrained_dinob},
                "text_encoder": {"type": "clip", "model_name": "openai/clip-vit-base-patch32"}
            },
            {
                "name": "DINOv2l-CLIP",
                "vision_encoder": {"type": "dinov2", 
                "model_name": "dinov2_vitl14",
                "ckp_path": pretrained_dinol},
                "text_encoder": {"type": "clip", "model_name": "openai/clip-vit-base-patch32"}
            }

        ]
        
        for config in test_configs:
            print(f"\n📋 Testing {config['name']} configuration:")
            
            try:
                # Create model with config
                model = create_medical_vlm(
                    vision_encoder=config['vision_encoder'],
                    text_encoder=config['text_encoder']
                )
                print(f"  ✅ Model created successfully")
                
                # Try loading each pretrained file
                for pretrained_file in [config['vision_encoder']['ckp_path']]:
                    try:
                        print(f"  📂 Attempting to load: {os.path.basename(pretrained_file)}")
                        
                        # Load checkpoint
                        checkpoint = torch.load(pretrained_file, map_location='cpu')
                        
                        # Try to load state dict
                        if hasattr(model, 'load_checkpoint'):
                            # Use model's load_checkpoint method if available
                            model.load_from_path(pretrained_file)
                            print(f"    ✅ Loaded using load_checkpoint method")
                        else:
                            # Try direct state dict loading
                            model.load_state_dict(checkpoint, strict=False)
                            print(f"    ✅ Loaded using state_dict (strict=False)")
                        
                        # Test forward pass with loaded model
                        model.eval()
                        with torch.no_grad():
                            test_outputs = model(
                                images=test_inputs["images"],
                                input_ids=test_inputs["input_ids"], 
                                attention_mask=test_inputs["attention_mask"]
                            )
                            print(f"    🔄 Forward pass successful:")
                            print(f"       Image embeds: {test_outputs['image_embeds'].shape}")
                            print(f"       Text embeds: {test_outputs['text_embeds'].shape}")
                            
                    except Exception as e:
                        print(f"    ❌ Failed to load {os.path.basename(pretrained_file)}: {str(e)[:100]}...")
                        
            except Exception as e:
                print(f"  ❌ Failed to create {config['name']} model: {e}")
    
    def test_checkpoint_compatibility():
        """Test checkpoint loading with different model configurations"""
        print(f"\n🔧 Testing checkpoint compatibility...")
        
        try:
            # Create a simple model and save a checkpoint
            temp_model = create_medical_vlm(
                vision_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"},
                text_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"}
            )
            
            # Save temporary checkpoint
            temp_checkpoint_path = "temp_test_checkpoint.pt"
            torch.save(temp_model.state_dict(), temp_checkpoint_path)
            print(f"  💾 Created temporary checkpoint: {temp_checkpoint_path}")
            
            # Test loading into same model type
            new_model = create_medical_vlm(
                vision_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"},
                text_encoder={"type": "clip", "model_name": "openai/clip-vit-base-patch32"}
            )
            
            new_model.load_state_dict(torch.load(temp_checkpoint_path, map_location='cpu'))
            print(f"  ✅ Successfully loaded checkpoint into same model type")
            
            # Test forward pass
            new_model.eval()
            with torch.no_grad():
                outputs = new_model(
                    images=test_inputs["images"],
                    input_ids=test_inputs["input_ids"],
                    attention_mask=test_inputs["attention_mask"]
                )
                print(f"  🔄 Forward pass after loading successful")
            
            # Clean up
            os.remove(temp_checkpoint_path)
            print(f"  🗑️  Cleaned up temporary checkpoint")
            
        except Exception as e:
            print(f"  ❌ Checkpoint compatibility test failed: {e}")
    
    # Run pretrained loading tests
    test_pretrained_loading()
    test_checkpoint_compatibility()
    
    print(f"\n🎉 All tests completed!")

