import os
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from constants import CLIP_MAX_LENGTH, CLIP_MODEL_NAME


class MedicalDataset(Dataset):
    """Dataset class for medical images with text descriptions."""
    
    def __init__(
        self, 
        data_frame, 
        image_dir: str, 
        transform: Optional[callable] = None, 
        max_length: int = CLIP_MAX_LENGTH
    ):
        """
        Initialize the medical dataset.
        
        Args:
            data_frame: DataFrame containing image metadata
            image_dir: Directory containing images
            transform: Optional image transformations
            max_length: Maximum text length for tokenization
        """
        self.data_frame = data_frame
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data_frame)
        
    def __getitem__(self, idx: Union[int, torch.Tensor]) -> dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image, text, and metadata
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        try:
            row = self.data_frame.iloc[idx]
            img_path = os.path.join(
                self.image_dir, 
                row['Classification'], 
                row['Path']
            )
            description_en = row['DescriptionEN']

            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Process text
            text_inputs = self._process_text(description_en)
            
            return {
                "image": image,
                "type": row['Type'],
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0),
                "path": row['Path'],
                "classification": row['Classification'],
                "description": description_en
            }
            
        except Exception as e:
            print(f"Error loading image at index {idx} (path: {img_path if 'img_path' in locals() else 'unknown'}): {e}")
            # Fallback to first item, but prevent infinite recursion
            if idx != 0:
                return self.__getitem__(0)
            raise RuntimeError(f"Failed to load dataset item at index {idx}") from e
    
    def _process_text(self, text: str) -> torch.Tensor:
        """
        Process text using CLIP processor.
        
        Args:
            text: Input text description
            
        Returns:
            Processed text tensors
        """
        text_inputs = self.processor(
            text=text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Validate text length
        if text_inputs.input_ids.shape[1] != self.max_length:
            raise ValueError(
                f"Text input_ids shape mismatch: expected {self.max_length}, "
                f"got {text_inputs.input_ids.shape[1]}"
            )
        
        return text_inputs
        