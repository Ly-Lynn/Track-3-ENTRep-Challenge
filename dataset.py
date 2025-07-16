import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor
import os

class MedicalDataset(Dataset):
    def __init__(self, data_frame, image_dir, transform=None, max_length=77):
        self.data_frame = data_frame
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        try:
            img_path = os.path.join(self.image_dir, self.data_frame.iloc[idx]['Classification'], self.data_frame.iloc[idx]['Path'])  
            description_en = self.data_frame.iloc[idx]['DescriptionEN']

            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            text_inputs = self.processor(
                text=description_en,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            assert text_inputs.input_ids.shape[1] == self.max_length, f"Text input_ids shape mismatch: {text_inputs.input_ids.shape}"
            result = {
                "image": image,
                "type": self.data_frame.iloc[idx]['Type'],
                "input_ids": text_inputs.input_ids.squeeze(0),
                "attention_mask": text_inputs.attention_mask.squeeze(0),
                "path": self.data_frame.iloc[idx]['Path'],
                "classification": self.data_frame.iloc[idx]['Classification'],
                "description": description_en
            }
            return result
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if idx != 0:
                return self.__getitem__(0)
            raise e
        