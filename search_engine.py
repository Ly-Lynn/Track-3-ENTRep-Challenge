import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import MedicalDataset
from config import Config
import os
import pickle

class MedicalImageSearch:
    def __init__(self, model, data_frame, image_dir, load_dir=False, device=None):
        self.model = model
        self.data_frame = data_frame
        self.image_dir = image_dir
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        if os.path.exists(load_dir):
            self._load_index(load_dir)
        else:
            os.makedirs(Config.PUBLIC_DB_PATH, exist_ok=True)
            self.image_embeddings, self.image_paths = self._compute_image_embeddings()
            self._build_faiss_index()
            self.save_index(Config.PUBLIC_DB_PATH)
            print("Index saved to ", Config.PUBLIC_DB_PATH)
            self._load_index(Config.PUBLIC_DB_PATH)
        
    def save_index(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(save_dir, "vector.index"))
        
        with open(os.path.join(save_dir, "vector_metadata.pkl"), "wb") as f:
            pickle.dump({
                "image_paths": self.image_paths,
                "data_frame": self.data_frame
            }, f)

    def _load_index(self, load_dir):
        self.index = faiss.read_index(os.path.join(load_dir, "vector.index"))
        
        with open(os.path.join(load_dir, "vector_metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            self.image_paths = metadata["image_paths"]
            self.data_frame = metadata["data_frame"]
        
    def _compute_image_embeddings(self):
        dataset = MedicalDataset(
            self.data_frame,
            self.image_dir,
            transform=self.transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False
        )
        
        all_embeddings = []
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing embeddings"):
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(images, input_ids, attention_mask)
                image_embeds = outputs["image_embeds"]
                
                all_embeddings.append(image_embeds.cpu().numpy())
                all_paths.extend(batch["path"])
        
        all_embeddings = np.vstack(all_embeddings)
        
        return all_embeddings, all_paths
    
    def _build_faiss_index(self):
        self.image_embeddings = self.image_embeddings.astype('float32')
        faiss.normalize_L2(self.image_embeddings)
        
        d = self.image_embeddings.shape[1] 
        self.index = faiss.IndexFlatIP(d)   
        self.index.add(self.image_embeddings)
        
    def _get_text_embedding(self, query):
        text_inputs = self.model.processor(
            text=[query],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            text_embeds = text_outputs.pooler_output
            text_embeds = self.model.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds.cpu().numpy().astype('float32')
    
    def _get_image_embedding(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vision_outputs = self.model.model.vision_model(pixel_values=image)
            image_embeds = vision_outputs.pooler_output
            image_embeds = self.model.image_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        return image_embeds.cpu().numpy().astype('float32')
    
    def _search_with_embeddings(self, query_embedding, top_k=5):
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            path = self.image_paths[idx]
            # print("path: ", path)
            original_data = self.data_frame[self.data_frame['Path'] == path].iloc[0]
            # print("original_data: ", self.data_frame[self.data_frame['Path'] == path])
            results.append({
                'path': path,
                'similarity': float(sim),
                'classification': original_data['Classification'] if 'Classification' in original_data else None,
                'type': original_data['Type'] if 'Type' in original_data else None,
                'description': original_data['Description'] if 'Description' in original_data else None,
                'description_en': original_data['DescriptionEN'] if 'DescriptionEN' in original_data else None,
                'image': path  
            })
        
        return results
    
    def search_by_text(self, query, top_k=5):
        text_embedding = self._get_text_embedding(query)
        return self._search_with_embeddings(text_embedding, top_k)
    
    def search_by_image(self, image_path, top_k=5):
        image_embedding = self._get_image_embedding(image_path)
        return self._search_with_embeddings(image_embedding, top_k)