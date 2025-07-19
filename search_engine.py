import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import MedicalDataset
from transformers import CLIPProcessor
import os
import pickle

class MedicalImageSearch:
    def __init__(self, model, data_frame, image_dir, load_dir=None, save_dir="search_db"):
        self.model = model
        self.data_frame = data_frame
        self.image_dir = image_dir
        self.save_dir = save_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        if load_dir and os.path.exists(load_dir):
            print(f"Loading existing index from {load_dir}")
            self._load_index(load_dir)
        else:
            print("Building new index...")
            os.makedirs(self.save_dir, exist_ok=True)
            self.image_embeddings, self.image_paths = self._compute_image_embeddings()
            self._build_faiss_index()
            self.save_index(self.save_dir)
            print(f"Index saved to {self.save_dir}")
        
    def save_index(self, save_dir):
        """Save FAISS index and metadata to directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(save_dir, "vector.index"))
        
        with open(os.path.join(save_dir, "vector_metadata.pkl"), "wb") as f:
            pickle.dump({
                "image_paths": self.image_paths,
                "data_frame": self.data_frame
            }, f)

    def _load_index(self, load_dir):
        """Load FAISS index and metadata from directory"""
        self.index = faiss.read_index(os.path.join(load_dir, "vector.index"))
        
        with open(os.path.join(load_dir, "vector_metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            self.image_paths = metadata["image_paths"]
            self.data_frame = metadata["data_frame"]
        
    def _compute_image_embeddings(self):
        """Compute embeddings for all images in the dataset"""
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
        """Build FAISS index from image embeddings"""
        self.image_embeddings = self.image_embeddings.astype('float32')
        faiss.normalize_L2(self.image_embeddings)
        
        d = self.image_embeddings.shape[1] 
        self.index = faiss.IndexFlatIP(d)   # Inner product for cosine similarity
        self.index.add(self.image_embeddings)
        
    def _get_text_embedding(self, query):
        """Get text embedding for search query"""
        text_inputs = self.processor(
            text=[query],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            text_embeds = self.model.text_encoder(input_ids, attention_mask)
            # text_embeds is already normalized from the text encoder
        
        return text_embeds.cpu().numpy().astype('float32')
    
    def _get_image_embedding(self, image_path):
        """Get image embedding for search by image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use vision encoder directly to get features
            image_embeds = self.model.vision_encoder(image_tensor, return_features=True)
            # Normalize the embeddings to match training
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        
        return image_embeds.cpu().numpy().astype('float32')
    
    def _search_with_embeddings(self, query_embedding, top_k=5):
        """Search with precomputed embeddings"""
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.index.search(query_embedding, top_k)
        results = []
        
        for idx, sim in zip(indices[0], similarities[0]):
            path = self.image_paths[idx]
            # Find matching row in dataframe
            matching_rows = self.data_frame[self.data_frame['Path'] == path]
            
            if len(matching_rows) > 0:
                original_data = matching_rows.iloc[0]
                results.append({
                    'path': path,
                    'similarity': float(sim),
                    'classification': original_data.get('Classification', None),
                    'type': original_data.get('Type', None),
                    'description': original_data.get('Description', None),
                    'description_en': original_data.get('DescriptionEN', None),
                    'image': path  
                })
        
        return results
    
    def search_by_text(self, query, top_k=5):
        """Search images by text query"""
        text_embedding = self._get_text_embedding(query)
        return self._search_with_embeddings(text_embedding, top_k)
    
    def search_by_image(self, image_path, top_k=5):
        """Search images by image similarity"""
        image_embedding = self._get_image_embedding(image_path)
        return self._search_with_embeddings(image_embedding, top_k)