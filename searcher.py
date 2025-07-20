"""
Medical Image Searcher using FAISS for efficient similarity search
"""

# Fix OpenMP issue on macOS
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List, Dict, Any
from PIL import Image
from torchvision import transforms
import pickle
import faiss
import numpy as np
from utils import create_df_from_json
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MedicalDataset
from models.model import create_medical_vlm

class Searcher:
    def __init__(self, model=None, config=None):
        """
        Initialize the Searcher with a model and configuration
        
        Args:
            model: Pre-trained medical VLM model or None (will create from config)
            config: Configuration dictionary with model and data paths
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize CLIP processor for text encoding
        self.text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize or load model
        if model is None and config is not None:
            self.model = self._create_model_from_config(config)
        else:
            self.model = model
            
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        # Initialize transform for images (same as dataset)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        # Initialize storage variables
        self.index = None
        self.image_paths = None
        self.data_frame = None
        self.image_embeddings = None

    def _create_model_from_config(self, config):
        """Create model from configuration"""
        model_config = config["model"]
        model = create_medical_vlm(
            vision_encoder=model_config["vision_encoder"],
            text_encoder=model_config["text_encoder"],
            temperature=model_config.get("temperature", 0.07)
        )
        
        # Load pretrained weights if specified
        if "ckp_path" in model_config:
            model.load_from_path(model_config["ckp_path"])
            
        return model

    def build_index(self, save_dir=None):
        """
        Build the FAISS index from the dataset or load if already exists
        
        Args:
            save_dir: Optional directory to save the index after building
        """
        if save_dir and os.path.exists(save_dir):
            # Check if index files exist
            index_file = os.path.join(save_dir, "vector.index")
            metadata_file = os.path.join(save_dir, "vector_metadata.pkl")
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                try:
                    print(f"📁 Found existing index at {save_dir}, loading...")
                    self.load_index(save_dir)
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load existing index: {e}")
                    print("🔄 Rebuilding index...")
            else:
                print(f"📁 Index directory exists but files incomplete, rebuilding...")
        
        print("🔨 Building FAISS index...")
        self._build_faiss_index()
        
        if save_dir:
            self.save_index(save_dir)
            stats = self.get_stats()
            print(f"✅ Index built and saved to {save_dir}")
            print(f"📊 Index stats: {stats['total_images']} images, {stats['embedding_dimension']}D embeddings")

    def load_index(self, load_dir):
        """
        Load a pre-built FAISS index
        
        Args:
            load_dir: Directory containing the saved index
        """
        print(f"Loading FAISS index from {load_dir}")
        self._load_index(load_dir)
        
        # Validate loaded index
        stats = self.get_stats()
        if stats["status"] == "Index loaded" and stats["total_images"] > 0:
            print(f"✅ Index loaded successfully: {stats['total_images']} images, {stats['embedding_dimension']}D embeddings")
        else:
            raise ValueError("Failed to load index properly")

    def save_index(self, save_dir):
        """Save the current index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "vector.index"))
        
        # Save metadata
        with open(os.path.join(save_dir, "vector_metadata.pkl"), "wb") as f:
            pickle.dump({
                "image_paths": self.image_paths,
                "data_frame": self.data_frame
            }, f)
            
        print(f"Index saved to {save_dir}")

    def _load_index(self, load_dir):
        """Load FAISS index and metadata from disk"""
        # Load FAISS index
        index_path = os.path.join(load_dir, "vector.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(load_dir, "vector_metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.image_paths = metadata["image_paths"]
            self.data_frame = metadata["data_frame"]

    def _compute_embeddings(self):
        """Compute embeddings for all images in the dataset"""
        if self.config is None:
            raise ValueError("Configuration required to compute embeddings")
            
        # Load dataset - using test split for embeddings
        all_df, _, _, _ = create_df_from_json(
            self.config["evaluator"]["json_path"]
        )
        
        # Store dataframe for later use
        self.data_frame = all_df

        # Create dataset with proper path
        dataset = MedicalDataset(
            all_df,
            self.config["evaluator"]["img_dir"],
            transform=self.transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0 
        )
        
        all_image_embeddings = []
        # all_text_embeddings = []
        all_image_paths = []
        
        print("Computing embeddings...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing embeddings"):
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(images, input_ids, attention_mask)
                
                all_image_embeddings.append(outputs["image_embeds"].cpu())
                # all_text_embeddings.append(outputs["text_embeds"].cpu())
                all_image_paths.extend(batch["path"])
        
        # Store image paths
        self.image_paths = all_image_paths
        
        embedding_results = {
            "image_embeddings": torch.cat(all_image_embeddings, dim=0),
            # "text_embeddings": torch.cat(all_text_embeddings, dim=0),
            "image_paths": all_image_paths
        }
        
        return embedding_results

    def _build_faiss_index(self):
        """Build FAISS index from computed embeddings"""
        embedding_results = self._compute_embeddings()
        
        # Convert to numpy and normalize
        self.image_embeddings = embedding_results["image_embeddings"].numpy().astype('float32')
        faiss.normalize_L2(self.image_embeddings)
        
        # Create FAISS index
        d = self.image_embeddings.shape[1]  # dimension
        self.index = faiss.IndexFlatIP(d)   # Inner product index (cosine similarity after normalization)
        self.index.add(self.image_embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} embeddings")

    def _encode_text_query(self, query: str) -> np.ndarray:
        """
        Encode text query using the model's text encoder
        
        Args:
            query: Text query string
            
        Returns:
            Normalized text embedding as numpy array
        """
        # Use CLIP processor to tokenize text (same as dataset)
        text_inputs = self.text_processor(
            text=query,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            # Get text embedding from model
            text_embedding = self.model.text_encoder(input_ids, attention_mask)
            
            # Convert to numpy and normalize
            text_embedding = text_embedding.cpu().numpy().astype('float32')
            
        return text_embedding

    def _search_with_embeddings(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar images using embedding
        
        Args:
            query_embedding: Query embedding as numpy array
            top_k: Number of top results to return
            
        Returns:
            List of search results with metadata
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
            
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            path = self.image_paths[idx]
            
            # Find original data in dataframe
            original_data = self.data_frame[self.data_frame['Path'] == path].iloc[0]
            
            result = {
                'path': path,
                'similarity': float(sim),
                'classification': original_data.get('Classification', ''),
                'type': original_data.get('Type', ''),
                'description': original_data.get('Description', ''),
                'description_en': original_data.get('DescriptionEN', ''),
                'image_path': os.path.join(self.config["evaluator"]["img_dir"], path)
            }
            results.append(result)
        
        return results
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for images using text query
        
        Args:
            query: Text description to search for
            top_k: Number of top results to return
            
        Returns:
            List of search results with metadata
        """
        # Encode text query
        text_embedding = self._encode_text_query(query)
        
        # Search using embeddings
        return self._search_with_embeddings(text_embedding, top_k)


    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "No index loaded"}
            
        return {
            "status": "Index loaded",
            "total_images": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "index_type": type(self.index).__name__
        }

if __name__ == "__main__":
    # Example configuration
    config = {
        "evaluator": {
            "batch_size": 32,
            "image_size": 224,
            "img_dir": "Dataset/images/",
            "json_path": "Dataset/splits_info.json",
            "k_values": [1, 5, 10]
        },
        "model": {
            "vision_encoder": {
                "type": "endovit",
                "feature_dim": 768,
                "model_name": "egeozsoy/EndoViT",
                "ckp_path": "Pretrained/backbones/ent_vit/best_model.pth"
            },
            "text_encoder": {
                "type": "clip",
                "feature_dim": 768,
                "model_name": "openai/clip-vit-base-patch32",
            },
            "ckp_path": "Pretrained/checkpoints/endovit_clip/best.pt",
            "temperature": 0.07
        }
    }
    
    # Create searcher
    searcher = Searcher(config=config)
    
    # Build index
    searcher.build_index("test_index")
    
    # Test search
    results = searcher.search_by_text("edema and polypoid degeneration of the uncinate process", 10)
    print(f"Found {len(results)} results")
    print(results)