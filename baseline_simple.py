import json
import torch
import numpy as np
import faiss
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from PIL import Image
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from metrics import Metrics

class MedCLIPSearcher:
    def __init__(self, img_dir: str = "Dataset/images/"):
        """
        Initialize MedCLIP searcher with auto-downloaded model
        
        Args:
            img_dir: Directory containing images
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MedCLIP with auto-download from HuggingFace
        print("Initializing MedCLIP model (auto-downloading if needed)...")
        try:
            self.processor = MedCLIPProcessor()
            self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            # Try to load from HuggingFace Hub instead of local
            self.model = MedCLIPModel.from_pretrained("flaviagiammarino/medclip-vit")
            print("✅ Model loaded from HuggingFace Hub")
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Using initialized model without pretrained weights...")
            self.processor = MedCLIPProcessor()
            self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.img_dir = img_dir
        self.index = None
        self.image_paths = []
        self.image_data = []
    
    def load_test_data(self, json_path: str) -> List[Dict]:
        """Load test data from splits_info.json"""
        print(f"Loading test data from {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_data = data.get('test', [])
        print(f"Loaded {len(test_data)} test samples")
        
        return test_data
    
    def build_image_index(self, test_data: List[Dict], max_samples: int = 500):
        """
        Build FAISS index from test images (limit samples for testing)
        
        Args:
            test_data: List of test data dictionaries
            max_samples: Maximum number of samples to process (for testing)
        """
        print(f"Building image embeddings and FAISS index (max {max_samples} samples)...")
        
        # Limit data for testing
        test_data = test_data[:max_samples]
        
        # Store image data
        self.image_data = test_data
        self.image_paths = [item['Path'] for item in test_data]
        
        # Compute image embeddings
        all_embeddings = []
        batch_size = 16  # Smaller batch size for stability
        
        for i in tqdm(range(0, len(test_data), batch_size), desc="Computing image embeddings"):
            batch_data = test_data[i:i+batch_size]
            try:
                batch_embeddings = self._compute_batch_image_embeddings(batch_data)
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
        
        if not all_embeddings:
            raise RuntimeError("No embeddings computed successfully")
        
        # Concatenate all embeddings
        self.image_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy().astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.image_embeddings)
        
        # Build FAISS index
        d = self.image_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        self.index.add(self.image_embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} images, embedding dim: {d}")
    
    def _compute_batch_image_embeddings(self, batch_data: List[Dict]) -> torch.Tensor:
        """Compute embeddings for a batch of images"""
        images = []
        valid_indices = []
        
        for idx, item in enumerate(batch_data):
            img_path = os.path.join(self.img_dir, item['Path'])
            try:
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    images.append(image)
                    valid_indices.append(idx)
                else:
                    print(f"Image not found: {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        if not images:
            # Return dummy embedding if no valid images
            dummy_size = getattr(self.model, 'projection_dim', 512)
            return torch.zeros(1, dummy_size)
        
        # Process images
        try:
            inputs = self.processor(
                text=[""],  # Dummy text, we only need image embeddings
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                img_embeds = outputs['img_embeds']
            
            return img_embeds
            
        except Exception as e:
            print(f"Error in model inference: {e}")
            # Return dummy embedding on error
            dummy_size = getattr(self.model, 'projection_dim', 512)
            return torch.zeros(len(images), dummy_size)
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar images using text query
        
        Args:
            query: Text description to search for
            top_k: Number of top results to return
            
        Returns:
            List of search results with metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_image_index() first.")
        
        # Encode text query
        text_embedding = self._encode_text_query(query)
        
        # Search
        similarities, indices = self.index.search(text_embedding, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < len(self.image_data):
                item_data = self.image_data[idx]
                result = {
                    'image_path': os.path.join(self.img_dir, item_data['Path']),
                    'path': item_data['Path'],
                    'similarity': float(sim),
                    'classification': item_data.get('Classification', ''),
                    'type': item_data.get('Type', ''),
                    'description': item_data.get('Description', ''),
                    'description_en': item_data.get('DescriptionEN', '')
                }
                results.append(result)
        
        return results
    
    def _encode_text_query(self, query: str) -> np.ndarray:
        """Encode text query using MedCLIP"""
        try:
            inputs = self.processor(
                text=[query],
                images=Image.new('RGB', (224, 224), color='white'),  # Dummy image
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_embedding = outputs['text_embeds'].cpu().numpy().astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(text_embedding)
            
            return text_embedding
            
        except Exception as e:
            print(f"Error encoding text query: {e}")
            # Return dummy embedding on error
            dummy_size = getattr(self.model, 'projection_dim', 512)
            dummy_embedding = np.random.randn(1, dummy_size).astype('float32')
            faiss.normalize_L2(dummy_embedding)
            return dummy_embedding

def create_test_queries(test_data: List[Dict], min_images: int = 2) -> List[Tuple[str, List[str]]]:
    """
    Create test queries from test data
    Each query will search for images with the same description
    
    Args:
        test_data: List of test data
        min_images: Minimum number of images per description to create a query
        
    Returns:
        List of (query_text, relevant_image_paths) tuples
    """
    print("Creating test queries...")
    
    # Group by English description to create queries
    desc_to_images = {}
    for item in test_data:
        desc_en = item.get('DescriptionEN', '').strip()
        if desc_en and len(desc_en) > 10:  # Filter out very short descriptions
            if desc_en not in desc_to_images:
                desc_to_images[desc_en] = []
            desc_to_images[desc_en].append(item['Path'])
    
    # Create queries (only keep descriptions with multiple images for meaningful evaluation)
    test_queries = []
    for desc, img_paths in desc_to_images.items():
        if len(img_paths) >= min_images:  # At least min_images with same description
            test_queries.append((desc, img_paths))
    
    print(f"Created {len(test_queries)} test queries from {len(desc_to_images)} unique descriptions")
    return test_queries

def main():
    # Configuration
    IMG_DIR = "Dataset/images/"
    JSON_PATH = "Dataset/splits_info.json"
    K_VALUES = [1, 3, 5, 10]
    MAX_SAMPLES = 300  # Limit for testing
    
    print("="*60)
    print("MedCLIP Baseline Evaluation (Simple Version)")
    print("="*60)
    
    # Initialize searcher
    searcher = MedCLIPSearcher(img_dir=IMG_DIR)
    
    # Load test data
    test_data = searcher.load_test_data(JSON_PATH)
    
    # Build image index
    searcher.build_image_index(test_data, max_samples=MAX_SAMPLES)
    
    # Create test queries from the same data that was indexed
    indexed_data = searcher.image_data
    test_queries = create_test_queries(indexed_data, min_images=2)
    
    if len(test_queries) == 0:
        print("No valid test queries created. Each query needs multiple relevant images.")
        return
    
    # Initialize metrics
    metrics = Metrics.from_test_set_tuples(test_queries)
    
    # Perform searches
    print(f"\nPerforming searches for {len(test_queries)} queries...")
    search_results = []
    all_results = []
    
    max_k = max(K_VALUES)
    
    for query_text, relevant_paths in tqdm(test_queries, desc="Searching"):
        try:
            results = searcher.search_by_text(query_text, max_k)
            search_results.append(results)
            
            all_results.append({
                "query": query_text,
                "relevant_paths": relevant_paths,
                "results": results
            })
        except Exception as e:
            print(f"Error searching for query '{query_text[:50]}...': {e}")
            search_results.append([])  # Empty results for failed query
    
    # Save search results
    results_file = "search_results_medclip_baseline_simple.json"
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Search results saved to: {results_file}")
    
    # Compute metrics
    print(f"\nComputing metrics for k values: {K_VALUES}")
    try:
        final_metrics = metrics.compute_all_metrics(search_results, K_VALUES)
        
        # Print results
        print("\n" + "="*60)
        print("MEDCLIP BASELINE RESULTS (Simple Version)")
        print("="*60)
        
        for k in K_VALUES:
            print(f"\nResults for k={k}:")
            print(f"  Hit Rate@{k}: {final_metrics['hit_rate'][k]:.3f}")
            print(f"  Recall@{k}: {final_metrics['recall'][k]:.3f}")
            print(f"  MRR@{k}: {final_metrics['mrr'][k]:.3f}")
        
        # Save metrics
        metrics_file = "medclip_baseline_simple_metrics.txt"
        metrics.save_metrics(final_metrics, K_VALUES, metrics_file)
        print(f"\nMetrics saved to: {metrics_file}")
        
        print(f"\n📊 Summary:")
        print(f"  Processed {len(indexed_data)} images")
        print(f"  Created {len(test_queries)} test queries")
        print(f"  Average Hit Rate@5: {final_metrics['hit_rate'][5]:.3f}")
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 