"""
Text-to-Image Search Evaluator for Medical Vision-Language Models
Fixed version with proper embedding extraction.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor


class MedVLMEvaluator:
    """
    Evaluator for text-to-image medical search systems.
    Fixed version with proper embedding extraction.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config['batch_size']
        self.image_size = (config['image_size'], config['image_size'])
        self.img_dir = config['img_dir']
        self.test_df = self.load_testset_from_json(config['json_path'])
        # Check model structure
        if not (hasattr(model, 'vision_encoder') and hasattr(model, 'text_encoder')):
            print("Warning: Model should have 'vision_encoder' and 'text_encoder' attributes")
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_testset_from_json(self, json_path: Union[str, Path]) -> pd.DataFrame:
        """Load test set from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['test'])
        return df
    
    def load_and_preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.image_transform(image)
            # Ensure we return a proper tensor
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Image transform did not return a tensor, got {type(tensor)}")
            return tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    def extract_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Extract text embeddings using text_encoder."""
        self.model.eval()
        embeddings = []
        
        print(f"Processing {len(texts)} text queries in batches of {self.batch_size}...")
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize using CLIP processor
                tokens = self.text_processor(
                    text=batch_texts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=77
                )
                
                # Convert to tensors and move to device
                input_ids = torch.tensor(tokens['input_ids']).to(self.device)
                attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)
                
                # Extract embeddings using text encoder
                batch_embeddings = self.model.text_encoder(input_ids, attention_mask)
                
                # Ensure 2D output
                if len(batch_embeddings.shape) > 2:
                    batch_embeddings = batch_embeddings.mean(dim=1)  # Pool if needed
                
                embeddings.append(batch_embeddings.cpu())
                
                print(f"  Processed batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}, "
                      f"shape: {batch_embeddings.shape}")
        
        final_embeddings = torch.cat(embeddings, dim=0)
        print(f"Final text embeddings shape: {final_embeddings.shape}")
        return final_embeddings
    
    def extract_image_embeddings(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """Extract image embeddings using vision_encoder."""
        self.model.eval()
        embeddings = []
        
        print(f"Processing {len(image_paths)} images in batches of {self.batch_size}...")
        
        with torch.no_grad():
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                batch_images = []
                
                # Load and preprocess images
                for path in batch_paths:
                    img_tensor = self.load_and_preprocess_image(path)
                    batch_images.append(img_tensor)
                
                # Stack to create batch
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Extract embeddings using vision encoder with return_features=True
                batch_embeddings = self.model.vision_encoder(batch_tensor, return_features=True)
                
                # Handle different output formats
                if isinstance(batch_embeddings, tuple):
                    batch_embeddings = batch_embeddings[0]
                
                # Ensure 2D output (batch_size, embedding_dim)
                if len(batch_embeddings.shape) > 2:
                    # Apply global average pooling if spatial dimensions exist
                    while len(batch_embeddings.shape) > 2:
                        batch_embeddings = batch_embeddings.mean(dim=-1)
                
                embeddings.append(batch_embeddings.cpu())
                
                print(f"  Processed batch {i//self.batch_size + 1}/{(len(image_paths) + self.batch_size - 1)//self.batch_size}, "
                      f"input shape: {batch_tensor.shape}, output shape: {batch_embeddings.shape}")
        
        final_embeddings = torch.cat(embeddings, dim=0)
        print(f"Final image embeddings shape: {final_embeddings.shape}")
        return final_embeddings
    
    def create_query_to_relevant_mapping(self, 
                                       metadata_df: pd.DataFrame, 
                                       query_column: str = 'DescriptionEN') -> Dict[int, List[int]]:
        """
        Create mapping from query indices to relevant image indices.
        Each query can correspond to multiple relevant images.
        """
        query_to_relevant = {}
        
        # Get unique queries
        unique_queries = metadata_df[query_column].unique()
        print(f"Found {len(unique_queries)} unique queries")
        
        for query_idx, query_text in enumerate(unique_queries):
            # Find all images with this description
            query_matches = metadata_df[metadata_df[query_column] == query_text]
            
            if len(query_matches) == 0:
                query_to_relevant[query_idx] = []
                print(f"WARNING: Query {query_idx} '{query_text}' has no matching images!")
                continue
            
            # Use all matching images as relevant (not just the first one)
            relevant_image_indices = query_matches.index.tolist()
            query_to_relevant[query_idx] = relevant_image_indices

        return query_to_relevant
    
    def calculate_hitrate_at_k(self, 
                              top_k_indices: torch.Tensor, 
                              query_to_relevant_mapping: Dict[int, List[int]], 
                              k: int) -> float:
        """Calculate HitRate@k: Proportion of queries with correct image in top-k."""
        n_queries = top_k_indices.size(0)
        hit_count = 0
        
        for i in range(n_queries):
            relevant_indices = set(query_to_relevant_mapping.get(i, []))
            retrieved_indices = set(top_k_indices[i, :k].tolist())
            
            if not relevant_indices.isdisjoint(retrieved_indices):
                hit_count += 1
        
        return hit_count / n_queries if n_queries > 0 else 0.0
    
    def calculate_mrr_at_k(self, 
                          top_k_indices: torch.Tensor, 
                          query_to_relevant_mapping: Dict[int, List[int]], 
                          k: int) -> float:
        """Calculate Mean Reciprocal Rank@k."""
        n_queries = top_k_indices.size(0)
        reciprocal_ranks = []
        
        for i in range(n_queries):
            relevant_indices = query_to_relevant_mapping.get(i, [])
            best_rank = float('inf')
            
            for rank, retrieved_idx in enumerate(top_k_indices[i, :k].tolist()):
                if retrieved_idx in relevant_indices:
                    best_rank = rank + 1  # 1-indexed
                    break
            
            if best_rank != float('inf'):
                reciprocal_ranks.append(1.0 / best_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_recall_at_k(self, 
                             top_k_indices: torch.Tensor, 
                             query_to_relevant_mapping: Dict[int, List[int]], 
                             k: int) -> float:
        """Calculate Recall@k. For 1-to-1 mapping, Recall@k = HitRate@k."""
        n_queries = top_k_indices.size(0)
        recall_scores = []
        
        for i in range(n_queries):
            relevant_indices = set(query_to_relevant_mapping.get(i, []))
            retrieved_indices = set(top_k_indices[i, :k].tolist())
            
            found_relevant = len(relevant_indices.intersection(retrieved_indices))
            total_relevant = len(relevant_indices)
            
            recall = found_relevant / total_relevant if total_relevant > 0 else 0.0
            recall_scores.append(recall)
        
        return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    def calculate_precision_at_k(self, 
                                top_k_indices: torch.Tensor, 
                                query_to_relevant_mapping: Dict[int, List[int]], 
                                k: int) -> float:
        """Calculate Precision@k."""
        n_queries = top_k_indices.size(0)
        precision_scores = []
        
        for i in range(n_queries):
            relevant_indices = set(query_to_relevant_mapping.get(i, []))
            retrieved_indices = set(top_k_indices[i, :k].tolist())
            
            found_relevant = len(relevant_indices.intersection(retrieved_indices))
            precision = found_relevant / k if k > 0 else 0.0
            precision_scores.append(precision)
        
        return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    
    def calculate_metrics_with_topk(self,
                                   text_embeddings: torch.Tensor,
                                   image_embeddings: torch.Tensor,
                                   k_values: List[int],
                                   query_to_relevant_mapping: Dict[int, List[int]]) -> Dict[str, float]:
        """Calculate all metrics for text-to-image search."""
        print(f"Computing similarities...")
        print(f"Text embeddings shape: {text_embeddings.shape}")
        print(f"Image embeddings shape: {image_embeddings.shape}")
        
        # Validate shapes
        if text_embeddings.shape[1] != image_embeddings.shape[1]:
            raise ValueError(f"Embedding dimensions don't match: "
                           f"text {text_embeddings.shape[1]} vs image {image_embeddings.shape[1]}")
        
        results = {}
        
        # Normalize embeddings for cosine similarity
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        
        # Calculate similarity matrix (queries x images)
        similarity_matrix = torch.mm(text_embeddings, image_embeddings.t())
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        # Get top-k indices
        max_k = max(k_values)
        _, top_k_indices_all = torch.topk(similarity_matrix, max_k, dim=1)
        
        # Calculate metrics for each k
        for k in k_values:
            results[f"HitRate@{k}"] = self.calculate_hitrate_at_k(
                top_k_indices_all, query_to_relevant_mapping, k
            )
            
            results[f"MRR@{k}"] = self.calculate_mrr_at_k(
                top_k_indices_all, query_to_relevant_mapping, k
            )
            
            results[f"Recall@{k}"] = self.calculate_recall_at_k(
                top_k_indices_all, query_to_relevant_mapping, k
            )
            
            results[f"Precision@{k}"] = self.calculate_precision_at_k(
                top_k_indices_all, query_to_relevant_mapping, k
            )
        
        return results
    
    def evaluate(self, 
                    k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """Main evaluation function."""
        print("=== MEDICAL VLM EVALUATION ===")
        
        
        # Get unique queries
        unique_queries = self.test_df['DescriptionEN'].unique().tolist()
        print(f"Found {len(unique_queries)} unique queries")
        
        # Prepare image paths
        image_paths = []
        for _, row in self.test_df.iterrows():
            image_paths.append(Path(self.img_dir) / row['Path'])
        
        # Extract embeddings
        print(f"\n=== EXTRACTING EMBEDDINGS ===")
        text_embeddings = self.extract_text_embeddings(unique_queries)
        image_embeddings = self.extract_image_embeddings(image_paths)
        
        # Create relevance mapping
        print(f"\n=== CREATING RELEVANCE MAPPING ===")
        query_to_relevant_mapping = self.create_query_to_relevant_mapping(
            self.test_df, 'DescriptionEN'
        )
        
        # Calculate metrics
        print(f"\n=== CALCULATING METRICS ===")
        results = self.calculate_metrics_with_topk(
            text_embeddings, 
            image_embeddings, 
            k_values, 
            query_to_relevant_mapping
        )
        
        return results


def evaluate_text_to_image_search(model, 
                                 config) -> Dict[str, float]:
    """Convenience function for evaluation."""
    evaluator = MedVLMEvaluator(model, config)
    
    return evaluator.evaluate(
        k_values=config['k_values']
    )


# Example usage for your specific case
if __name__ == "__main__":
    # This would be replaced with your actual model and config
    from models.model import create_medical_vlm
    model = create_medical_vlm(
        vision_encoder={'type': 'endovit', 'feature_dim': 768, 'model_name': 'egeozsoy/EndoViT'},
        text_encoder={'type': 'clip', 'feature_dim': 768, 'model_name': 'openai/clip-vit-base-patch32'},
        temperature=0.07
    )
    model.load_from_path('/Users/thuylinh.lynguyen/Documents/Code/Track-3-ENTRep-Challenge/Pretrained/checkpoints/dinos_clip/best.pt')
    config = {
        'batch_size': 32,
        'image_size': (224, 224),
        'img_dir': 'Dataset/images',
        'json_path': 'Dataset/splits_info.json'
    }
    res = evaluate_text_to_image_search(model, config)
    print(res)