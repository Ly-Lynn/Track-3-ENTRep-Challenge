"""
Evaluation Metrics for Medical Image Search System

Hit Rate@k: Proportion of queries with at least one relevant item in top-k results
Recall@k: Proportion of relevant items found in top-k results  
MRR@k: Mean Reciprocal Rank - average of reciprocal ranks of first relevant items

Relevance is determined by checking if ground truth label_path appears in search results.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Union


class Metrics:
    def __init__(self, ground_truth: List[Tuple[str, str]]):
        """Initialize the Metrics evaluator"""
        self.ground_truth = ground_truth
    
    def _is_relevant(self, result_path: str, ground_truth_path: str) -> bool:
        return result_path.strip() == ground_truth_path.strip()
    
    def _find_relevant_positions(self, search_results: List[Dict[str, Any]], ground_truth_path: str) -> List[int]:
        relevant_positions = []
        for i, result in enumerate(search_results):
            result_path = result.get("path")
            if self._is_relevant(result_path, ground_truth_path):
                relevant_positions.append(i)
        return relevant_positions
    
    def hit_rate_at_k(self, 
                     search_results_list: List[List[Dict[str, Any]]], 
                     k: int = 5) -> float:

        hits = 0
        for (description, label_path), search_results in zip(self.ground_truth, search_results_list):
            top_k_results = search_results
            
            relevant_positions = self._find_relevant_positions(top_k_results, label_path)
            if len(relevant_positions) > 0:
                hits += 1
        
        return hits 
    
    def recall_at_k(self, 
                   search_results_list: List[List[Dict[str, Any]]], 
                   k: int = 5) -> float:
        
        total_found_relevant = 0
        total_relevant = 0
        
        for (description, label_path), search_results in zip(self.ground_truth, search_results_list):
            top_k_results = search_results
            relevant_in_top_k = self._find_relevant_positions(top_k_results, label_path)
            total_found_relevant += len(relevant_in_top_k)
            
            all_relevant = self._find_relevant_positions(search_results, label_path)
            total_relevant += len(all_relevant)
        
        if total_relevant == 0:
            return 0.0
        
        return total_found_relevant / total_relevant
    
    def mrr_at_k(self, 
                search_results_list: List[List[Dict[str, Any]]], 
                k: int = 5) -> float:
        reciprocal_ranks = []
        
        for (description, label_path), search_results in zip(self.ground_truth, search_results_list):
            top_k_results = search_results
            
            relevant_positions = self._find_relevant_positions(top_k_results, label_path)
            
            if len(relevant_positions) > 0:
                first_relevant_rank = relevant_positions[0] + 1
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks)
    
    def compute_all_metrics(self, 
                           search_results_list: List[List[Dict[str, Any]]], 
                           k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
        """
        Compute all metrics for multiple k values
        
        Args:
            ground_truth: List of (description, label_path) tuples
            search_results_list: List of search results for each query
            k_values: List of k values to evaluate
        
        Returns:
            Dictionary with metric names as keys and k-value results as values
        """
        results = {
            "hit_rate": {},
            "recall": {},
            "mrr": {}
        }
        
        for k in k_values:
            results["hit_rate"][k] = self.hit_rate_at_k(search_results_list, k)
            results["recall"][k] = self.recall_at_k(search_results_list, k)
            results["mrr"][k] = self.mrr_at_k(search_results_list, k)
        
        return results
    
    def print_results(self, 
                     metrics_results: Dict[str, Dict[int, float]], 
                     k_values: List[int]):
        """
        Print metrics results in a formatted table
        
        Args:
            metrics_results: Results from compute_all_metrics
            k_values: List of k values
        """
        # Save metrics to file
        with open("metrics_results.txt", "a") as f:
            # Write each metric with k value
            for metric_name in ['hit_rate', 'recall', 'mrr']:
                values = metrics_results[metric_name]
                for k in k_values:
                    f.write(f"{metric_name}@{k:<8} | {values[k]:.3f}\n")
            
            f.write("="*50 + "\n")


if __name__ == "__main__":
    # Test với dữ liệu mẫu - format mới
    
    # Ground truth: [(description, label_path), ...]
    ground_truth = [
        ("normal vocal cord tissue", "img1.jpg"),
        ("polyp in the throat", "img5.jpg"), 
        ("inflammation of epiglottis", "img10.jpg")
    ]
    
    # Search results từ FAISS cho mỗi query
    search_results_list = [
        # Results cho query 1: img1.jpg ở vị trí đầu -> relevant
        [
            {"image_path": "img1.jpg", "sim_score": 0.9},   # relevant (match)
            {"image_path": "img2.jpg", "sim_score": 0.8},   # not relevant
            {"image_path": "img3.jpg", "sim_score": 0.7},   # not relevant
        ],
        # Results cho query 2: img5.jpg ở vị trí thứ 2 -> relevant
        [
            {"image_path": "img4.jpg", "sim_score": 0.85},  # not relevant
            {"image_path": "img5.jpg", "sim_score": 0.75},  # relevant (match)
            {"image_path": "img6.jpg", "sim_score": 0.65},  # not relevant
        ],
        # Results cho query 3: img10.jpg không có trong top-3 -> not relevant
        [
            {"image_path": "img7.jpg", "sim_score": 0.8},   # not relevant
            {"image_path": "img8.jpg", "sim_score": 0.7},   # not relevant
            {"image_path": "img9.jpg", "sim_score": 0.6},   # not relevant
        ]
    ]
    
    # Test metrics
    metrics = Metrics(ground_truth)
    
    print("Testing Metrics với ground truth và search results:")
    
    # Test từng metric riêng lẻ
    for k in [1, 2, 3]:
        hit_rate = metrics.hit_rate_at_k(search_results_list, k)
        recall = metrics.recall_at_k(search_results_list, k)
        mrr = metrics.mrr_at_k(search_results_list, k)
        
        print(f"\nk={k}:")
        print(f"  Hit Rate@{k}: {hit_rate:.3f}")
        print(f"  Recall@{k}: {recall:.3f}")
        print(f"  MRR@{k}: {mrr:.3f}")
    
    # Test tất cả metrics cùng lúc
    all_metrics = metrics.compute_all_metrics(search_results_list, [1, 2, 3, 5])
    metrics.print_results(all_metrics, [1, 2, 3, 5]) 