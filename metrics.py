import json
import os
from typing import List, Dict, Any, Tuple, Optional

class Metrics:
    def __init__(self, enhanced_test_file: str = None, test_set_tuples: Optional[List[Tuple[str, List[str]]]] = None):
        if test_set_tuples is not None:
            self._init_from_tuples(test_set_tuples)
        elif enhanced_test_file is not None:
            self._init_from_file(enhanced_test_file)
        else:
            raise ValueError("Either enhanced_test_file or test_set_tuples must be provided")
    
    def _init_from_file(self, enhanced_test_file: str):
        self.enhanced_test_data = self._load_enhanced_test_set(enhanced_test_file)
        self.test_queries = self.enhanced_test_data.get('test_queries', [])
        
        print(f"✅ Loaded {len(self.test_queries)} enhanced test queries from file")
        print(f"📊 Total relevant images in dataset: {self.enhanced_test_data.get('total_relevant_images', 0)}")
    
    def _init_from_tuples(self, test_set_tuples: List[Tuple[str, List[str]]]):
        self.test_queries = []
        total_relevant = 0
        
        for query_desc, relevant_paths in test_set_tuples:
            query_data = {
                'query_description': query_desc,
                'total_relevant_images': len(relevant_paths),
                'relevant_images': [{'path': path} for path in relevant_paths]
            }
            self.test_queries.append(query_data)
            total_relevant += len(relevant_paths)
        
        self.enhanced_test_data = {
            'total_queries': len(test_set_tuples),
            'total_relevant_images': total_relevant,
            'test_queries': self.test_queries
        }
        
        print(f"✅ Loaded {len(test_set_tuples)} enhanced test queries from tuples")
        print(f"📊 Total relevant images: {total_relevant}")
    
    @classmethod
    def from_test_set_tuples(cls, test_set_tuples: List[Tuple[str, List[str]]]):
        return cls(test_set_tuples=test_set_tuples)
        
    def _load_enhanced_test_set(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise Exception(f"Error loading enhanced test set: {e}")
    
    def _get_relevant_paths_for_query(self, query_idx: int) -> List[str]:
        if query_idx >= len(self.test_queries):
            return []
        
        query = self.test_queries[query_idx]
        relevant_paths = [img['path'] for img in query.get('relevant_images', [])]
        return relevant_paths
    
    def _is_relevant(self, result_path: str, relevant_paths: List[str]) -> bool:
        if not result_path:
            return False
        
        result_filename = os.path.basename(result_path.strip())
        
        for rel_path in relevant_paths:
            rel_filename = os.path.basename(rel_path.strip())
            if result_filename == rel_filename:
                return True
        return False
    
    def _find_relevant_positions(self, search_results: List[Dict[str, Any]], 
                                relevant_paths: List[str]) -> List[int]:
        relevant_positions = []
        for i, result in enumerate(search_results):
            result_path = result.get("image_path")
            if self._is_relevant(result_path, relevant_paths):
                relevant_positions.append(i)
        return relevant_positions
    
    def hit_rate_at_k(self, 
                     search_results_list: List[List[Dict[str, Any]]], 
                     k: int = 5) -> float:
        if len(search_results_list) != len(self.test_queries):
            raise ValueError(f"Search results length ({len(search_results_list)}) != test queries length ({len(self.test_queries)})")
        
        hits = 0
        for query_idx, search_results in enumerate(search_results_list):
            top_k_results = search_results[:k]
            
            relevant_paths = self._get_relevant_paths_for_query(query_idx)
            relevant_positions = self._find_relevant_positions(top_k_results, relevant_paths)
            
            if len(relevant_positions) > 0:
                hits += 1
        
        return hits / len(self.test_queries)
    
    def recall_at_k(self, 
                   search_results_list: List[List[Dict[str, Any]]], 
                   k: int = 5) -> float:
        if len(search_results_list) != len(self.test_queries):
            raise ValueError(f"Search results length ({len(search_results_list)}) != test queries length ({len(self.test_queries)})")
        
        total_recall = 0.0
        
        for query_idx, search_results in enumerate(search_results_list):
            top_k_results = search_results[:k]
            
            relevant_paths = self._get_relevant_paths_for_query(query_idx)
            total_relevant_for_query = len(relevant_paths)
            
            if total_relevant_for_query == 0:
                continue
            
            relevant_positions = self._find_relevant_positions(top_k_results, relevant_paths)
            found_relevant = len(relevant_positions)
            
            query_recall = found_relevant / total_relevant_for_query
            total_recall += query_recall
        
        return total_recall / len(self.test_queries)
    
    def mrr_at_k(self, 
                search_results_list: List[List[Dict[str, Any]]], 
                k: int = 5) -> float:
        if len(search_results_list) != len(self.test_queries):
            raise ValueError(f"Search results length ({len(search_results_list)}) != test queries length ({len(self.test_queries)})")
        
        reciprocal_ranks = []
        
        for query_idx, search_results in enumerate(search_results_list):
            top_k_results = search_results[:k]
            
            relevant_paths = self._get_relevant_paths_for_query(query_idx)
            relevant_positions = self._find_relevant_positions(top_k_results, relevant_paths)
            
            if len(relevant_positions) > 0:
                first_relevant_rank = relevant_positions[0] + 1
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    def compute_all_metrics(self, 
                           search_results_list: List[List[Dict[str, Any]]], 
                           k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
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
        
    def save_metrics(self, 
                    metrics_results: Dict[str, Dict[int, float]], 
                    k_values: List[int],
                    filename: str = "enhanced_metrics_results.txt",
                    format_style: str = "detailed",
                    mode: str = "w"):
        
        with open(filename, mode) as f:
            if format_style == "detailed":
                for k in k_values:
                    f.write(f"Results for k={k}:\n")
                    f.write(f"  Hit Rate@{k}: {metrics_results['hit_rate'][k]:.3f}\n")
                    f.write(f"  Recall@{k}: {metrics_results['recall'][k]:.3f}\n") 
                    f.write(f"  MRR@{k}: {metrics_results['mrr'][k]:.3f}\n")
                    f.write("-"*30 + "\n")
            else:
                for metric_name in ['hit_rate', 'recall', 'mrr']:
                    values = metrics_results[metric_name]
                    for k in k_values:
                        f.write(f"{metric_name}@{k:<8} | {values[k]:.3f}\n")
            
            f.write(f"\nNOTE: Metrics computed using enhanced test set with multiple relevant images.\n")
        
        print(f"💾 Enhanced metrics saved to {filename}")
