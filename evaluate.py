import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from metrics import Metrics
from models.model import create_medical_vlm
from searcher import Searcher
from utils import load_json, validate_json, create_df_from_json

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a MedicalVLM model with enhanced test set')
    parser.add_argument('--file_config', type=str, default='endovit.yaml', help='File config')
    return parser.parse_args()

def load_config(file_config):
    with open(os.path.join('Config/eval/', file_config), 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_enhanced_test_set(enhanced_test_path):
    """
    Load enhanced test set with multiple relevant images per query
    
    Returns:
        List of (query_description, original_path) tuples for compatibility
    """
    with open(enhanced_test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_queries = data.get('test_queries', [])
    results = []
    for item in test_queries:
        relevant_images = []
        for img in item['relevant_images']:
            relevant_images.append(img['path'])
        results.append((item['query_description'], relevant_images))
    
    print(f"✅ Loaded {len(test_queries)} enhanced test queries")
    print(f"📊 Total relevant images in dataset: {data.get('total_relevant_images', 0)}")
    
    return results

def main():
    args = parse_args()
    config = load_config(args.file_config)
    model_cfg = config['model']
    evaluator_cfg = config['evaluator']

    original_dataset_path = evaluator_cfg['json_path']  # For building index
    enhanced_test_path = evaluator_cfg.get('test_path')  # For evaluation
    
    print(f"📂 Original dataset: {original_dataset_path}")
    print(f"📂 Enhanced test set: {enhanced_test_path}")

    test_set = load_enhanced_test_set(enhanced_test_path)
    print(f"Test set length: {len(test_set)}")
    
    metrics = Metrics.from_test_set_tuples(test_set)

    searcher = Searcher(config=config)
    searcher.build_index(f"{args.file_config.split('.')[0]}_index")

    search_results = []
    max_k = max(evaluator_cfg['k_values']) 
    all_res = []
    
    print(f"🔍 Searching with max_k={max_k} for {len(test_set)} queries...")
    for query, gt in tqdm(test_set, desc="Searching"):
        results = searcher.search_by_text(query, max_k)  
        search_results.append(results)
        all_res.append({
            "query": query,
            "gt": gt,
            "results": results
        })
    
    search_results_file = f"search_results_{args.file_config.split('.')[0]}_enhanced.json"
    with open(search_results_file, "w") as f:
        json.dump(all_res, f, indent=4)
    print(f"💾 Search results saved to: {search_results_file}")
    print(f"Search results length: {len(search_results)}")

    print("🔢 Evaluating enhanced metrics for all k values...")
    
    try:
        final_metrics = metrics.compute_all_metrics(search_results, evaluator_cfg['k_values'])
        
        output_file = f'enhanced_metrics_{args.file_config.split(".")[0]}.txt'
        metrics.save_metrics(final_metrics, evaluator_cfg['k_values'], output_file)
        
    except Exception as e:
        print(f"❌ Error during metrics computation: {e}")
        print("Please check that search results format is compatible with enhanced metrics")
        raise

if __name__ == "__main__":
    main()