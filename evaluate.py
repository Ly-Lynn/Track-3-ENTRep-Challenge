from evaluator import evaluate_text_to_image_search
from models.model import create_medical_vlm
from utils import load_json, validate_json, create_df_from_json
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import argparse
import yaml
import os
from utils import create_df_from_json
from torchvision import transforms
import json
from metrics import Metrics
from searcher import Searcher
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a MedicalVLM model')
    parser.add_argument('--file_config', type=str, default='endovit.yaml', help='File config')
    return parser.parse_args()

def load_config(file_config):
    with open(os.path.join('Config/eval/', file_config), 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_test_set(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = data['test']
    res = []
    for item in data:
        res.append((item['DescriptionEN'], item['Path']))
    return res

def main():
    args = parse_args()
    config = load_config(args.file_config)
    model_cfg = config['model']
    evaluator_cfg = config['evaluator']

    # Load test set
    test_set = load_test_set(os.path.join(evaluator_cfg['json_path']))
    print(f"Test set length: {len(test_set)}")
    metrics = Metrics(test_set)

    # Build index
    searcher = Searcher(config=config)
    searcher.build_index("vector_index")

    # Search
    search_results = []
    all_metrics = {}
    
    # Initialize metrics storage
    for k in evaluator_cfg['k_values']:
        all_metrics[k] = {
            'hit_rate': 0.0,
            'recall': 0.0, 
            'mrr': 0.0
        }
    
    # Collect search results for all queries
    for query, _ in tqdm(test_set, desc="Searching"):
        query_results = []
        for k in evaluator_cfg['k_values']:
            results = searcher.search_by_text(query, k)
            query_results.append(results)
        search_results.append(query_results)
    
    print(f"Search results length: {len(search_results)}")

    # Calculate and save average metrics for each k
    for k_idx, k in enumerate(evaluator_cfg['k_values']):
        print(f"Evaluating metrics for k={k}")
        
        # Get results for current k value
        k_results = [results[k_idx] for results in search_results]
        
        # Calculate metrics
        final_results = metrics.compute_all_metrics(k_results, [k])
        metrics.print_results(final_results, [k])
        
        # Update running averages
        all_metrics[k]['hit_rate'] = final_results['hit_rate'][k]
        all_metrics[k]['recall'] = final_results['recall'][k]
        all_metrics[k]['mrr'] = final_results['mrr'][k]
    
    # Save average metrics to file
    with open('average_metrics.txt', 'w') as f:
        for k in evaluator_cfg['k_values']:
            f.write(f"Results for k={k}:\n")
            f.write(f"Average Hit Rate@{k}: {all_metrics[k]['hit_rate']:.3f}\n")
            f.write(f"Average Recall@{k}: {all_metrics[k]['recall']:.3f}\n") 
            f.write(f"Average MRR@{k}: {all_metrics[k]['mrr']:.3f}\n")
            f.write("="*50 + "\n")

if __name__ == "__main__":
    main()