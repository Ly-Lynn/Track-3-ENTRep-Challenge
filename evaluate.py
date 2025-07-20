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
    data = data['train']
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

    # Search once with max k and slice for different k values
    search_results = []
    max_k = max(evaluator_cfg['k_values']) 
    
    for query, _ in tqdm(test_set, desc="Searching"):
        results = searcher.search_by_text(query, max_k)  
        search_results.append(results)
    
    print(f"Search results length: {len(search_results)}")

    print("Evaluating metrics for all k values...")
    final_metrics = metrics.compute_all_metrics(search_results, evaluator_cfg['k_values'])
    
    metrics.print_and_save_results(final_metrics, evaluator_cfg['k_values'], 'final_metrics_train.txt')

if __name__ == "__main__":
    main()