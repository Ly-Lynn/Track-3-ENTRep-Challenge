"""
Utilities for handling medical image dataset.

This module provides functions to load, validate, and process metadata 
from JSON files containing train/val/test splits with medical image descriptions
in both Vietnamese and English.

Expected JSON format:
{
    "train": [
        {
            "Path": "image_file.png",
            "Classification": "vc-open",
            "Type": "abnormal", 
            "Description": "Vietnamese description",
            "DescriptionEN": "English description"
        },
        ...
    ],
    "val": [...],
    "test": [...]
}
"""

import json
import os
import re
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd

from constants import SUPPORTED_IMAGE_EXTENSIONS


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load JSON file safely.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Loaded JSON data as dictionary
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in {json_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading JSON file {json_path}: {e}")

def _is_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese characters."""
    vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    return any(char in text.lower() for char in vietnamese_chars)


def _is_english(text: str) -> bool:
    """Check if text is primarily English."""
    if not text:
        return False
    text_alpha = re.sub(r'[^a-zA-Z]', '', text)
    ratio = len(text_alpha) / max(1, len(text.replace(" ", "")))
    return ratio > 0.7


def _validate_image_path(path: str) -> bool:
    """Validate if path has supported image extension."""
    if not path:
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def validate_json(json_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load and validate JSON dataset file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Tuple of (train_data, val_data, test_data)
        
    Raises:
        KeyError: If required keys are missing
        ValueError: If data structure is invalid
    """
    json_data = load_json(json_path)
    
    # Validate required keys
    required_keys = ["train", "val", "test"]
    missing_keys = [key for key in required_keys if key not in json_data]
    if missing_keys:
        raise KeyError(f"Missing required keys in JSON: {missing_keys}")
    
    train, val, test = json_data["train"], json_data["val"], json_data["test"]
    
    # Validate data items
    all_items = train + val + test
    warnings = []
    
    for idx, data_item in enumerate(all_items):
        # Check required fields
        required_fields = ["Path", "Classification", "Type", "Description", "DescriptionEN"]
        missing_fields = [field for field in required_fields if field not in data_item]
        if missing_fields:
            warnings.append(f"Item {idx}: Missing fields {missing_fields}")
            continue
            
        desc = data_item.get("Description", "")
        desc_en = data_item.get("DescriptionEN", "")
        path = data_item.get("Path", "")
        
        # Validate descriptions
        if not desc:
            warnings.append(f"Item {idx}: Empty Vietnamese description")
        elif not _is_vietnamese(desc):
            warnings.append(f"Item {idx}: Description may not be Vietnamese: {desc[:50]}...")
            
        if not desc_en:
            warnings.append(f"Item {idx}: Empty English description")
        elif not _is_english(desc_en):
            warnings.append(f"Item {idx}: DescriptionEN may not be English: {desc_en[:50]}...")
        
        # Validate image path
        if not _validate_image_path(path):
            warnings.append(f"Item {idx}: Invalid or unsupported image path: {path}")
    
    # Print warnings
    if warnings:
        print(f"⚠️ Found {len(warnings)} validation warnings:")
        for warning in warnings[:10]:  # Limit to first 10 warnings
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  - ... and {len(warnings) - 10} more warnings")
    
    return train, val, test

def create_df_from_json(json_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create pandas DataFrames from JSON dataset file.
    
    Args:
        json_path: Path to JSON dataset file
        
    Returns:
        Tuple of (all_df, train_df, val_df, test_df)
    """
    train_meta, val_meta, test_meta = validate_json(json_path)
    
    # Create DataFrames
    all_meta = train_meta + val_meta + test_meta
    all_df = pd.DataFrame(all_meta)
    train_df = pd.DataFrame(train_meta)
    val_df = pd.DataFrame(val_meta)
    test_df = pd.DataFrame(test_meta)
    
    # Print dataset statistics
    print(f"📊 Dataset statistics:")
    print(f"  - Total samples: {len(all_df)}")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Validation: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples")
    
    return all_df, train_df, val_df, test_df


def load_test_imgs(path: str) -> pd.DataFrame:
    """
    Load test images from directory and create DataFrame.
    
    Args:
        path: Directory path containing test images
        
    Returns:
        DataFrame with image metadata
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test images directory not found: {path}")
    
    try:
        image_files = [
            f for f in os.listdir(path) 
            if _validate_image_path(f)
        ]
        
        results = []
        for img in image_files:
            results.append({
                'Path': img,
                'label': '',
                'Type': '',
                'Classification': '',
                'DescriptionEN': '', 
                'Description': ''
            })
        
        print(f"📁 Loaded {len(results)} test images from {path}")
        return pd.DataFrame(results)
        
    except Exception as e:
        raise RuntimeError(f"Error loading test images from {path}: {e}")


