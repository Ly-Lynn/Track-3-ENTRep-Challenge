"""
Create Enhanced Test Set with Multiple Relevant Images

This script processes splits_info.json to create a new test set where each query 
has an array of ALL relevant image paths from the entire dataset (train/test/val).

This provides a more accurate ground truth for evaluation metrics.
"""

import json
from collections import defaultdict
from typing import Dict, List, Set
import argparse

def load_splits_data(json_path: str) -> Dict:
    """Load splits_info.json data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_description_to_paths_map(data: Dict) -> Dict[str, List[Dict]]:
    """
    Build mapping from DescriptionEN to all image paths across all splits
    
    Returns:
        Dict mapping description to list of image info dicts
    """
    description_map = defaultdict(list)
    
    # Process all splits: train, test, val
    for split_name in ['train', 'test', 'val']:
        if split_name not in data:
            continue
            
        for item in data[split_name]:
            desc_en = item.get('DescriptionEN', '').strip()
            if desc_en:  # Only process non-empty descriptions
                image_info = {
                    'path': item.get('Path', ''),
                    'classification': item.get('Classification', ''),
                    'type': item.get('Type', ''),
                    'description_vi': item.get('Description', ''),
                    'description_en': desc_en,
                    'source_split': split_name
                }
                description_map[desc_en].append(image_info)
    
    return description_map

def create_enhanced_test_set(data: Dict, description_map: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Create enhanced test set with multiple relevant images per query
    
    Args:
        data: Original splits data
        description_map: Mapping from description to all relevant images
    
    Returns:
        List of enhanced test items
    """
    if 'test' not in data:
        print("Warning: No 'test' split found in data")
        return []
    
    enhanced_test = []
    processed_descriptions = set()  # To avoid duplicates
    
    for item in data['test']:
        desc_en = item.get('DescriptionEN', '').strip()
        
        if not desc_en or desc_en in processed_descriptions:
            continue  # Skip empty descriptions or already processed ones
            
        # Get all relevant images for this description
        relevant_images = description_map.get(desc_en, [])
        
        if not relevant_images:
            continue  # Skip if no relevant images found
        
        # Create enhanced test item
        enhanced_item = {
            'query_description': desc_en,
            'query_description_vi': item.get('Description', ''),
            'original_path': item.get('Path', ''),  # Original path from test set
            'classification': item.get('Classification', ''),
            'type': item.get('Type', ''),
            'total_relevant_images': len(relevant_images),
            'relevant_images': relevant_images
        }
        
        enhanced_test.append(enhanced_item)
        processed_descriptions.add(desc_en)
    
    return enhanced_test

def print_statistics(enhanced_test: List[Dict]):
    """Print statistics about the enhanced test set"""
    print("="*80)
    print("ENHANCED TEST SET STATISTICS")
    print("="*80)
    
    total_queries = len(enhanced_test)
    total_relevant_images = sum(item['total_relevant_images'] for item in enhanced_test)
    
    print(f"Total unique queries: {total_queries}")
    print(f"Total relevant images: {total_relevant_images}")
    print(f"Average relevant images per query: {total_relevant_images/total_queries:.2f}")
    
    # Distribution analysis
    image_counts = [item['total_relevant_images'] for item in enhanced_test]
    max_images = max(image_counts) if image_counts else 0
    
    print("\nDistribution of relevant images per query:")
    for i in range(1, min(max_images + 1, 11)):  # Show up to 10
        count = sum(1 for x in image_counts if x == i)
        if count > 0:
            print(f"  {i} image(s): {count} queries")
    
    if max_images > 10:
        count_more = sum(1 for x in image_counts if x > 10)
        print(f"  >10 images: {count_more} queries")
    
    # Show distribution by source split
    print("\nRelevant images by source split:")
    source_counts = defaultdict(int)
    for item in enhanced_test:
        for img in item['relevant_images']:
            source_counts[img['source_split']] += 1
    
    for split, count in source_counts.items():
        percentage = (count / total_relevant_images) * 100
        print(f"  {split}: {count} images ({percentage:.1f}%)")

def print_top_examples(enhanced_test: List[Dict], top_n: int = 5):
    """Print top examples with most relevant images"""
    print(f"\nTop {top_n} queries with most relevant images:")
    print("-" * 80)
    
    # Sort by number of relevant images
    sorted_queries = sorted(enhanced_test, 
                           key=lambda x: x['total_relevant_images'], 
                           reverse=True)
    
    for i, item in enumerate(sorted_queries[:top_n], 1):
        print(f"\n{i}. Query: \"{item['query_description']}\"")
        print(f"   Total relevant images: {item['total_relevant_images']}")
        print(f"   Original test path: {item['original_path']}")
        
        # Show distribution by split
        split_counts = defaultdict(int)
        for img in item['relevant_images']:
            split_counts[img['source_split']] += 1
        
        split_info = ", ".join([f"{split}: {count}" for split, count in split_counts.items()])
        print(f"   Distribution: {split_info}")
        
        # Show first few images
        print("   Sample relevant images:")
        for j, img in enumerate(item['relevant_images'][:3]):
            print(f"     - {img['path']} (from {img['source_split']})")
        if item['total_relevant_images'] > 3:
            print(f"     ... and {item['total_relevant_images'] - 3} more images")

def save_enhanced_test_set(enhanced_test: List[Dict], output_file: str):
    """Save enhanced test set to JSON file"""
    output_data = {
        'description': 'Enhanced test set with multiple relevant images per query',
        'total_queries': len(enhanced_test),
        'total_relevant_images': sum(item['total_relevant_images'] for item in enhanced_test),
        'creation_info': {
            'source': 'Generated from splits_info.json',
            'method': 'Grouped by DescriptionEN across all splits'
        },
        'test_queries': enhanced_test
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Enhanced test set saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create enhanced test set with multiple relevant images')
    parser.add_argument('--input', type=str, default='Dataset/splits_info.json',
                       help='Path to input splits_info.json file')
    parser.add_argument('--output', type=str, default='enhanced_test_set.json',
                       help='Output file path for enhanced test set')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top examples to display')
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input}")
    print(f"Output file: {args.output}")
    
    # Load and process data
    data = load_splits_data(args.input)
    print("✅ Data loaded successfully")
    
    # Build description to paths mapping
    description_map = build_description_to_paths_map(data)
    print(f"✅ Built mapping for {len(description_map)} unique descriptions")
    
    # Create enhanced test set
    enhanced_test = create_enhanced_test_set(data, description_map)
    print(f"✅ Created enhanced test set with {len(enhanced_test)} queries")
    
    # Print statistics and examples
    print_statistics(enhanced_test)
    print_top_examples(enhanced_test, args.top_n)
    
    # Save results
    save_enhanced_test_set(enhanced_test, args.output)
    
    print(f"\n✅ Enhanced test set creation complete!")
    print(f"Original test queries with single images -> Enhanced queries with multiple relevant images")
    print(f"This will provide more accurate evaluation metrics!")

if __name__ == "__main__":
    main()