"""
Find descriptions with multiple relevant images from splits_info.json

This script analyzes the dataset to identify descriptions that appear with 
multiple different images, which is useful for understanding data distribution
and potential evaluation challenges.
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

def load_splits_data(json_path: str) -> Dict:
    """Load splits_info.json data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_descriptions(data: Dict, split: str = 'test') -> Dict[str, List[Dict]]:
    """
    Analyze descriptions to find those with multiple images
    
    Args:
        data: The loaded JSON data
        split: Which split to analyze ('train', 'test', 'val')
    
    Returns:
        Dictionary mapping descriptions to list of associated images
    """
    description_to_images = defaultdict(list)
    
    if split not in data:
        print(f"Warning: Split '{split}' not found in data")
        return {}
    
    # Group images by description
    for item in data[split]:
        desc_en = item.get('DescriptionEN', '').strip()
        if desc_en:  # Only process non-empty descriptions
            description_to_images[desc_en].append({
                'path': item.get('Path', ''),
                'classification': item.get('Classification', ''),
                'type': item.get('Type', ''),
                'description_vi': item.get('Description', '')
            })
    
    return description_to_images

def find_multi_relevant(description_to_images: Dict[str, List[Dict]], 
                       min_images: int = 2) -> Dict[str, List[Dict]]:
    """
    Filter descriptions that have multiple relevant images
    
    Args:
        description_to_images: Mapping of descriptions to images
        min_images: Minimum number of images to be considered "multi-relevant"
    
    Returns:
        Filtered dictionary with only multi-relevant descriptions
    """
    multi_relevant = {}
    
    for desc, images in description_to_images.items():
        if len(images) >= min_images:
            multi_relevant[desc] = images
    
    return multi_relevant

def print_statistics(description_to_images: Dict[str, List[Dict]], 
                    multi_relevant: Dict[str, List[Dict]]):
    """Print analysis statistics"""
    total_descriptions = len(description_to_images)
    multi_relevant_count = len(multi_relevant)
    
    print("="*80)
    print("DATASET ANALYSIS STATISTICS")
    print("="*80)
    print(f"Total unique descriptions: {total_descriptions}")
    print(f"Descriptions with multiple images: {multi_relevant_count}")
    print(f"Percentage with multiple images: {multi_relevant_count/total_descriptions*100:.2f}%")
    
    # Distribution analysis
    image_counts = [len(images) for images in description_to_images.values()]
    max_images = max(image_counts) if image_counts else 0
    
    print("\nDistribution of images per description:")
    for i in range(1, min(max_images + 1, 11)):  # Show up to 10
        count = sum(1 for x in image_counts if x == i)
        print(f"  {i} image(s): {count} descriptions")
    
    if max_images > 10:
        count_more = sum(1 for x in image_counts if x > 10)
        print(f"  >10 images: {count_more} descriptions")

def save_results(multi_relevant: Dict[str, List[Dict]], 
                output_file: str,
                split: str):
    """Save results to JSON file"""
    # Prepare output data
    output_data = {
        'split': split,
        'total_multi_relevant_descriptions': len(multi_relevant),
        'descriptions': []
    }
    
    # Sort by number of images (descending)
    sorted_descriptions = sorted(multi_relevant.items(), 
                                key=lambda x: len(x[1]), 
                                reverse=True)
    
    for desc, images in sorted_descriptions:
        desc_data = {
            'description_en': desc,
            'image_count': len(images),
            'images': images
        }
        output_data['descriptions'].append(desc_data)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")

def print_top_examples(multi_relevant: Dict[str, List[Dict]], top_n: int = 5):
    """Print top examples with most images"""
    print(f"\nTop {top_n} descriptions with most images:")
    print("-" * 80)
    
    # Sort by number of images
    sorted_descriptions = sorted(multi_relevant.items(), 
                                key=lambda x: len(x[1]), 
                                reverse=True)
    
    for i, (desc, images) in enumerate(sorted_descriptions[:top_n], 1):
        print(f"\n{i}. Description: \"{desc}\"")
        print(f"   Number of images: {len(images)}")
        print("   Sample images:")
        for j, img in enumerate(images[:3]):  # Show first 3 images
            print(f"     - {img['path']} ({img['classification']}, {img['type']})")
        if len(images) > 3:
            print(f"     ... and {len(images) - 3} more images")

def main():
    parser = argparse.ArgumentParser(description='Find descriptions with multiple relevant images')
    parser.add_argument('--json_path', type=str, default='Dataset/splits_info.json',
                       help='Path to splits_info.json file')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'test', 'val'],
                       help='Which split to analyze')
    parser.add_argument('--min_images', type=int, default=2,
                       help='Minimum number of images to be considered multi-relevant')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: auto-generated)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top examples to display')
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        args.output = f'multi_relevant_{args.split}_{args.min_images}plus.json'
    
    print(f"Loading data from: {args.json_path}")
    print(f"Analyzing split: {args.split}")
    print(f"Minimum images threshold: {args.min_images}")
    
    # Load and analyze data
    data = load_splits_data(args.json_path)
    description_to_images = analyze_descriptions(data, args.split)
    multi_relevant = find_multi_relevant(description_to_images, args.min_images)
    
    # Print statistics and examples
    print_statistics(description_to_images, multi_relevant)
    print_top_examples(multi_relevant, args.top_n)
    
    # Save results
    save_results(multi_relevant, args.output, args.split)
    
    print(f"\n✅ Analysis complete!")
    print(f"Found {len(multi_relevant)} descriptions with {args.min_images}+ images")

if __name__ == "__main__":
    main() 