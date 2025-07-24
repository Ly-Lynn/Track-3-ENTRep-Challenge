"""
This script creates a dataset from a json file
- metadata.json
{
    "train": [
        {
            "Path": "20066822_230920083932327831_954_image03.png",
            "Classification": "vc-open",
            "Type": "abnormal",
            "Description": "phù nề sụn phễu và nếp liên phễu",
            "DescriptionEN": "edema of the arytenoid cartilages and interarytenoid area"
        },
        ...
    ],
    "val": [
        ...
    ],
    "test":[
        ...
    ]
}
"""

import pandas as pd
import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def validate_json(json_path):
    json_data = load_json(json_path)
    train, val, test = json_data["train"], json_data["val"], json_data["test"]
    def is_vietnamese(text):
        vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        return any(char in text.lower() for char in vietnamese_chars)

    def is_english(text):
        if not text:
            return False
        text_alpha = re.sub(r'[^a-zA-Z]', '', text)
        ratio = len(text_alpha) / max(1, len(text.replace(" ", "")))
        return ratio > 0.7

    for idx, data_item in enumerate(train + val + test):
        desc = data_item.get("Description", "")
        desc_en = data_item.get("DescriptionEN", "")
        if not desc:
            print(f"Warning: Description at index {idx} is empty")
        elif not desc_en:
            print(f"Warning: DescriptionEN at index {idx} is empty")
        else:
            if not is_vietnamese(desc):
                print(f"Warning: Description at index {idx} may not be Vietnamese: {desc}")
            if not is_english(desc_en):
                print(f"Warning: DescriptionEN at index {idx} may not be English: {desc_en}")
    
    return train, val, test

def create_df_from_json(json_path):
    train_meta, val_meta, test_meta = validate_json(json_path)
    all_meta = train_meta + val_meta + test_meta
    all_df, train_df, val_df, test_df = pd.DataFrame(all_meta), pd.DataFrame(train_meta), pd.DataFrame(val_meta), pd.DataFrame(test_meta)
    return all_df, train_df, val_df, test_df
def load_test_imgs(path):
    res = []
    for img in os.listdir(path):
        res.append({
            'Path': img,
            'label': '',
            'Type': '',
            'Classification': '',
            'DescriptionEN': '', 
            'Description': ''
        })
    return pd.DataFrame(res)


def create_entrep_testset(json_path):
    data = load_json(json_path)
    test_raw = data["test"]
    testset = defaultdict(list)
    print(f"Creating entrep testset from {json_path}...")
    for item in tqdm(test_raw):
        query = item['DescriptionEN']
        if query:
            sample = {
                'Path': item['Path'],
                'Classification': item['Classification'],
                'Description': item['Description'],
                'DescriptionEN': item['DescriptionEN'],
                'Type': item['Type']
            }
            testset[query].append(sample)
    return testset
# def create_entrep_testdf(json_path):