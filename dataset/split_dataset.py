# create_split_files.py
import json
import os
from pathlib import Path

def create_triplet_files(input_json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading JSON from: {input_json_path}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    splits_data = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for img in data['images']:
        if img['split'] in ['train', 'restval']:
            split_name = 'train'
        elif img['split'] == 'val':
            split_name = 'val'
        elif img['split'] == 'test':
            split_name = 'test'
        else:
            continue
        
        feature_filename = img['filename'].replace('.jpg', '.hdf5')
        feature_path = f"features/{split_name}/{feature_filename}"
        
        for sentence in img['sentences']:
            triplet = {
                'image_id': img['imgid'],
                'cocoid': img['cocoid'],
                'filename': img['filename'],
                'filepath': img['filepath'],
                'feature_path': feature_path,
                'caption': sentence['raw'].strip(),
                'sentid': sentence['sentid']
            }
            splits_data[split_name].append(triplet)
    
    stats = {}
    for split_name, samples in splits_data.items():
        output_path = os.path.join(output_dir, f'{split_name}_data.json')
        
        output_data = {
            'split': split_name,
            'num_samples': len(samples),
            'num_images': len(set(s['image_id'] for s in samples)),
            'samples': samples
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        stats[split_name] = {
            'samples': len(samples),
            'images': output_data['num_images'],
            'path': output_path
        }
        
        print(f"{split_name}: {len(samples)} samples from {output_data['num_images']} images")
        print(f"Saved to: {output_path}")
    
    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    return splits_data

if __name__ == '__main__':
    input_json_path = 'mscoco2014/karpathy_split/dataset_coco.json'
    output_dir = 'dataset'
    
    if not os.path.exists(input_json_path):
        print(f"Error: File not found at {input_json_path}")
        print("Please check the path and try again.")
    else:
        print(f"Found JSON file at: {input_json_path}")
        create_triplet_files(input_json_path, output_dir)