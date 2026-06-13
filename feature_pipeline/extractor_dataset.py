# feature_pipeline/extractor_dataset.py
import json
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, json_path: str, split: str = 'train'):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            self.image_paths = []
            self.image_ids = []
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_path}")
            self.image_paths = []
            self.image_ids = []
            return
        
        self.image_paths = []
        self.image_ids = []
        
        for img in data['images']:
            try:
                if split == 'train' and img['split'] in ['train', 'restval']:
                    self.image_paths.append(f"{img['filepath']}/{img['filename']}")
                    self.image_ids.append(img['imgid'])
                elif split == 'val' and img['split'] == 'val':
                    self.image_paths.append(f"{img['filepath']}/{img['filename']}")
                    self.image_ids.append(img['imgid'])
                elif split == 'test' and img['split'] == 'test':
                    self.image_paths.append(f"{img['filepath']}/{img['filename']}")
                    self.image_ids.append(img['imgid'])
            except KeyError as e:
                print(f"Warning: Missing key {e} in image entry, skipping")
                continue
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            return {
                'image_path': self.image_paths[idx],
                'image_id': self.image_ids[idx]
            }
        except IndexError:
            print(f"Error: Index {idx} out of range")
            return None