# feature_pipeline/feature_saver.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class FeatureSaver:
    def __init__(self, dataset, encoder, images_root, features_dir,
                 batch_size=64, num_workers=8, device=None, use_amp=True):
        self.dataset = dataset
        self.encoder = encoder
        self.images_root = images_root
        self.features_dir = features_dir
        os.makedirs(self.features_dir, exist_ok=True)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == 'cuda'
        
        self.encoder.to(self.device)
        self.encoder.eval()

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if num_workers > 0 else False
        )
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_ids': []
        }
    
    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return [], []
        paths = [item['image_path'] for item in batch]
        ids = [item['image_id'] for item in batch]
        return paths, ids

    def run(self):
        for batch_paths, batch_ids in tqdm(self.dataloader, desc="Extracting Features"):
            if not batch_paths:
                continue
            
            full_paths = [os.path.join(self.images_root, path) for path in batch_paths]
            
            try:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        features = self.encoder(full_paths)
                else:
                    features = self.encoder(full_paths)
            except Exception as e:
                print(f"Error encoding batch: {e}")
                for img_id in batch_ids:
                    self.stats['failed'] += 1
                    self.stats['failed_ids'].append(img_id)
                continue
            
            if features is None:
                for img_id in batch_ids:
                    self.stats['failed'] += 1
                    self.stats['failed_ids'].append(img_id)
                continue
            
            features_cpu = features.cpu()
            
            for i, img_id in enumerate(batch_ids):
                try:
                    save_path = os.path.join(self.features_dir, f"{img_id}.npz")
                    
                    if i >= features_cpu.shape[0]:
                        self.stats['failed'] += 1
                        self.stats['failed_ids'].append(img_id)
                        continue
                    
                    feature_np = features_cpu[i].half().numpy()
                    np.savez_compressed(save_path, feature=feature_np)
                    self.stats['success'] += 1
                    
                except Exception as e:
                    print(f"Error saving feature for image_id {img_id}: {e}")
                    self.stats['failed'] += 1
                    self.stats['failed_ids'].append(img_id)
            
            self.stats['total'] += len(batch_ids)
            
            if self.device.type == 'cuda' and self.stats['total'] % 5000 == 0:
                torch.cuda.empty_cache()
        
        self._print_summary()
    
    def _print_summary(self):
        print("\n" + "="*50)
        print("Feature Extraction Summary")
        print("="*50)
        print(f"Total images processed: {self.stats['total']}")
        print(f"Successful: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['failed'] > 0:
            print(f"Success rate: {100 * self.stats['success'] / self.stats['total']:.2f}%")
        
        if self.stats['failed_ids']:
            print(f"\nFailed image IDs (first 10): {self.stats['failed_ids'][:10]}")
            failed_log = os.path.join(self.features_dir, 'failed_ids.txt')
            with open(failed_log, 'w') as f:
                for img_id in self.stats['failed_ids']:
                    f.write(f"{img_id}\n")
            print(f"Full list saved to: {failed_log}")
        
        print(f"Features saved to: {self.features_dir}")