# feature_pipeline/run.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import traceback
from pathlib import Path
from feature_pipeline.extractor_dataset import COCODataset
from feature_pipeline.feature_saver import FeatureSaver
from feature_pipeline.feature_extractor import ClipEncoder

def get_project_root():
    return Path(__file__).parent.parent

def prepare_paths(root):
    dataset_root = root / 'dataset' / 'mscoco2014'
    images_root = dataset_root
    coco_json = dataset_root / 'karpathy_split' / 'dataset_coco.json'
    
    features_root = root / 'features'
    
    train_features = features_root / 'train'
    val_features = features_root / 'val'
    test_features = features_root / 'test'
    
    train_features.mkdir(parents=True, exist_ok=True)
    val_features.mkdir(parents=True, exist_ok=True)
    test_features.mkdir(parents=True, exist_ok=True)
    
    return {
        'images_root': images_root,
        'coco_json': coco_json,
        'train_features': train_features,
        'val_features': val_features,
        'test_features': test_features,
    }

def get_optimal_batch_size():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory > 20:
            return 128
        elif gpu_memory > 10:
            return 64
        else:
            return 32
    return 16

def extract_split(dataset, encoder, images_root, features_dir, split_name):
    print("\n" + "="*50)
    print(f"Extracting features for {split_name.upper()} split")
    print("="*50)
    
    try:
        saver = FeatureSaver(
            dataset, encoder,
            str(images_root),
            str(features_dir),
            batch_size=get_optimal_batch_size(),
            num_workers=8,
            use_amp=True
        )
        saver.run()
        return True
    except Exception as e:
        print(f"Error extracting features for {split_name}: {e}")
        traceback.print_exc()
        return False

def main():
    try:
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("No GPU found, using CPU (will be slow)")
        
        root = get_project_root()
        paths = prepare_paths(root)
        
        if not paths['coco_json'].exists():
            print(f"Error: COCO JSON file not found at {paths['coco_json']}")
            return
        
        print("Loading datasets...")
        train_dataset = COCODataset(str(paths['coco_json']), split='train')
        val_dataset = COCODataset(str(paths['coco_json']), split='val')
        test_dataset = COCODataset(str(paths['coco_json']), split='test')
        
        if len(train_dataset) == 0:
            print("Error: No training images found")
            return
        
        print("Loading CLIP encoder...")
        encoder = ClipEncoder(device='cuda')
        
        results = {}
        results['train'] = extract_split(train_dataset, encoder, paths['images_root'], paths['train_features'], 'train')
        results['val'] = extract_split(val_dataset, encoder, paths['images_root'], paths['val_features'], 'val')
        results['test'] = extract_split(test_dataset, encoder, paths['images_root'], paths['test_features'], 'test')
        
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        for split, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {split}: {'Completed' if success else 'Failed'}")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()