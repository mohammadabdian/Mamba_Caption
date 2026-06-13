# feature_pipeline/feature_extractor.py
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from transformers import CLIPVisionModel, CLIPProcessor

class ClipEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        super(ClipEncoder, self).__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        
        try:
            self.vision_model = CLIPVisionModel.from_pretrained(
                model_name, 
                local_files_only=True
            ).to(self.device)
            
            self.processor = CLIPProcessor.from_pretrained(
                model_name,
                local_files_only=True
            )
        except Exception as e:
            print(f"Error loading CLIP model from cache: {e}")
            print("Trying to download from mirror...")
            os.environ['HF_ENDPOINT'] = 'https://hf.devneeds.ir/'
            self.vision_model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        self.vision_model.eval()

    def preprocess_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Warning: Error loading {image_path}: {e}")
            return None

    def forward(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = []
        for path in image_paths:
            img = self.preprocess_image(path)
            if img is not None:
                images.append(img)

        if not images:
            return None

        try:
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.vision_model(**inputs)
                features = outputs.last_hidden_state
            
            return features
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return None