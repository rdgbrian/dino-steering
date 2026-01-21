# segment_and_embed.py
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

class SegmentEmbedder:
    def __init__(self, sam_checkpoint="sam_vit_h.pth", dino_model='dinov2_vitl14'):
        print("Loading SAM...")
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.cuda().eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        print("Loading DINO...")
        self.dino = torch.hub.load('facebookresearch/dinov2', dino_model)
        self.dino.cuda().eval()
        
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_bbox_from_mask(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return (cmin, rmin, cmax, rmax)
    
    def embed_segment(self, image, bbox):
        """Crop and embed a segment"""
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))
        
        img_tensor = self.transform(cropped).unsqueeze(0).cuda()
        
        with torch.no_grad():
            embedding = self.dino(img_tensor).cpu().numpy()[0]
        
        return embedding
    
    def process_image(self, image_path):
        """Get all segment embeddings from an image"""
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Get masks
        masks = self.mask_generator.generate(image_np)
        
        segments = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            bbox = self.get_bbox_from_mask(mask)
            
            if bbox is None:
                continue
            
            # Embed
            embedding = self.embed_segment(image, bbox)
            
            segments.append({
                'embedding': embedding,
                'bbox': bbox,
                'area': mask_data.get('area', 0),
                'segment_id': i
            })
        
        return segments

# Usage
embedder = SegmentEmbedder()
segments = embedder.process_image('dino-steering/images/real/coke.png')
print(f"Extracted {len(segments)} segment embeddings")