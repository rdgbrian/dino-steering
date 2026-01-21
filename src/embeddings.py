# embeddings.py
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class DINOEmbedder:
    def __init__(self, model_name='dinov2_vitl14', device='cuda'):
        """Load DINO model for embedding extraction"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading DINO model on {self.device}...")
        
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def embed_image(self, image_path):
        """Extract embedding for entire image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def embed_crop(self, image_path, bbox):
        """Extract embedding for cropped region (x1, y1, x2, y2)"""
        img = Image.open(image_path).convert('RGB')
        cropped = img.crop(bbox)
        
        img_tensor = self.transform(cropped).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def embed_masked(self, image_path, mask):
        """Extract embedding for masked region"""
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Apply mask (set background to mean color)
        masked_img = img_array.copy()
        masked_img[~mask] = [128, 128, 128]  # Gray background
        
        masked_pil = Image.fromarray(masked_img)
        img_tensor = self.transform(masked_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        return embedding.cpu().numpy()[0]