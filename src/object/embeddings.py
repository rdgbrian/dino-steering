# embeddings.py
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

class DINOEmbedder:
    def __init__(self, model_name='dinov2_vitl14', device='cuda'):
        """Load DINO model for embedding extraction"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading DINO model on {self.device}...")
        
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Standard DINO preprocessing
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def embed_batch(self, images):
        """
        Extract embeddings for a batch of images
        
        Args:
            images: List of PIL Images or numpy arrays (H, W, 3)
        
        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        # Transform all images - this handles resizing/cropping to same size
        img_tensors = []
        for img in images:
            # Convert numpy to PIL if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            # Apply transform (handles resize, crop, normalize)
            img_tensor = self.transform(img)
            img_tensors.append(img_tensor)
        
        # Stack into batch (all tensors are now same size: [3, 224, 224])
        batch = torch.stack(img_tensors).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model(batch)
        
        return embeddings.cpu().numpy()
    
    def embed_single(self, image):
        """
        Extract embedding for a single image
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.embed_batch([image])
        return embeddings[0]