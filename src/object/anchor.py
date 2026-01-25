# anchor.py
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Union
import json
import numpy as np
from PIL import Image
import pickle

from src.object.embeddings import DINOEmbedder
from src.object.transformations import (
    ImageTransform, GaussianBlur, RandomRotation, 
    ColorJitter, RandomCrop, LaplacianOfGaussian, 
    AddNoise, Identity
)


@dataclass
class AnchorConfig:
    """Configuration for symbolic anchor extraction"""
    
    # Transformation settings - each specifies n_samples
    transforms: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"type": "Identity", "n_samples": 1},  # Original, just once
        {"type": "GaussianBlur", "radius_min": 0.5, "radius_max": 2.0, "n_samples": 5},
        # {"type": "RandomRotation", "angle_min": -30, "angle_max": 30, "n_samples": 5},
        # {"type": "ColorJitter", "brightness": 0.3, "contrast": 0.3, "n_samples": 4},
        # {"type": "RandomCrop", "scale_min": 0.8, "scale_max": 1.0, "n_samples": 3},
    ])
    
    # Anchor dimensionality
    anchor_dim: int = 2  # Number of top eigenvectors (r)
    
    # DINO settings
    dino_model: str = 'dinov2_vitl14'
    device: str = 'cuda'
    
    @property
    def total_augmentations(self) -> int:
        """Total number of augmented samples that will be generated"""
        return sum(t.get("n_samples", 1) for t in self.transforms)
    
    def to_json(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls(**config_dict)
        
        # Validate anchor_dim
        if config.anchor_dim > config.total_augmentations:
            raise ValueError(
                f"anchor_dim ({config.anchor_dim}) cannot exceed total augmentations ({config.total_augmentations}). "
                f"Reduce anchor_dim or increase n_samples in transforms."
            )
        
        return config
    
    def instantiate_transforms(self) -> List[tuple[ImageTransform, int]]:
        """
        Convert transform configs to actual transform objects with sample counts
        
        Returns:
            List of (transform_object, n_samples) tuples
        """
        transform_map = {
            "Identity": Identity,
            "GaussianBlur": GaussianBlur,
            "RandomRotation": RandomRotation,
            "ColorJitter": ColorJitter,
            "RandomCrop": RandomCrop,
            "LaplacianOfGaussian": LaplacianOfGaussian,
            "AddNoise": AddNoise,
        }
        
        transform_objects = []
        for t_config in self.transforms:
            t_type = t_config["type"]
            n_samples = t_config.get("n_samples", 1)
            
            # Extract params (everything except type and n_samples)
            t_params = {k: v for k, v in t_config.items() 
                       if k not in ["type", "n_samples"]}
            
            transform_obj = transform_map[t_type](**t_params)
            transform_objects.append((transform_obj, n_samples))
        
        return transform_objects


class SymbolicAnchor:
    """
    Symbolic anchor: a subspace in embedding space representing an object
    """
    
    def __init__(self, config: AnchorConfig):
        """
        Args:
            config: AnchorConfig specifying how to extract anchor
        """
        self.config = config
        self.embedder = DINOEmbedder(
            model_name=config.dino_model,
            device=config.device
        )
        
        # Anchor components (set after extraction)
        self.anchor_basis = None  # (d, r) array - top r eigenvectors
        self.eigenvalues = None   # (r,) array - corresponding eigenvalues
        self.mean_embedding = None  # (d,) array - mean of training embeddings
        
        # Metadata
        self.n_samples = 0
        self.embedding_dim = None
    
    def extract_from_images(self, images: List[Union[Image.Image, str]]):
        """
        Extract symbolic anchor from a list of images
        
        Args:
            images: List of PIL Images or image paths
        """
        # Load images if paths provided
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            else:
                pil_images.append(img)
        
        # Generate augmented images
        augmented_images = self._augment_images(pil_images)
        
        # Embed all images
        print(f"Embedding {len(augmented_images)} images...")
        embeddings = self.embedder.embed_batch(augmented_images)
        
        # Extract anchor via V^T V eigendecomposition
        anchor = self._compute_anchor(embeddings)
        
        print(f"Anchor extracted: {self.anchor_dim}D subspace in {self.embedding_dim}D space")
        print(f"Top {self.anchor_dim} eigenvalues explain {self.explained_variance_ratio():.2%} of variance")

        return anchor
    
    def _augment_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Apply transformations according to config
        
        Each transform is applied n_samples times (with random params if applicable)
        """
        transforms_with_counts = self.config.instantiate_transforms()
        augmented = []
        
        for img in images:
            # Apply each transform the specified number of times
            for transform, n_samples in transforms_with_counts:
                for _ in range(n_samples):
                    aug_img = transform(img)
                    augmented.append(aug_img)
        
        print(f"Generated {len(augmented)} augmented samples from {len(images)} images")
        print(f"  Per image: {len(augmented) // len(images)} augmentations")
        
        return augmented
    
    def _compute_anchor(self, embeddings: np.ndarray):
        """
        Compute anchor via V^T V eigendecomposition
        
        Args:
            embeddings: (n, d) array of DINO embeddings
        """
        V = embeddings  # (n, d)
        self.n_samples = V.shape[0]
        self.embedding_dim = V.shape[1]
        
        # Center embeddings
        self.mean_embedding = V.mean(axis=0)
        V_centered = V - self.mean_embedding
        
        # Compute covariance matrix V^T V
        VTV = V_centered.T @ V_centered / (self.n_samples - 1)  # (d, d)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(VTV)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top r eigenvectors
        r = self.config.anchor_dim
        self.anchor_basis = eigenvectors[:, :r]  # (d, r)
        self.eigenvalues = eigenvalues[:r]       # (r,)
        
        # Print eigenvalue distribution
        print(f"Top 5 eigenvalues: {eigenvalues[:5]}")
    
    def project(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding onto anchor subspace
        
        Args:
            embedding: (d,) array
        
        Returns:
            (d,) array - projection onto subspace
        """
        if self.anchor_basis is None:
            raise ValueError("Anchor not yet extracted. Call extract_from_images first.")
        
        # Center
        emb_centered = embedding - self.mean_embedding
        
        # Project onto subspace
        projection = self.anchor_basis @ (self.anchor_basis.T @ emb_centered)
        
        return projection + self.mean_embedding
    
    def distance_to_subspace(self, embedding: np.ndarray) -> float:
        """
        Compute distance from embedding to anchor subspace
        
        Args:
            embedding: (d,) array
        
        Returns:
            float - L2 distance to subspace (reconstruction error)
        """
        projection = self.project(embedding)
        return np.linalg.norm(embedding - projection)
    
    def match_image(self, image: Union[Image.Image, str]) -> float:
        """
        Compute how well an image matches this anchor
        
        Args:
            image: PIL Image or image path
        
        Returns:
            float - distance to anchor subspace (lower = better match)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        embedding = self.embedder.embed_single(image)
        return self.distance_to_subspace(embedding)
    
    def explained_variance_ratio(self) -> float:
        """Fraction of total variance explained by anchor subspace"""
        if self.eigenvalues is None:
            return 0.0
        
        # Need all eigenvalues to compute total variance
        # For now, approximate as ratio of top-r to sum of top-100
        # (proper version would need all d eigenvalues)
        return self.eigenvalues.sum() / self.eigenvalues.sum()
    
    @property
    def anchor_dim(self) -> int:
        """Dimensionality of anchor subspace"""
        return self.config.anchor_dim
    
    def save(self, path: str):
        """Save anchor to file"""
        data = {
            'config': asdict(self.config),
            'anchor_basis': self.anchor_basis,
            'eigenvalues': self.eigenvalues,
            'mean_embedding': self.mean_embedding,
            'n_samples': self.n_samples,
            'embedding_dim': self.embedding_dim,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Anchor saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load anchor from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        config = AnchorConfig(**data['config'])
        anchor = cls(config)
        
        anchor.anchor_basis = data['anchor_basis']
        anchor.eigenvalues = data['eigenvalues']
        anchor.mean_embedding = data['mean_embedding']
        anchor.n_samples = data['n_samples']
        anchor.embedding_dim = data['embedding_dim']
        
        print(f"Anchor loaded from {path}")
        print(f"  Subspace: {anchor.anchor_dim}D in {anchor.embedding_dim}D")
        print(f"  Trained on: {anchor.n_samples} samples")
        
        return anchor


# Utility for managing multiple anchors
class AnchorDatabase:
    """Manage multiple symbolic anchors for different objects"""
    
    def __init__(self):
        self.anchors = {}  # object_id -> SymbolicAnchor
    
    def add_anchor(self, object_id: str, anchor: SymbolicAnchor):
        """Add anchor for an object"""
        self.anchors[object_id] = anchor
    
    def match_image(self, image: Union[Image.Image, str]) -> Dict[str, float]:
        """
        Match image against all anchors
        
        Returns:
            dict mapping object_id -> distance
        """
        distances = {}
        for obj_id, anchor in self.anchors.items():
            distances[obj_id] = anchor.match_image(image)
        return distances
    
    def classify_image(self, image: Union[Image.Image, str]) -> str:
        """
        Classify image to nearest anchor
        
        Returns:
            object_id of best matching anchor
        """
        distances = self.match_image(image)
        return min(distances, key=distances.get)
    
    def save(self, directory: str):
        """Save all anchors to directory"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for obj_id, anchor in self.anchors.items():
            anchor.save(f"{directory}/{obj_id}.pkl")
        
        print(f"Saved {len(self.anchors)} anchors to {directory}")
    
    @classmethod
    def load(cls, directory: str):
        """Load all anchors from directory"""
        import os
        db = cls()
        
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                obj_id = filename[:-4]
                anchor = SymbolicAnchor.load(f"{directory}/{filename}")
                db.add_anchor(obj_id, anchor)
        
        print(f"Loaded {len(db.anchors)} anchors from {directory}")
        return db