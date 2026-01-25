# transformations.py
from abc import ABC, abstractmethod
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_laplace
from typing import Union

class ImageTransform(ABC):
    """Abstract base class for image transformations"""
    
    def __init__(self, **kwargs):
        """
        Initialize transform with parameters
        
        Args:
            **kwargs: Transform-specific parameters
        """
        self.params = kwargs
    
    @abstractmethod
    def transform(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply transformation to image
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            PIL Image (transformed)
        """
        pass
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Allow calling transform as function"""
        return self.transform(image)


# Concrete implementations
class GaussianBlur(ImageTransform):
    """Apply Gaussian blur to image"""
    
    def __init__(self, radius_min=0.5, radius_max=2.0):
        super().__init__(radius_min=radius_min, radius_max=radius_max)
    
    def transform(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        radius = np.random.uniform(
            self.params['radius_min'], 
            self.params['radius_max']
        )
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomRotation(ImageTransform):
    """Rotate image by random angle"""
    
    def __init__(self, angle_min=-30, angle_max=30):
        super().__init__(angle_min=angle_min, angle_max=angle_max)
    
    def transform(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        angle = np.random.uniform(
            self.params['angle_min'],
            self.params['angle_max']
        )
        return image.rotate(angle, expand=False, fillcolor=(128, 128, 128))


class ColorJitter(ImageTransform):
    """Randomly change brightness, contrast, saturation"""
    
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3):
        super().__init__(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation
        )
    
    def transform(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Brightness
        if self.params['brightness'] > 0:
            factor = np.random.uniform(
                1 - self.params['brightness'],
                1 + self.params['brightness']
            )
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        # Contrast
        if self.params['contrast'] > 0:
            factor = np.random.uniform(
                1 - self.params['contrast'],
                1 + self.params['contrast']
            )
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        # Saturation
        if self.params['saturation'] > 0:
            factor = np.random.uniform(
                1 - self.params['saturation'],
                1 + self.params['saturation']
            )
            image = ImageEnhance.Color(image).enhance(factor)
        
        return image


class RandomCrop(ImageTransform):
    """Random crop with scale factor"""
    
    def __init__(self, scale_min=0.8, scale_max=1.0):
        super().__init__(scale_min=scale_min, scale_max=scale_max)
    
    def transform(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        w, h = image.size
        scale = np.random.uniform(
            self.params['scale_min'],
            self.params['scale_max']
        )
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        left = np.random.randint(0, w - new_w + 1) if new_w < w else 0
        top = np.random.randint(0, h - new_h + 1) if new_h < h else 0
        
        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BILINEAR)


class LaplacianOfGaussian(ImageTransform):
    """Apply Laplacian of Gaussian filter"""
    
    def __init__(self, sigma_min=0.5, sigma_max=2.0):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max)
    
    def transform(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        sigma = np.random.uniform(
            self.params['sigma_min'],
            self.params['sigma_max']
        )
        
        # Apply LoG to each channel
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = gaussian_laplace(image[:, :, i], sigma=sigma)
        
        # Normalize to [0, 255]
        result = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
        
        return Image.fromarray(result)


class AddNoise(ImageTransform):
    """Add Gaussian noise to image"""
    
    def __init__(self, std_min=0.01, std_max=0.1):
        super().__init__(std_min=std_min, std_max=std_max)
    
    def transform(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        std = np.random.uniform(
            self.params['std_min'],
            self.params['std_max']
        )
        
        noise = np.random.randn(*image.shape) * std * 255
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)


class Identity(ImageTransform):
    """No transformation (passthrough)"""
    
    def __init__(self):
        super().__init__()
    
    def transform(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image


class CompositeTransform(ImageTransform):
    """Apply multiple transforms in sequence"""
    
    def __init__(self, transforms):
        """
        Args:
            transforms: List of ImageTransform instances
        """
        super().__init__()
        self.transforms = transforms
    
    def transform(self, image):
        for t in self.transforms:
            image = t(image)
        return image