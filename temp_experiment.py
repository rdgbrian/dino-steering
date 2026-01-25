# %%
from src.object.anchor import SymbolicAnchor, AnchorConfig
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import torch
# %%
# Load all images from lake/images/samples
image_dir = Path("lake/images/samples")
image_paths = list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
print(f"Found {len(image_paths)} images")

# %%
# Display 2 images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for i, img_path in enumerate(image_paths[:2]):
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(img_path.name)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# %%
# Create config and extract anchor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
# %%
config = AnchorConfig(
    transforms=[
        {"type": "Identity", "n_samples": 1},
        {"type": "GaussianBlur", "radius_min": 0.5, "radius_max": 2.0, "n_samples": 5},
    ],
    anchor_dim=5,
    dino_model='dinov2_vits14',  # Use small model for faster prototyping
    device=device
)

distiller = SymbolicAnchor(config)
# Open images first
images = [Image.open(path) for path in image_paths]
# Now pass the opened images
anchor = distiller.extract_from_images(images)
anchor
# %%
# Test matching
test_img = image_paths[0]
distance = anchor.match_image(test_img)
print(f"Distance to anchor: {distance:.4f}")

# %%
anchor.shape
# %%
