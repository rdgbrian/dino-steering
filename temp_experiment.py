# %%
from src.object.anchor import SymbolicAnchor, SimpleAnchor, AnchorConfig
from src.object.transformations import GaussianBlur
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
        {"type": "GaussianBlur", "radius_min": 50, "radius_max": 100, "n_samples": 20},
    ],
    anchor_dim=5,
    dino_model='dinov2_vits14',  # Use small model for faster prototyping
    device=device
)

distiller = SymbolicAnchor(config)
images = [Image.open(path) for path in image_paths]
anchor = distiller.extract_from_images(images)

# %%
# Inspect anchor
print(f"Basis shape: {anchor.anchor_basis.shape}")   # (d, d) — full decomposition
print(f"Active r:    {anchor.anchor_dim}")            # top-r used for projection
print(f"Variance explained: {anchor.explained_variance_ratio():.2%}")

# %%
# Change r without re-running extraction
anchor.anchor_dim = 5
print(f"r=10 variance explained: {anchor.explained_variance_ratio():.2%}")

# %%
# --- Sanity check: SimpleAnchor vs SymbolicAnchor on a blurred image ---

# Train SimpleAnchor on a single image
simple_anchor = SimpleAnchor(dino_model='dinov2_vits14', device=device)
simple_anchor.extract_from_images([images[0]])

# SymbolicAnchor already trained above (anchor, all images, r=10)

# %%
# Create original and blurred test image
test_img = images[0]
blur = GaussianBlur(radius_min=100, radius_max=100)  # fixed radius for reproducibility
blurred_img = blur(test_img)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(test_img)
axes[0].set_title("Original")
axes[0].axis('off')
axes[1].imshow(blurred_img)
axes[1].set_title("Blurred")
axes[1].axis('off')
plt.tight_layout()
plt.show()

# %%
# Compare match scores for both anchors
orig_simple   = simple_anchor.match_image(test_img)
orig_symbolic = anchor.match_image(test_img)
blur_simple   = simple_anchor.match_image(blurred_img)
blur_symbolic = anchor.match_image(blurred_img)

print(f"{'':20s}  {'SimpleAnchor':>14}  {'SymbolicAnchor':>14}")
print(f"{'Original':20s}  {orig_simple:>14.4f}  {orig_symbolic:>14.4f}")
print(f"{'Blurred':20s}  {blur_simple:>14.4f}  {blur_symbolic:>14.4f}")
print(f"{'Drop':20s}  {orig_simple - blur_simple:>14.4f}  {orig_symbolic - blur_symbolic:>14.4f}")

# %%
