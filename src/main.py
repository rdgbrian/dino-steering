# main.py
from object.embeddings import DINOEmbedder
from memory import VisualMemory
from segmentation import simple_grid_segments
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def scan_environment(image_dir, embedder, memory, segment_method='whole'):
    """
    Scan environment and build visual memory
    
    Args:
        image_dir: Directory with images
        embedder: DINOEmbedder instance
        memory: VisualMemory instance
        segment_method: 'whole', 'grid', or 'attention'
    """
    image_paths = sorted(Path(image_dir).glob('*.jpg')) + \
                  sorted(Path(image_dir).glob('*.png'))
    
    print(f"Scanning {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths):
        if segment_method == 'whole':
            # Just embed whole image
            emb = embedder.embed_image(str(img_path))
            memory.add_observation(emb, {
                'image_path': str(img_path),
                'type': 'whole_image'
            })
        
        elif segment_method == 'grid':
            # Grid segmentation
            bboxes = simple_grid_segments(str(img_path), grid_size=2)
            for bbox in bboxes:
                emb = embedder.embed_crop(str(img_path), bbox)
                memory.add_observation(emb, {
                    'image_path': str(img_path),
                    'bbox': bbox,
                    'type': 'grid_cell'
                })
    
    return memory

def visualize_memory(memory, output_path='results/clusters.png'):
    """Visualize memory clusters with t-SNE"""
    if len(memory.embeddings) < 2:
        print("Not enough observations to visualize")
        return
    
    # Run clustering first
    memory.cluster_memory()
    
    # t-SNE
    embeddings_matrix = np.array(memory.embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_matrix)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1],
        c=memory.clusters,
        cmap='tab10',
        alpha=0.6,
        s=100
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Visual Memory Clusters (t-SNE projection)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    plt.show()

def main():
    # Initialize
    print("Initializing DINO Visual Memory System...")
    embedder = DINOEmbedder()
    memory = VisualMemory(similarity_threshold=0.85)
    
    # Scan environment
    image_dir = 'data/apartment'  # Put your photos here
    scan_environment(image_dir, embedder, memory, segment_method='whole')
    
    # Analyze
    summary = memory.get_cluster_summary()
    print("\n=== Memory Summary ===")
    print(f"Total observations: {summary['num_observations']}")
    print(f"Discovered object types: {summary['num_object_types']}")
    print(f"Cluster sizes: {dict(summary['cluster_sizes'])}")
    
    # Visualize
    visualize_memory(memory)
    
    # Save
    memory.save('results/apartment_memory.pkl')

if __name__ == '__main__':
    main()