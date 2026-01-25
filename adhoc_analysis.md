# Ad-hoc Object Analysis via Temporal Visual Anchors

## Overview

A self-supervised pipeline for discovering, tracking, and recognizing objects in novel environments using video segmentation and subspace geometry.

---

## Pipeline Architecture

```
Video Burst (short trajectory)
    ↓
SAM 2 Temporal Tracking
    ↓
Multi-View DINO Embeddings per Object
    ↓
SVD-Based Symbolic Anchor Creation
    ↓
Subspace Similarity Matching
    ↓
Object Recognition & Discovery
```

---

## Phase 1: Temporal Segmentation & Tracking

### Input
Short video bursts captured during exploration (e.g., 2-5 seconds, ~30-60 frames)

### SAM 2 Processing
```
Frame 0:  [Object A] [Object B] [Object C]
Frame 5:  [Object A moved] [Object B] [Object C rotated]
Frame 10: [Object A] [Object B occluded] [Object C]
...
```

**Key Feature:** SAM 2 maintains object IDs across frames despite:
- Position changes
- Rotation/perspective shifts
- Partial occlusion
- Lighting variations

### Output
For each object in burst:
```
Object ID: 1
Frames: [0, 2, 5, 8, 11, ...]
Masks: [mask_0, mask_2, mask_5, ...]
BBoxes: [(x1,y1,x2,y2), ...]
```

---

## Phase 2: Multi-View Embedding Extraction

### Per-Object DINO Encoding

For each tracked object across its visible frames:

```
Object #1 trajectory:
    Frame 0  → Crop → DINO → emb_1_0  [1024-d]
    Frame 2  → Crop → DINO → emb_1_2  [1024-d]
    Frame 5  → Crop → DINO → emb_1_5  [1024-d]
    Frame 8  → Crop → DINO → emb_1_8  [1024-d]
    ...
    
Result: Embedding matrix X_1 = [N_views × 1024]
```

### Database Entry
```json
{
  "burst_id": "apartment_scan_001",
  "object_id": 1,
  "embeddings": [[...], [...], ...],  // N × 1024 matrix
  "frames": [0, 2, 5, 8, ...],
  "metadata": {
    "n_views": 12,
    "avg_bbox_size": 15680,
    "motion_type": "rotation"
  }
}
```

**Advantage:** Multiple views of same physical object = robust representation

---

## Phase 3: Intra-Burst Clustering

### Within-Burst Object Discovery

After processing entire burst:

```
Collected embeddings:
    Object #1: 12 views
    Object #2: 8 views
    Object #3: 15 views
    Object #4: 3 views (brief appearance)
    ...
```

### Similarity Analysis
```
For all pairs of objects in burst:
    Compute pairwise embedding similarity
    If similarity > threshold:
        → Same object type (different instances)
    Else:
        → Different object types
```

### Clustering Output
```
Cluster A: [Object #1, Object #5]  // Two chairs
Cluster B: [Object #2]             // One table
Cluster C: [Object #3, Object #4]  // Two lamps
```

**Result:** Discovered object type distribution in current scene

---

## Phase 4: SVD-Based Symbolic Anchor Creation

### The Core Innovation

**Problem:** How to represent an object type that generalizes across views and instances?

**Solution:** Compute the **principal subspace** spanned by multi-view embeddings

### Mathematical Formulation

For object with N views, embedding matrix **X** ∈ ℝ^(N×1024):

#### 1. Center the Data
```
X̄ = mean(X, axis=0)           // [1024]
X_centered = X - X̄             // [N × 1024]
```

#### 2. Singular Value Decomposition
```
X_centered = U Σ V^T

Where:
    U ∈ ℝ^(N×N)      : Left singular vectors (temporal modes)
    Σ ∈ ℝ^(N×1024)   : Singular values (importance)
    V^T ∈ ℝ^(1024×1024) : Right singular vectors (feature modes)
```

#### 3. Extract Principal Components
```
Top k components: V_k = V^T[:k, :]  // [k × 1024]

Typically k = 100 captures 95%+ variance
```

#### 4. Symbolic Anchor Representation
```
Anchor = {
    mean: X̄,                    // Center point [1024]
    components: V_k,            // Principal directions [100 × 1024]
    variance: Σ[:k]²,          // Explained variance per component
    n_views: N                  // Number of observations
}
```

### Geometric Interpretation

The symbolic anchor defines a **k-dimensional subspace** in the 1024-d embedding space.

```
            1024-d DINO Space
                    ↓
    [All possible visual features]
                    ↓
         Object Subspace (100-d)
    [Features that vary for this object]
                    ↓
            Anchor = Subspace
```

**Intuition:** Different views of the same object type span a low-dimensional manifold in feature space.

---

## Phase 5: Subspace Projection & Matching

### Projecting New Observations

Given a new embedding **e_new** ∈ ℝ^1024:

#### 1. Center
```
e_centered = e_new - Anchor.mean
```

#### 2. Project to Subspace
```
projections = e_centered @ V_k^T    // [100]

These are coordinates in the anchor's subspace
```

#### 3. Compute Angular Distances
```
For each principal component i:
    θ_i = arccos(projection_i / ||projection_i||)
    
Angular signature: θ = [θ_1, θ_2, ..., θ_k]
```

### Similarity Metrics

#### Option A: Subspace Angle (Grassmann Distance)
```
For two anchors A₁ and A₂:
    Compute principal angles between subspaces
    
Distance = √(Σ sin²(θ_i))

Where θ_i are canonical angles between subspaces
```

#### Option B: Reconstruction Error
```
Project new embedding to anchor subspace:
    e_reconstructed = Anchor.mean + (e_centered @ V_k^T) @ V_k
    
Error = ||e_new - e_reconstructed||²

Low error → Similar to anchor
High error → Different from anchor
```

#### Option C: Cosine Similarity in Subspace
```
Project both embeddings to same subspace
Compute cosine similarity of projections

sim = (proj_1 · proj_2) / (||proj_1|| ||proj_2||)
```

---

## Phase 6: Cross-Burst Recognition

### Recognition Pipeline

```
New video burst arrives
    ↓
SAM 2 tracking → Objects [1, 2, 3, ...]
    ↓
For each object:
    Extract multi-view embeddings
    Create temporary anchor
    ↓
    Compare to Anchor Database:
        For each existing anchor:
            Compute subspace similarity
            
    If max_similarity > threshold:
        → Match! (Object recognized)
        → Update anchor with new views
    Else:
        → New object type discovered
        → Add new anchor to database
```

### Database Update Strategy

#### Incremental Learning
```
When object recognized:
    1. Merge new embeddings with existing
    2. Recompute SVD (or incremental update)
    3. Refine anchor representation
    
Result: Anchors become more robust over time
```

#### Online Statistics
```
For each anchor:
    - Total observations: N
    - Unique bursts seen: M
    - Last seen: timestamp
    - Confidence: f(N, variance)
```

---

## Ad-hoc Analysis Capabilities

### 1. Object Type Discovery
```
Question: "What types of objects did I see?"

Answer:
    - Number of unique anchors created
    - Cluster sizes (instances per type)
    - Visualization: t-SNE of anchor centroids
```

### 2. Instance Counting
```
Question: "How many chairs are in this room?"

Answer:
    - Count objects matching "chair" anchor
    - Account for re-observations (temporal tracking)
    - Disambiguate multiple instances
```

### 3. Novel Object Detection
```
Question: "Have I seen this before?"

Answer:
    - Project to all known anchors
    - Compute similarity scores
    - Threshold-based decision:
        High similarity → Known
        Low similarity → Novel
```

### 4. Cross-Environment Recognition
```
Scenario: Apartment → Office → Beach

For each new environment:
    - Recognize shared object types (chairs exist everywhere)
    - Discover environment-specific objects
    - Compute overlap statistics
    
Output: Venn diagram of object types across environments
```

### 5. Temporal Object History
```
Question: "When did I last see this object type?"

Query anchor database by:
    - Similarity to current observation
    - Temporal metadata
    - Spatial context (if available)
```

### 6. Object Variability Analysis
```
For each anchor:
    Variance explained by top k components
    
High variance → Object appears in many different ways
Low variance → Object appearance is consistent

Applications:
    - Identify lighting-sensitive objects
    - Detect objects with articulation
    - Understand viewpoint dependence
```

---

## Mathematical Properties

### 1. View-Invariance
Multi-view SVD captures intrinsic object properties, not viewpoint-specific features.

### 2. Dimensionality Reduction
```
1024-d → 100-d: 10x compression
Typically retains 95%+ information
```

### 3. Subspace Stability
As more views are added, subspace converges to true object manifold.

### 4. Metric Space
Grassmann distance between subspaces defines a proper metric:
- d(A, A) = 0
- d(A, B) = d(B, A)
- Triangle inequality holds

---

## Advantages Over Single-View Approaches

| Aspect | Single-View | Multi-View + SVD |
|--------|-------------|------------------|
| **Robustness** | Viewpoint-dependent | View-invariant |
| **Representation** | Single point | Subspace manifold |
| **Noise handling** | Sensitive | Averaged out |
| **Matching** | Pairwise distance | Subspace similarity |
| **Interpretability** | Embedding vector | Principal variations |
| **Generalization** | Limited | Strong |

---

## Failure Modes & Limitations

### 1. Short Bursts
**Issue:** Few views (N < 5) → Poor subspace estimate

**Mitigation:**
- Require minimum N views for anchor creation
- Use single-view matching for brief observations
- Flag low-confidence anchors

### 2. Motion Blur
**Issue:** Fast motion → Poor DINO features

**Mitigation:**
- Frame selection based on sharpness
- Optical flow-based quality filtering

### 3. Severe Occlusion
**Issue:** Partial views → Incomplete embeddings

**Mitigation:**
- SAM 2 handles partial occlusion well
- SVD robust to some view corruption
- Weight views by visibility confidence

### 4. Identical Object Instances
**Issue:** Two identical chairs in same burst

**Mitigation:**
- Spatial clustering in addition to feature clustering
- Track separate instances via SAM 2 IDs
- Merge only if spatially impossible to be different objects

---

## Computational Complexity

### Per-Burst Processing
```
SAM 2 tracking: O(N_frames × H × W)     // Video segmentation
DINO embedding: O(N_objects × N_views)  // Per-object encoding
SVD computation: O(N_views × 1024²)     // Anchor creation
Database matching: O(N_anchors × 100²)  // Subspace comparison

Bottleneck: SAM 2 video segmentation
```

### Scalability
```
Anchors in memory: ~100KB each
1000 anchors ≈ 100MB

Database operations: Sublinear via indexing (FAISS, ANN)
```

---

## Extensions & Future Work

### 1. Hierarchical Anchors
```
Super-anchor: "furniture"
    ↳ Sub-anchor: "chair"
        ↳ Instance: "office chair #1"
```

### 2. Anchor Fusion
Merge anchors from multiple environments to create universal object representations.

### 3. Active Learning
Select informative viewpoints to maximize anchor quality.

### 4. Semantic Bootstrapping
Optional: Attach language labels to anchors post-hoc for human interpretability.

### 5. Continuous Learning
Update anchors in real-time as robot explores, with forgetting mechanisms for outdated objects.

---

## Implementation Pseudocode

```python
class TemporalVisualMemory:
    def __init__(self):
        self.anchor_db = {}
        self.burst_history = []
    
    def process_burst(self, video_frames):
        # Phase 1: Track objects
        tracked_objects = sam2.track_video(video_frames)
        
        # Phase 2: Extract embeddings
        for obj in tracked_objects:
            embeddings = []
            for frame_id in obj.visible_frames:
                emb = dino.embed(video_frames[frame_id], obj.bbox)
                embeddings.append(emb)
            
            obj.embeddings = np.array(embeddings)  # [N × 1024]
        
        # Phase 3: Intra-burst clustering
        clusters = self.cluster_objects(tracked_objects)
        
        # Phase 4: Create/update anchors
        for cluster in clusters:
            anchor = self.create_anchor(cluster.embeddings)
            
            # Phase 5: Match to database
            match = self.find_match(anchor)
            
            if match:
                self.update_anchor(match.id, anchor)
            else:
                self.add_anchor(anchor)
        
        return self.analyze()
    
    def create_anchor(self, embeddings, k=100):
        mean = embeddings.mean(axis=0)
        X_centered = embeddings - mean
        U, S, Vt = svd(X_centered)
        
        return {
            'mean': mean,
            'components': Vt[:k],
            'variance': S[:k]**2
        }
    
    def find_match(self, anchor, threshold=0.85):
        best_score = -1
        best_match = None
        
        for anchor_id, db_anchor in self.anchor_db.items():
            similarity = grassmann_distance(
                anchor['components'], 
                db_anchor['components']
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match = anchor_id
        
        return best_match if best_score > threshold else None
    
    def analyze(self):
        return {
            'n_unique_objects': len(self.anchor_db),
            'total_observations': sum(a['n_views'] for a in self.anchor_db.values()),
            'object_distribution': self.get_cluster_sizes()
        }
```

---

## Summary

**The ad-hoc analysis system enables:**

1. **Real-time object discovery** without pre-defined categories
2. **Robust recognition** across viewpoints and environments  
3. **Principled matching** via subspace geometry
4. **Interpretable representations** through SVD components
5. **Online learning** with incremental anchor updates
6. **Zero labels** required at any stage

**Key innovation:** Treating objects as subspaces rather than points in embedding space, leveraging temporal tracking for multi-view data collection.

This is **mathematically principled**, **computationally tractable**, and **practically useful** for robotics applications. 🚀