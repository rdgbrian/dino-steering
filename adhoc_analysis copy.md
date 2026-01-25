# Ad-hoc Object Analysis via Temporal Visual Anchors

A self-supervised pipeline for discovering, tracking, and recognizing objects in novel environments by representing entities as **evolving geometric subspaces**.

---

## 1. Pipeline Architecture

The system transitions from raw pixels to a compressed, symbolic geometric representation.



1.  **Video Burst:** Short trajectory (2–5s) captured during exploration.
2.  **Temporal Tracking (SAM 2):** Maintains consistent object IDs and pixel-perfect masks across frames.
3.  **Feature Extraction (DINOv2):** Generates high-dimensional semantic embeddings ($d=1024$).
4.  **Anchor Synthesis (Incremental SVD):** Computes the principal subspace of the object’s appearance.
5.  **Manifold Matching:** Uses Grassmann distance or Mutual Subspace Method (MSM) for recognition.

---

## 2. Phase: Temporal Segmentation & Embedding

### Input & Tracking
SAM 2 processes the burst to generate a set of masked crops for each object $O_i$. Unlike static detectors, this captures the object's **visual variance** (rotations, lighting shifts, and perspective changes) over time.

### Multi-View Embedding
For each object trajectory, we extract a sequence of embeddings:
$$X = [e_1, e_2, \dots, e_N]^T \in \mathbb{R}^{N \times 1024}$$

* **Refinement:** Use **RoI Align** on DINOv2 feature maps to ensure the embedding is strictly focused on the SAM 2 mask, preventing background "noise" from polluting the subspace.

---

## 3. Phase: Symbolic Anchor Creation (The Innovation)

Instead of a single "average" vector, an object is defined by the **subspace** it spans.

### Mathematical Formulation
To create an anchor, we perform Singular Value Decomposition (SVD) on the centered embedding matrix $\bar{X}$:
$$\bar{X} = U \Sigma V^T$$

**Key Components of the Anchor:**
* **Centroid ($\mu$):** The average appearance $\in \mathbb{R}^{1024}$.
* **Basis ($V_k$):** The top $k$ right-singular vectors representing the "axes" of visual change.
* **Energy ($\sigma^2$):** Singular values representing the "weight" or importance of each axis.

### Improvement: Dynamic Rank Selection
Rather than a fixed $k=100$, we select $k$ such that we capture a specific ratio of variance (e.g., 95%):
$$k = \text{argmin}_k \left( \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{j=1}^N \sigma_j^2} \geq 0.95 \right)$$



* **Result:** Simple objects (like a soccer ball) receive a low-rank anchor; complex objects (like a houseplant) receive a high-rank anchor.

---

## 4. Phase: Matching & Recognition

To compare a new observation (or a new burst) against the database, we use the **Mutual Subspace Method (MSM)**.

### Subspace Similarity
Given a stored Anchor $A$ and a new candidate Anchor $B$, we calculate the **Canonical Angles** $\theta$ between their subspaces:
$$\text{Sim}(A, B) = \frac{1}{k} \sum_{i=1}^k \cos^2(\theta_i)$$



* **Low Similarity:** New object discovery.
* **High Similarity:** Instance recognition. The new views are merged into the existing anchor.

### Improvement: Incremental Updates
To avoid re-computing SVD from scratch as more data arrives, we use **Incremental PCA (IPCA)**. This allows the anchor to "learn" new viewpoints online without the need to store every raw embedding from the past.

---

## 5. Ad-hoc Analysis Capabilities

| Capability | Logic |
| :--- | :--- |
| **Object Discovery** | Identifying clusters of subspaces that do not match the existing database. |
| **Instance Counting** | Disambiguating objects by checking if they share a subspace but occupy different spatial coordinates. |
| **Novelty Detection** | High reconstruction error when projecting a new view onto all known subspaces. |
| **Variability Analysis** | Using the ratio of $\sigma_1 / \sigma_k$ to determine if an object is "viewpoint-stable" or highly complex. |

---

## 6. Comparison of Approaches

| Feature | Point-Cloud Embeddings | **Temporal Visual Anchors** |
| :--- | :--- | :--- |
| **Representation** | Static Centroid | Dynamic Subspace |
| **Robustness** | Weak to pose/lighting changes | High (Subspace absorbs variance) |
| **Memory** | High (if storing all views) | Efficient (Basis vectors only) |
| **Matching** | Cosine Distance | Grassmann / MSM Distance |

---