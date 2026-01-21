# Visual Memory System for Cross-Environment Object Recognition

## Project Overview

Building a **label-free visual memory system** that enables robots (or agents) to discover and recognize objects on-the-fly in novel environments without semantic labels or domain-specific training.

---

## Core Concept

**Traditional VLA Approach:**
- Pre-trained on specific objects/environments
- Requires labels ("chair", "table", etc.)
- Struggles with novel objects or different visual contexts
- Needs fine-tuning for new domains

**Your Approach:**
- Self-supervised visual memory
- No labels - pure embedding-based recognition
- Online learning as the agent explores
- Instant adaptation to new environments

---

## The Pipeline

```
Step 1: Segmentation (SAM/SAM 2)
   Image → Find object boundaries → Binary masks (no labels)
   
Step 2: Embedding Extraction (DINO)
   For each mask:
   - Crop to bounding box
   - Resize to 224×224
   - Extract DINO embedding (1024-d vector)
   
Step 3: Visual Memory
   - Store embeddings with metadata
   - Cluster similar embeddings
   - Recognize objects by similarity matching
   
Step 4: Ad-hoc Analysis
   - "How many object types did I see?"
   - "Have I seen this before?"
   - No labels needed!
```

---

## Key Components

### **1. Segmentation: SAM 2**
- **What it does:** Finds object boundaries in images
- **Output:** Binary masks (no class labels)
- **Why:** Isolates objects from background/context
- **Label-free:** ✅ Completely class-agnostic

### **2. Feature Extraction: DINO v2**
- **What it does:** Converts image regions into semantic embeddings
- **Model:** Vision Transformer (ViT-L/14)
- **Training:** Self-supervised on 142M images (no labels)
- **Output:** 1024-dimensional embedding vector
- **Why:** Captures visual concept without semantic bias

### **3. Visual Memory System**
- **Storage:** Dictionary of embeddings + metadata
- **Clustering:** DBSCAN or similar (discover object types)
- **Recognition:** Cosine similarity / L2 distance
- **Query:** "Is this similar to anything I've seen?"

---

## Example Workflow

### **Apartment Scan (Memory Building)**
```
Frame 1: Chair (angle 1) → Segment → Embed → Store [emb_1]
Frame 5: Chair (angle 2) → Segment → Embed → Similar to [emb_1]
Frame 10: Table → Segment → Embed → New cluster [emb_2]
Frame 20: Lamp → Segment → Embed → New cluster [emb_3]
Frame 30: Chair (angle 3) → Segment → Embed → Clusters with [emb_1]

Result: Discovered 3 object types (chair, table, lamp)
        No labels used!
```

### **Cross-Environment Recognition**
```
Beach/Sim environment:
- See beach chair → Extract embedding
- Compare to apartment memory
- High similarity to apartment chair cluster
- Recognition: "Same type of object!"

Optional: Apply steering to improve alignment
```

---

## What Makes This Novel

1. **Label-free throughout:** No semantic supervision anywhere
2. **Online learning:** Builds memory as it explores
3. **Cross-domain:** Works across different visual contexts
4. **Symbolic anchors:** Objects become anchors without explicit naming
5. **Embedding-as-identity:** Visual similarity IS the recognition signal

---

## Technical Details

### **Models**
- **SAM 2:** Segment Anything Model 2 (Meta, 2024)
  - Checkpoint: `sam2.1_hiera_large.pt` or `sam2.1_hiera_base_plus.pt`
  - VRAM: ~3-4GB
  
- **DINO v2:** Self-supervised Vision Transformer (Meta, 2023)
  - Model: `dinov2_vitl14` (300M params)
  - VRAM: ~2-3GB
  - Output: 1024-d embeddings

### **Clustering**
- **Method:** DBSCAN (density-based)
- **Metric:** Cosine similarity
- **Parameters:** `eps=0.3`, `min_samples=2`

### **Hardware Requirements**
- **GPU:** 8GB VRAM minimum (you have this ✅)
- **Works with:** T4, L4, RTX 3070, etc.
- **Local execution:** Run on your machine, no cloud needed

---

## Implementation Status

### **Completed**
- ✅ Core pipeline design
- ✅ DINO embedding extractor module
- ✅ Visual memory system module
- ✅ Requirements.txt

### **In Progress**
- 🔄 SAM 2 integration (Windows compatibility)
- 🔄 Segmentation + embedding pipeline
- 🔄 Data collection (apartment photos)

### **To Do**
- ⏳ Full pipeline integration
- ⏳ Clustering and visualization
- ⏳ Cross-environment experiments
- ⏳ Optional: Steering method integration
- ⏳ Paper writing (Week of Jan 27-Feb 1)

---

## NeuS 2026 Paper Plan

### **Title Ideas**
- "Self-Supervised Visual Memory for Cross-Domain Object Recognition"
- "Label-Free Visual Memory: Enabling VLAs to Discover Objects Online"
- "Embedding-Based Object Discovery for Lifelong Robot Learning"

### **Key Contributions**
1. **Visual memory system** that builds object representations online
2. **Label-free approach** using only embedding similarity
3. **Cross-environment validation** (apartment → sim/different location)
4. **Real-world demonstration** with actual photos

### **Experimental Results to Show**
1. **Object discovery:** Clustering quality, discovered object types
2. **Instance recognition:** Multiple views of same object cluster together
3. **Cross-environment:** Objects recognized across different contexts
4. **Ablations:** Whole-image vs segmented embeddings
5. **Optional:** With/without steering improvements

### **Timeline (10 Days Left)**
- **Weekend (Jan 25-26):** Take apartment photos, get pipeline working
- **Week 1 (Jan 27-30):** Run experiments, generate results
- **Week 2 (Jan 31-Feb 1):** Write paper, submit

---

## Why This Matters

### **For Robotics**
- Robots can explore novel environments without pre-training
- No need for labeled data in new domains
- Instant object discovery and recognition

### **For VLAs**
- Current VLAs struggle with distribution shift
- Your approach: adapt on-the-fly through visual memory
- Foundation for lifelong learning systems

### **For Your Research**
- Connects to symbolic anchors (DARPA TIAMAT)
- Foundation for steering methods (future work)
- Practical system with real-world validation
- Publishable workshop paper (NeuS 2026)

---

## Connection to Broader Research Goals

### **Symbolic Anchors**
Objects in memory serve as symbolic anchors:
- Apartment chair ↔ Beach chair = same anchor
- No explicit naming needed
- Alignment through embedding space

### **Steering (Future Work)**
Once objects are discovered:
- Apply steering to align embeddings
- Improve cross-environment recognition
- Bridge visual domain gaps

### **DARPA TIAMAT**
Addresses project goals:
- Novel environment adaptation
- Symbolic reasoning with visual data
- Robot navigation with uncertain maps

---

## Key Insights

1. **DINO is not a segmentation model** - it's a feature extractor
2. **SAM/SAM 2 are label-free** - they find boundaries without classes
3. **Embeddings > Labels** - similarity matching is sufficient for recognition
4. **Real data > Sim data** - apartment photos more impressive than Isaac Sim
5. **Simplicity wins** - prototype with whole-image first, add segmentation if time

---

## Files Structure

```
dino-visual-memory/
├── requirements.txt          # Dependencies
├── embeddings.py             # DINO embedding extractor
├── memory.py                 # Visual memory system
├── segmentation.py           # SAM integration
├── segment_and_embed.py      # Full pipeline
├── main.py                   # Main execution script
├── data/
│   ├── apartment/           # Your photos here
│   └── other_location/      # Optional cross-env data
├── results/
│   ├── clusters.png         # t-SNE visualization
│   └── apartment_memory.pkl # Saved memory state
└── checkpoints/
    └── sam2.1_hiera_large.pt # SAM 2 weights
```

---

## Next Session Action Items

1. **Resolve SAM 2 installation** (Windows or fallback to SAM 1)
2. **Take apartment photos** (~50 images, different rooms/angles)
3. **Test full pipeline** on a few images
4. **Generate first clustering visualization**
5. **Iterate and refine**

---

## References

- **DINO v2:** [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **SAM 2:** [Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
- **Original SAM:** [Segment Anything](https://arxiv.org/abs/2304.02643)

---

**Bottom Line:** You're building a system that lets agents "see" and "remember" objects purely through visual similarity, with no labels required. It's practical, novel, and addresses real robotics challenges. 🚀