# P2: Generalisation Experiment

## Objective
Test whether UnivFD's Nearest Neighbor classification method improves external generalisation over the baseline, which implements UnivFD's Linear Classification method.

## Background
UnivFD [Ojha et al.] proposes two lightweight classification methods on top of a frozen general-purpose backbone (originally CLIP ViT-L/14):
1. **Linear Classification (LC)** — single linear layer trained with BCE loss on frozen features
2. **Nearest Neighbor (NN)** — non-parametric classification via cosine similarity to a training-set feature bank

The team's baseline (P1) implements Method 1 with SigLIP2 substituted for CLIP as the frozen backbone. P2 tests whether Method 2 (NN) improves generalisation over Method 1 (LC), holding the backbone and training data constant.

## Method

**Setup**
- Backbone: frozen `google/siglip2-base-patch16-256`
- Training data: SID_Set balanced subset (1000 real + 1000 AI)
- Train/val split: 80/20, seed 42
- Baseline and P2-a share the backbone, training data, and split. Only the classification head differs.

**P2-a — Nearest Neighbor Classifier**
1. Forward all 1600 training images through frozen SigLIP2 → extract L2-normalized features (768-dim)
2. Store as feature bank: (features, labels)
3. At inference: extract feature for query image, compute cosine similarity to bank
4. Take top-k=5 neighbors; `P(AI)` = fraction of AI-labeled neighbors
5. No learned parameters

## Evaluation

**Internal (SID val split, 400 images):**
- Clean AUC
- Robustness AUCs across 6 transformation groups: JPEG (4 severities), Blur (3), Resize (2), Noise (3), Color jitter, Center crop

**External (organiser-provided WildFake subset, 400 images):**
- Real: COCO val2017
- AI: DALL·E Advanced (dalle3 with IsAdvanced=1 flag)

Both models evaluated through the same shared evaluator to ensure comparability.

## Results

| Model | Clean | JPEG | Blur | Resize | Noise | Color | Crop | Robust Mean | External |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (LC) | 0.9985 | 0.9967 | 0.9978 | 0.9982 | 0.9909 | 0.9976 | 0.9992 | 0.9967 | 0.9689 |
| P2a-NN (k=5) | 0.9818 | 0.9784 | 0.9775 | 0.9810 | 0.9655 | 0.9747 | 0.9807 | 0.9763 | 0.8560 |
| Delta | -0.0167 | -0.0183 | -0.0203 | -0.0172 | -0.0254 | -0.0229 | -0.0185 | -0.0204 | **-0.1129** |

**Key finding:** P2-a underperforms baseline across every condition. The largest gap is on the external eval (-0.113 AUC), the metric most relevant to the P2 generalisation question.

