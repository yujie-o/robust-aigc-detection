# P2: Generalisation Experiment

## Objective
Test whether UnivFD's Nearest Neighbor classification method improves generalisation over the baseline (UnivFD Linear Classification) when using identical frozen SigLIP2 features.

## Motivation
The baseline (P1) already implements UnivFD's Linear Classification recipe: frozen SigLIP2 backbone + single linear layer + BCE loss. P2 tests UnivFD's second classification method — Nearest Neighbor in feature space — to determine whether the non-parametric approach generalises differently than a trained linear probe.

## Setup
- **Backbone**: frozen `google/siglip2-base-patch16-256` (same as baseline)
- **Training data**: SID_Set balanced subset — 1000 real + 1000 AI
- **Split**: 80/20 train/val, seed 42
- Same across baseline and P2-a; only the classification head differs

## Method (P2-a)
1. Extract L2-normalized SigLIP2 features for all 1600 training images
2. Build feature bank of (features, labels)
3. For each val image: compute cosine similarity to bank
4. Take top-k=5 neighbors; P(AI) = fraction of AI-labeled neighbors
5. No learned parameters — non-parametric classification

## Results

| Model | Head | Learned params | Internal Val AUC | External AUC (WildFake) |
|---|---|---|---|---|
| Baseline | Linear probe (trained) | ~768 | 0.9985 | TBD |
| P2-a | k-NN classifier (k=5) | 0 | 0.9818 | TBD |

## Preliminary Observation
On internal val, baseline is 0.0167 higher than P2-a. This is expected: the linear probe is optimised for the exact training distribution while the NN classifier uses raw features with zero learned parameters. The critical comparison is external AUC — UnivFD's central claim is that linear probes overfit to generator-specific artifacts, and NN classification should generalise better under distribution shift.

## Todo
- [ ] Locate + download organiser eval subset (WildFake: 4998 COCO + 8843 DALL·E)
- [ ] Run baseline through external eval → external AUC
- [ ] Run P2-a through external eval → external AUC
- [ ] k-ablation: try k=1, 3, 10, 20
- [ ] Decide on P2-b (data breadth experiment) based on external results

## Limitations
- Single external eval on DALL·E only — generalisation claim limited to one distribution shift
- k=5 chosen without ablation
- SID_Set training subset is small (2000 images total)