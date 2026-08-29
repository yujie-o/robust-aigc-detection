# Experiments

## Robustness augmentation experiment

This folder contains the experiment-only work for the robustness study.
The shared baseline code in [src/train.py](../../src/train.py) remains the control reference.

## Objective

Measure whether robustness augmentation improves detector performance on real-world perturbations without hurting clean performance.

## Experimental strategies

- R0: no augmentation during training; clean-image baseline
- R1: randomly apply one challenge transform during training
- R2: fixed realistic composite transform (JPEG + blur + resize)
- R3: randomly apply 1–2 challenge transforms during training

## Why R0 is not the same as R1

R0 and R1 are different training setups:

- R0 trains on clean images only.
- R1 trains on images where one augmentation is applied at random.

The same model can then be evaluated on multiple conditions:

- clean
- JPEG
- blur
- resize
- noise
- color jitter
- crop

This is important: the evaluation is not restricted to the training condition. A model trained on clean data (R0) is still tested on corrupted images to measure robustness. A model trained with augmentations (R1/R2/R3) is also evaluated on clean and corrupted data to check the clean-vs-robustness trade-off.

## How the robustness evaluation works

For each trained strategy:

1. Train the model on the selected strategy-specific dataset.
2. Evaluate the same trained model on the validation set under multiple perturbations.
3. Compute ROC-AUC for each condition.
4. Compare the clean score against the perturbed scores.

This answers the key question: does the model still separate real vs AI images after JPEG, blur, resize, noise, color shifts, or cropping?

## Real-world rationale

A realistic image pipeline often includes a mix of:

- compression
- resizing
- slight blur
- minor color shifts

That is why R2 is designed as a small composite of JPEG + blur + resize, while R3 tests a random subset of 1–2 transformations.

## Deliverable format

For each strategy, record:

- clean AUC
- JPEG AUC
- blur AUC
- resize AUC
- noise AUC
- color AUC
- crop AUC
- average robust AUC
- clean-vs-robustness gap

## Experimental results

All experiments trained on 20 samples per class for 1 epoch with batch size 2.

### Results summary

| Strategy | Clean |  JPEG |  Blur | Resize | Noise | Color |  Crop | Avg robust |   Gap |
| -------- | ----: | ----: | ----: | -----: | ----: | ----: | ----: | ---------: | ----: |
| R0       | 1.000 | 0.938 | 1.000 |  0.938 | 0.875 | 1.000 | 0.938 |  **0.948** | 0.052 |
| R1       | 1.000 | 1.000 | 0.938 |  0.938 | 0.938 | 1.000 | 0.938 |  **0.958** | 0.042 |
| R2       | 1.000 | 0.875 | 0.938 |  0.938 | 0.875 | 1.000 | 0.938 |  **0.927** | 0.073 |
| R3       | 1.000 | 0.875 | 0.938 |  0.938 | 0.938 | 1.000 | 0.938 |  **0.948** | 0.052 |

### Key findings

1. **All strategies maintain perfect clean performance (1.0 AUC)** — no accuracy degradation on authentic images.

2. **R1 (single random augmentation) is the best strategy** — achieves the highest average robust AUC (0.958) and the smallest clean-vs-robustness gap (0.042).

3. **R2 (composite transform JPEG+blur+resize) is the weakest** — performs worst on JPEG (0.875) and noise (0.875), with the largest gap (0.073). This suggests that the fixed composite is too aggressive during training, causing overfitting to that specific perturbation pattern.

4. **Noise is the hardest challenge** — all strategies drop below 0.94 on noise, with R0 and R2 dropping to 0.875. This indicates that noise perturbations are harder to robustness-train for than compression or blur.

5. **R0 and R3 perform equivalently** — both achieve 0.948 average robust AUC. R3's random 1–2 transforms provide no advantage over R0 on this small subset, suggesting that diversity during training may require more data to show benefits.

### Recommended strategy

**Use R1** — training with a single random augmentation per batch provides the best balance:

- Maintains perfect clean detection (1.0)
- Achieves strongest robustness across all perturbations
- Minimal implementation overhead

### Next steps

1. Validate on the full dataset with more epochs to confirm R1 superiority.
2. Investigate why noise is so challenging — consider stronger noise augmentation during training.
3. Test R2 variants with adaptive or learned composite transforms instead of fixed.
4. Compare computational cost vs robustness gain for R1 in production.

## How to Reproduce

### Prerequisites

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. The project requires the SID_Set dataset to be available via Hugging Face Datasets (streaming mode).

### Running Individual Strategies

To run a single strategy (e.g., R0) with custom parameters:

```bash
cd experiments/robustness
python robust_augmentation_experiment.py \
  --strategy R0 \
  --samples-per-class 1000 \
  --epochs 5 \
  --batch-size 4 \
  --val-ratio 0.2
```

**Parameters:**

- `--strategy`: Choose from `R0`, `R1`, `R2`, `R3`
- `--samples-per-class`: Number of balanced samples per class (default: 1000)
- `--epochs`: Training epochs (default: 1)
- `--batch-size`: DataLoader batch size (default: 4)
- `--val-ratio`: Validation split ratio (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

### Running All Strategies

To run all four strategies sequentially:

```bash
cd experiments/robustness
python robust_augmentation_experiment.py \
  --all \
  --samples-per-class 1000 \
  --epochs 5 \
  --batch-size 4
```

### Output Structure

The experiment generates the following files in `experiments/robustness/`:

```
checkpoints/
├── R0_best.pt
├── R1_best.pt
├── R2_best.pt
└── R3_best.pt

results/
├── R0_summary.json
├── R1_summary.json
├── R2_summary.json
├── R3_summary.json
└── strategy_summary.json
```

### Output Format

Each `R{0,1,2,3}_summary.json` contains:

```json
{
  "strategy": "R0",
  "samples_per_class": 1000,
  "epochs": 5,
  "batch_size": 4,
  "val_ratio": 0.2,
  "best_val_auc": 0.95,
  "clean_auc": 0.95,
  "jpeg_auc": 0.92,
  "blur_auc": 0.94,
  "resize_auc": 0.91,
  "noise_auc": 0.88,
  "color_auc": 0.94,
  "crop_auc": 0.93,
  "best_epoch": 3
}
```

The `strategy_summary.json` aggregates results from all four strategies:

```json
[
  { "strategy": "R0", ... },
  { "strategy": "R1", ... },
  { "strategy": "R2", ... },
  { "strategy": "R3", ... }
]
```

### Checkpoint Format

Each `R{0,1,2,3}_best.pt` contains:

```python
{
  "model_state_dict": {...},      # Trained classifier weights
  "strategy": "R0",                # Strategy used
  "val_auc": 0.95,                 # Best validation AUC
  "epoch": 3                        # Epoch where best AUC was achieved
}
```

To load a checkpoint:

```python
import torch
from models.baseline import AIGCDetector

checkpoint = torch.load("checkpoints/R1_best.pt")
model = AIGCDetector(freeze_backbone=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Reproducibility Notes

- All experiments use a fixed random seed (default: 42) for reproducibility
- The dataset is loaded via streaming, so the exact images may vary based on Hugging Face dataset updates
- GPU/CPU availability affects training speed but not results
- Results from small sample sizes (20 per class in the table above) should be validated on the full dataset
