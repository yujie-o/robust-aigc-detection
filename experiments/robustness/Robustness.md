# Experiments

## Robustness augmentation experiment

This folder contains the experiment-only work for the robustness study.
The shared baseline code in [src/train.py](../../src/train.py) [src/evalutaion/run_evaluation.py] remains the control reference (../../src/evaluation/run_evaluation.py).

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

All experiments were run with 1000 samples per class, 1 epoch, and batch size 4 (using validation dataset).

### Results summary (for validation samples)

| Strategy |    Clean |     JPEG |     Blur |   Resize |    Noise |    Color |     Crop | Avg robust |      Gap |
| -------- | -------: | -------: | -------: | -------: | -------: | -------: | -------: | ---------: | -------: |
| R0       | 0.998475 | 0.995600 | 0.997450 | 0.998150 | 0.989325 | 0.998075 | 0.999225 |   0.996304 | 0.002171 |
| R1       | 0.998575 | 0.996925 | 0.998225 | 0.998425 | 0.992600 | 0.998075 | 0.999200 |   0.997242 | 0.001333 |
| R2       | 0.998450 | 0.996550 | 0.998275 | 0.998550 | 0.992350 | 0.997475 | 0.998950 |   0.997025 | 0.001425 |
| R3       | 0.998425 | 0.998375 | 0.997625 | 0.998375 | 0.993350 | 0.998100 | 0.999075 |   0.997483 | 0.001058 |

### Results summary (for test samples)

| Strategy |  Clean |   JPEG |   Blur | Resize |  Noise |  Color |   Crop | Avg robust | External |
| -------- | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ---------: | -------: |
| R0       | 0.9985 | 0.9967 | 0.9978 | 0.9982 | 0.9915 | 0.9987 | 0.9992 |     0.9970 |   0.9751 |
| R1       | 0.9987 | 0.9971 | 0.9981 | 0.9985 | 0.9940 | 0.9986 | 0.9991 |     0.9976 |   0.9814 |

### Key findings

1. **The clean performance is already extremely high for all strategies**, with all values above 0.9984 AUC. This means the detector is separating real vs AI images very strongly even without heavy augmentation.

2. **R3 has the best average robust AUC** at 0.997483, narrowly outperforming R1 at 0.997242 and R2 at 0.997025. The difference is small, but R3 is the strongest on average over all challenge conditions.

3. **R1 has the best clean AUC** at 0.998575 and the smallest gap between clean and robust performance after R3. This makes R1 a very strong choice if the goal is to maximize both clean and robust performance together.

4. **Noise remains the hardest condition** for all strategies, with the lowest AUCs across the board (R0: 0.989325, R1: 0.992600, R2: 0.992350, R3: 0.993350). This is the main remaining robustness weakness.

5. **The overall spread between strategies is very small**. All models are within about 0.0015 AUC of each other on robust mean performance, indicating that the first epoch is already highly effective on this validation split.

### Recommended strategy

**If the goal is the strongest average robust performance, choose R3.**

**If the goal is the best clean performance with nearly the same robustness, choose R1.**

The practical difference is tiny, and both are stronger than the clean baseline (R0) on robust average, while preserving very strong clean AUC.

## How to Reproduce

### Prerequisites

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. The project requires the SID_Set dataset to be available via Hugging Face Datasets (streaming mode).
3. The project requires the testing datasets to be under `raw_data\wildfake\<dataset_name>`

### Running Individual Strategies

To run a single strategy (e.g., R2):

```bash
python experiments/robustness/robust_augmentation_experiment.py --strategy R2
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
python experiments/robustness/robust_augmentation_experiment.py --all
```

### Output Structure

The experiment generates the following files in `experiments/robustness/`:

```
checkpoints/
├── R0_best.pt
├── R1_best.pt
├── R2_best.pt
└── R3_best.pt


The experiment generates the following files in `results/`:

results/
├── R0_conditions.json
├── R1_conditions.json
├── R2_conditions.json
├── R3_conditions.json
└── summary_table.md
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

- All experiments use a fixed random seed (default: 42) and fixed parameters for reproducibility
- The dataset is loaded via streaming, so the exact images may vary based on Hugging Face dataset updates
- GPU/CPU availability affects training speed but not results
