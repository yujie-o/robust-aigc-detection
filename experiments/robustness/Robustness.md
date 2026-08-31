# Experiments

## Robustness augmentation experiment

This folder contains the experiment-only work for the robustness study.
The shared baseline code — [`src/train.py`](../../src/train.py) and
[`src/evaluation/run_evaluation.py`](../../src/evaluation/run_evaluation.py) —
remains the control reference.

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

For R1 and R3, the specific transform(s) picked for each training image are
randomised at training time and are not implied by the strategy name alone.

## How the robustness evaluation works

Evaluation calls
`run_full_combined_evaluation()` from `src/evaluation/run_evaluation.py`
directly. For each trained strategy:

1. Train the model on the selected strategy-specific dataset (`R0`–`R3`), reusing `train.py`'s `SAMPLES_PER_CLASS`, `VAL_RATIO`, and `SEED` by default so the split matches the baseline model.
2. Save the best-val-AUC checkpoint.
3. Reload those weights and run the shared combined evaluator, which:
   - Re-derives the same SID_Set validation split (same seed) and evaluates it under all 14 fixed perturbation conditions (clean, 4× JPEG quality levels, 3× blur, 2× resize, 3× noise, color jitter, crop).
   - Runs a clean-only pass over a local WildFake subset for external generalisation.
   - Computes ROC-AUC per condition, per-group averages, degradation vs. clean, the JPEG-quality trend, and the internal-vs-external AUC gap.
4. Append one row to `results/summary_table.md` and write `results/{strategy}_conditions.json` with full per-image predictions.

This answers the key question: does the model still separate real vs AI images after JPEG, blur, resize, noise, color shifts, or cropping — and does it generalise beyond the dataset it was trained on?

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
- external (WildFake) AUC
- clean-vs-robustness gap

## Experimental results

All experiments were run with 1000 samples per class, 1 epoch, batch size 4,
and the default `val_ratio=0.2`/`seed=42` inherited from `train.py`. Internal
scores are computed on the held-out SID_Set validation split (400 images);
external scores are computed on a local WildFake subset (COCO `val2017` as
real, the `dalle` generator subfolder as fake), sampled to the same
`samples-per-class` used for training.

### Results summary

| Model | Clean  | JPEG   | Blur   | Resize | Noise  | Color  | Crop   | Robust Mean | External |
| ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ----------- | -------- |
| R0    | 0.9985 | 0.9967 | 0.9978 | 0.9982 | 0.9915 | 0.9987 | 0.9992 | 0.9970      | 0.9751   |
| R1    | 0.9987 | 0.9971 | 0.9981 | 0.9985 | 0.9940 | 0.9986 | 0.9991 | 0.9976      | 0.9814   |
| R2    | 0.9985 | 0.9971 | 0.9981 | 0.9986 | 0.9938 | 0.9986 | 0.9989 | 0.9975      | 0.9871   |
| R3    | 0.9984 | 0.9970 | 0.9980 | 0.9983 | 0.9939 | 0.9981 | 0.9990 | 0.9974      | 0.9827   |

### Key findings

1. **Clean performance is already extremely high for all strategies**, with all values above 0.9984 AUC. This means the detector is separating real vs AI images very strongly even without heavy augmentation.

2. **R1 has the best clean AUC** at 0.9987, and the best robust mean at 0.9976 which is the strongest combination of "don't hurt clean accuracy" and "improve robustness" of the four strategies.

3. **R2 has the best external (WildFake) generalisation** at 0.9871, well ahead of R0 (0.9751), R1 (0.9814), and R3 (0.9827). Its fixed JPEG→blur→resize composite most closely resembles the kind of degradation real-world images (re-encoded, resized, mildly compressed) actually go through, which likely explains the generalisation gain.

4. **Noise remains the hardest internal condition** for every strategy (lowest column-wise AUC across the board), but augmentation training closes much of that gap: R0 sits at 0.9915 while R1/R2/R3 are all ≥0.9938.

5. **The spread between strategies is small on internal metrics** (≤0.0006 AUC on robust mean) but noticeably larger on external generalisation (0.9751–0.9871, a 0.012 spread) — the external benchmark is the more discriminating signal between strategies here, since all four already saturate the internal SID_Set conditions.

### Recommended strategy

- **If external generalisation to real-world / out-of-distribution AI images is the priority, choose R2.** It has the largest external AUC by a clear margin, at a negligible cost to clean/internal-robust performance.
- **If the priority is maximizing performance on data that looks like SID_Set (clean + the fixed corruption battery), choose R1.** It has the best clean AUC and best internal robust mean, with the smallest clean-vs-robust gap among the augmented strategies.
- **R0 (no augmentation) is not recommended** — it's dominated by every augmented strategy on both internal robust mean and external generalisation, at no clean-AUC advantage worth keeping.

Given how close R1 and R2 are internally, and that external generalisation is the more differentiating and arguably more decision-relevant metric for real-world deployment, **R2 is the overall recommendation** unless the target distribution is known to closely match SID_Set, in which case R1 is preferable.

## How to Reproduce

### Prerequisites

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. **SID_Set** is fetched automatically — it streams from the Hugging Face Hub (`saberzl/SID_Set`) the first time it's needed, no manual download required. It's cached under HF's own cache directory, not under this project's `raw_data/`.
3. **WildFake is not auto-downloaded.** You must manually populate:
   ```
   raw_data/wildfake/val2017/   # real images (COCO val2017)
   raw_data/wildfake/dalle/     # fake images (WildFake's "dalle" generator subset)
   ```
   `robust_augmentation_experiment.py` checks for both folders **before training starts** and fails fast with a clear error if either is missing — this avoids wasting a full training run only to fail during the external-evaluation step at the end.

### Running Individual Strategies

To run a single strategy (e.g., R2):

```bash
python experiments/robustness/robust_augmentation_experiment.py --strategy R2
```

**Parameters:**

- `--strategy`: Choose from `R0`, `R1`, `R2`, `R3`
- `--epochs`: Training epochs (default: `train.py`'s `EPOCHS`, currently 1)
- `--batch-size`: DataLoader batch size (default: `train.py`'s `BATCH_SIZE`, currently 4)
- `--samples-per-class`: Balanced samples per class (default: `train.py`'s `SAMPLES_PER_CLASS`, currently 1000)
- `--val-ratio`: Validation split ratio (default: `train.py`'s `VAL_RATIO`, currently 0.2)
- `--seed`: Random seed for reproducibility (default: `train.py`'s `SEED`, currently 42)

`--samples-per-class`, `--val-ratio`, and `--seed` default to `train.py`'s
constants so every strategy's validation split matches the baseline model's
out of the box. They can be overridden, but doing so means that run's
results won't be directly comparable to the baseline row — the script
prints a warning if any of the three differ from `train.py`'s values.

### Running All Strategies

To run all four strategies sequentially:

```bash
python experiments/robustness/robust_augmentation_experiment.py --all
```

### Output Structure

The experiment generates the following files in `experiments/robustness/checkpoints/`:

```
checkpoints/
├── R0_best.pt
├── R1_best.pt
├── R2_best.pt
├── R3_best.pt
```

And the following in `results/` (shared with `run_evaluation.py` — rows are
appended, so re-running a strategy adds a new row rather than replacing the
old one):

```
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
  "strategy": "R0",               # Strategy used
  "val_auc": 0.95,                # Best validation AUC (during training, on clean val data)
  "epoch": 3                      # Epoch where best AUC was achieved
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

- All experiments use a fixed random seed (default: 42, sourced from `train.py`) and fixed parameters for reproducibility.
- The SID_Set split is deterministic given the seed, but is loaded via streaming, so the exact images may vary based on Hugging Face dataset updates upstream.
- The random transform(s) picked per training image for R1/R3 are also deterministic given the seed, **as long as the training `DataLoader` uses `num_workers=0`** (the default here). Adding multiprocess data loading workers without a `worker_init_fn` would break this determinism.
- GPU/CPU availability affects training speed but not results.
- WildFake results depend on whatever is currently in `raw_data/wildfake/{val2017,dalle}/` — since that data is manually placed and not versioned by this pipeline, external AUCs are only comparable across runs performed against the same local WildFake snapshot.
