# Robust AI-Generated Image Detection

TikTok TechJam 2026 — Track 5

## Problem Statement

Develop a model that distinguishes AI-generated images from authentic
images while remaining robust under real-world transformations such as:

- JPEG compression
- Gaussian blur
- Resizing
- Gaussian noise
- Colour jitter
- Cropping

## Current Stage

Research and experimentation.

## Baseline

The baseline detector uses the pretrained
`google/siglip2-base-patch16-256` model as a frozen image encoder.

The architecture is:

```text
Input Image
    ↓
SigLIP2 Image Encoder
    ↓
768-dimensional Image Representation
    ↓
Binary Linear Classifier
    ↓
P(AI)
```

The SigLIP2 backbone is frozen during baseline training, while the binary
classification head is trained to distinguish real and fully AI-generated
images.

### Dataset

The baseline is trained using SID_Set.

Labels used:

- `0` — Real image
- `1` — Fully AI-generated image
- `2` — Tampered/AI-edited image (excluded)

### Installation

```bash
pip install -r requirements.txt
```

### Training

Run:

```bash
python src/train.py
```

The best checkpoint is selected using validation ROC-AUC and saved to:

```text
checkpoints/baseline_best.pt
```

### Inference

Run:

```bash
python src/inference.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/baseline_best.pt
```

Example output:

```text
P(AI): 0.5709
```

The output is a continuous probability between 0 and 1, where a higher
score indicates that the image is more likely to be AI-generated.

## Project Structure

- `src/` — reusable implementation
- `experiments/` — experiment configurations and results
- `notebooks/` — exploratory analysis

## Team

- Ong Yu Jie
- Li Meiyi
- Ng Jia Yi
- Rachel Yao Xin Ru
- Rajaram Sushmiithaa