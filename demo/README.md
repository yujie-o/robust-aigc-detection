# AIGC Detection Demo

An interactive Gradio app for demonstrating our robust AI-generated image detector end-to-end.

## Objective

Provide a live, visual demonstration of the project's shipped solution: a binary classifier that distinguishes AI-generated images from authentic photographs, and  remains reliable after common post-processing transforms like JPEG compression, blur, resizing, and noise.

This app is the centerpiece of the required demo video. Upload an image, see a prediction, apply a transform, watch the prediction hold.

## What It Does

The app wraps our trained detector in a simple web UI with three panels:

1. **Input image** — drag-and-drop or click to upload any image (JPEG, PNG).
2. **Transform dropdown** — optionally apply one of the six post-processing transform families used during training and evaluation:
   - JPEG compression (quality 50 or 30)
   - Gaussian blur (sigma 1.0 or 2.0)
   - Resize down-then-up (0.5x or 0.25x)
   - Gaussian noise (sigma 0.05)
   - Color jitter
   - Center crop (80%)
3. **Prediction output** — shows the transformed image, a verdict label ( Authentic or AI-generated), the raw AI probability (0–1), and a confidence bar.

Under the hood, the app loads a trained `AIGCDetector` checkpoint (frozen SigLIP2-Base backbone + linear probe head), passes the input image through the same preprocessing pipeline used during training, and reports the sigmoid probability that the image is AI-generated.

## Architecture

```
Input image
    ↓
HuggingFace AutoProcessor (resize, normalize)
    ↓
SigLIP2-Base-Patch16-256 (frozen backbone) → pooler_output (768-dim features)
    ↓
Linear(768 → 1) → sigmoid
    ↓
AI probability [0, 1]
```

Backbone: `google/siglip2-base-patch16-256` (frozen — never fine-tuned).
Classifier head: single `nn.Linear` layer trained on SID_Set with augmentation strategy R0–R3.

## Setup

From the repo root:

```bash
pip install gradio
```

All other dependencies (torch, transformers, PIL, numpy) are already in the project's `requirements.txt`.

## Running the App

Default (loads `experiments/robustness/checkpoints/R2_best.pt`):

```bash
python demo/app.py
```

Point at a specific checkpoint (e.g. the winning R2 strategy):

```bash
python demo/app.py --checkpoint experiments/robustness/checkpoints/R2_best.pt
```

Other flags:

```bash
python demo/app.py --port 7861          # use a different local port
python demo/app.py --share              # create a public gradio.live link (useful for remote demos)
```

Once launched, open the URL printed in the terminal — typically `http://127.0.0.1:7860`.

**First launch note:** The SigLIP2 backbone weights (~400 MB) are downloaded from HuggingFace on first run. Subsequent launches use the cached copy and start in a few seconds.


## Troubleshooting

**`ModuleNotFoundError: No module named 'gradio'`**
Run `pip install gradio` in your active environment.

**`FileNotFoundError: Checkpoint not found`**
Train a strategy first, or point `--checkpoint` at an existing `.pt` file. Check `dir experiments\robustness\checkpoints` to see what's available.

**Slow first launch**
Normal — HuggingFace is downloading the backbone weights. Wait for the Gradio URL to appear.

**CUDA out of memory**
The app auto-detects CPU vs GPU. To force CPU, edit `app.py` and change `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")` to `DEVICE = torch.device("cpu")`. Inference is slower but works fine for a demo.

## File Structure

```
demo/
├── app.py           # The Gradio app
├── README.md        # This file
└── samples/         # Images for testing
```