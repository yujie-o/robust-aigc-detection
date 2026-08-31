# Robust AI-Generated Image Detection

TikTok TechJam 2026 — Track 5

## Project Overview

Our project develops a machine learning model that distinguishes **AI-generated images from authentic images** while remaining robust under common real-world image transformations.

The detector is evaluated under transformations including:

- JPEG compression
- Gaussian blur
- Resizing
- Gaussian noise
- Colour jitter
- Cropping

Our approach begins with a common **SigLIP2-based baseline**, which serves as the foundation for subsequent generalisation, robustness, and hybrid detection experiments.

### Baseline

The baseline detector uses the pretrained `google/siglip2-base-patch16-256` model as a frozen image encoder.

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

The SigLIP2 backbone is frozen during baseline training, while the binary classification head is trained to distinguish real and fully AI-generated images.

### Dataset

The baseline is trained using **SID_Set**.

Labels used:

- `0` — Real image
- `1` — Fully AI-generated image
- `2` — Tampered/AI-edited image (excluded)

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yujie-o/robust-aigc-detection.git
cd robust-aigc-detection
```

### 2. Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

To run the interactive demo, also install Gradio:

```bash
pip install gradio
```

### 3. Dataset Setup

The training dataset is not included in the repository.

<!-- TODO: Add the final SID_Set folder structure/instructions here. -->

For external evaluation using the WildFake subset, organise the dataset as:

```text
raw_data/
└── wildfake/
    ├── val2017/
    └── dalle/
```

where:

- `val2017/` contains real images from COCO val2017
- `dalle/` contains AI-generated images from DALL-E Advanced

The external evaluation dataset is kept separate from the training data.

---

## Steps to Reproduce Our Results

### 1. Train the Baseline

Run:

```bash
python src/train.py
```

The best checkpoint is selected using validation ROC-AUC and saved to:

```text
checkpoints/baseline_best.pt
```

### 2. Train the model with R2 strategy

```bash
python experiments/robustness/robust_augmentation_experiment.py --strategy R2
```

This will train the baseline model with R2 augmentation (JPEG compression → Gaussian blur → resize round-trip, applied to every training image) and perform the evaluation. Results from R2 strategy will be saved in `summary_table.md`  and `R2_conditions.json` under `results/` folder. The model checkpoint will be saved in `experiments/robustness/checkpoints/R2_best.pt`

---

## Running Evaluation and Inference

### Evaluation

<!-- TODO: Add the final evaluation command and instructions here. -->

The evaluation pipeline is used to compare the baseline and experimental models under clean and transformed image conditions.

The evaluated transformations include:

- JPEG compression
- Gaussian blur
- Resizing
- Gaussian noise
- Colour jitter
- Cropping

External evaluation is also performed using the WildFake subset to assess performance on data that was not used during training.

### Inference

To run inference on an individual image:

```bash
python src/inference.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/baseline_best.pt
```

Example output:

```text
P(AI): 0.5709
```

The output is a continuous probability between `0` and `1`, where a higher score indicates that the image is more likely to be AI-generated.

<!-- TODO: Add/update the inference command for the final selected model if required. -->

---

## Interactive Demo

We provide an interactive Gradio web app that demonstrates the detector end-to-end. Upload an image, optionally apply a post-processing transform, and see the model's prediction in real time. This is the app used in our demo video.

### What It Does

The app wraps our trained detector in a simple web UI with three panels:

1. **Input image** — drag-and-drop or click to upload any image (JPEG, PNG).
2. **Transform dropdown** — optionally apply one of the six post-processing transform families used during training and evaluation:
   - JPEG compression (quality 50 or 30)
   - Gaussian blur (sigma 1.0 or 2.0)
   - Resize down-then-up (0.5x or 0.25x)
   - Gaussian noise (sigma 0.05)
   - Colour jitter
   - Centre crop (80%)
3. **Prediction output** — shows the transformed image, a verdict label (Authentic or AI-generated), the raw AI probability (0–1), and a confidence bar.

Under the hood, the app loads a trained `AIGCDetector` checkpoint (the same frozen SigLIP2 backbone + linear probe head described in the [Baseline](#baseline) section) and reports the sigmoid probability that the image is AI-generated.

### Running the App

Default (loads `experiments/robustness/checkpoints/R3_best.pt`):

```bash
python demo/app.py
```

Point at a specific checkpoint:

```bash
python demo/app.py --checkpoint experiments/robustness/checkpoints/R3_best.pt
```

Other flags:

```bash
python demo/app.py --port 7861          # use a different local port
python demo/app.py --share              # create a public gradio.live link
```

Once launched, open the URL printed in the terminal, typically `http://127.0.0.1:7860`.

**First launch note:** The SigLIP2 backbone weights (~400 MB) are downloaded from HuggingFace on first run. Subsequent launches use the cached copy and start in a few seconds.

### Troubleshooting

**`ModuleNotFoundError: No module named 'gradio'`**
Run `pip install gradio` in your active environment.

**`FileNotFoundError: Checkpoint not found`**
Train a strategy first, or point `--checkpoint` at an existing `.pt` file. Run `dir experiments\robustness\checkpoints` to see what's available.

**Slow first launch**
Normal — HuggingFace is downloading the backbone weights. Wait for the Gradio URL to appear.

**CUDA out of memory**
The app auto-detects CPU vs GPU. To force CPU, edit `demo/app.py` and change `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")` to `DEVICE = torch.device("cpu")`. Inference is slower but works fine for a demo.

---

## Project Structure

```text
robust-aigc-detection/
├── src/              # Reusable implementation
├── experiments/      # Experiment configurations and results
├── demo/             # Interactive Gradio app
├── checkpoints/      # Locally saved model checkpoints
└── requirements.txt  # Python dependencies
```

---

## Limitations and Future Improvements

<!--
TODO: Add the final reflection here after all experiments are completed.

Include:
- Limitations of the final solution
- Conditions where the detector performs poorly
- Computational or inference limitations
- What could be improved given more time
-->

---

## Team Member Contributions

- **Ong Yu Jie** — Baseline model and training infrastructure
- **Ng Jia Yi** — Generalisation experiments
- **Rajaram Sushmiithaa** — Robustness experiments
- **Rachel Yao Xin Ru** — Hybrid detection experiments
- **Li Meiyi** — Evaluation and error analysis
