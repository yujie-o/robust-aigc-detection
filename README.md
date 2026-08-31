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

### 2. Run Evaluation

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

### 3. Run Inference

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

## Project Structure

```text
robust-aigc-detection/
├── src/              # Reusable implementation
├── experiments/      # Experiment configurations and results
├── notebooks/        # Exploratory analysis
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
