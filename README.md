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

Our final selected model uses the **R2 robustness augmentation strategy**, where JPEG compression, Gaussian blur, and resize round-trip are applied sequentially to every training image.

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

The SigLIP2 backbone is frozen during training, while the binary classification head is trained to distinguish real and fully AI-generated images.

### Dataset

The model is trained using **SID_Set** (`saberzl/SID_Set`).

Labels used:

- `0` — Real image
- `1` — Fully AI-generated image
- `2` — Tampered/AI-edited image (excluded)

For external evaluation, we use a subset of **WildFake** consisting of:

- Real images from COCO val2017
- AI-generated images from DALL-E Advanced

The external evaluation dataset is kept completely separate from the training data.

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

No manual setup is required for the training dataset.

**SID_Set** is streamed directly from Hugging Face using the `datasets` library:

```text
saberzl/SID_Set
```

The training pipeline automatically samples a balanced subset of real (`label 0`) and fully AI-generated (`label 1`) images. Tampered/AI-edited images (`label 2`) are excluded.

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

The WildFake subset is used only for external evaluation and is never used during training.

---

## Steps to Reproduce Our Results

### 1. Train the Baseline

Run:

```bash
python src/train.py
```

The training pipeline streams a balanced subset of SID_Set and trains the binary classification head while keeping the SigLIP2 backbone frozen.

The best checkpoint is selected using validation ROC-AUC and saved to:

```text
checkpoints/baseline_best.pt
```

### 2. Train and Evaluate the R2 Model

Run:

```bash
python experiments/robustness/robust_augmentation_experiment.py --strategy R2
```

The R2 strategy applies the following transformations sequentially to every training image:

```text
JPEG Compression
    ↓
Gaussian Blur
    ↓
Resize Round-Trip
```

After training, the model is evaluated using **ROC-AUC** on clean images, six post-processing transformation families (JPEG compression, Gaussian blur, resizing, Gaussian noise, colour jitter, and cropping), and the external WildFake subset.

The trained model checkpoint is saved to:

```text
experiments/robustness/checkpoints/R2_best.pt
```

Evaluation results are saved under:

```text
results/
├── summary_table.md
└── R2_conditions.json
```

`summary_table.md` contains the overall evaluation results, including the average robustness score and external WildFake performance, while `R2_conditions.json` contains the detailed results for each evaluation condition.

---

## Interactive Demo

We provide an interactive Gradio web app that demonstrates the detector end-to-end. Users can upload an image, optionally apply a post-processing transform, and view the model's prediction in real time.

The demo uses our final **R2 model**.

### What It Does

The app provides three main components:

1. **Input image** — drag and drop or select an image to upload.
2. **Transform selection** — optionally apply one of the post-processing transformations used during evaluation:
   - JPEG compression
   - Gaussian blur
   - Resize down-then-up
   - Gaussian noise
   - Colour jitter
   - Centre crop
3. **Prediction output** — displays the transformed image, predicted class, and AI probability.

The app loads the trained `AIGCDetector` using the same frozen SigLIP2 backbone and linear classification head described above.

The model outputs:

```text
P(AI) ∈ [0, 1]
```

where a higher probability indicates that the image is more likely to be AI-generated.

### Running the App

By default, the app loads the final R2 model:

```bash
python demo/app.py
```

To specify the R2 checkpoint manually:

```bash
python demo/app.py --checkpoint experiments/robustness/checkpoints/R2_best.pt
```

Other available options include:

```bash
python demo/app.py --port 7861
python demo/app.py --share
```

Once launched, open the URL printed in the terminal, typically:

```text
http://127.0.0.1:7860
```

**First launch note:** The SigLIP2 backbone weights are downloaded from Hugging Face on the first run. Subsequent launches use the cached model.

### Troubleshooting

**`ModuleNotFoundError: No module named 'gradio'`**

Install Gradio:

```bash
pip install gradio
```

**`FileNotFoundError: Checkpoint not found`**

Train the R2 strategy first or point `--checkpoint` to an existing `.pt` checkpoint.

On Windows, available robustness checkpoints can be checked using:

```bash
dir experiments\robustness\checkpoints
```

**Slow first launch**

The SigLIP2 backbone may need to be downloaded from Hugging Face during the first launch. Wait for the Gradio URL to appear.

**CUDA out of memory**

The application automatically detects whether CUDA is available. CPU inference can also be used, although it will be slower.

---

## Project Structure

```text
robust-aigc-detection/
├── src/              # Core model, training, data and evaluation code
├── experiments/      # Generalisation, robustness and hybrid experiments
├── demo/             # Interactive Gradio application
├── checkpoints/      # Locally saved baseline checkpoints
├── results/          # Evaluation results and summary tables
└── requirements.txt  # Python dependencies
```

---

## Limitations and Future Improvements

While our final R2 model improves robustness to common post-processing transformations, several limitations and opportunities for further improvement remain.

- **Limited training data:** Experiments were conducted using a balanced subset of SID_Set due to time and computational constraints. Given more resources, training on a larger and more diverse dataset could further improve the detector's performance and reliability.

- **Limited transformation coverage:** R2 specifically trains with JPEG compression, Gaussian blur, and resize round-trip. Although the model is evaluated against additional transformations, real-world images may undergo other distortions or combinations that were not covered in our experiments.

- **Broader generalisation testing:** While the model demonstrates strong performance on the external WildFake subset, this represents only a subset of possible AI-generation methods. Given more time, we would evaluate the detector on additional unseen generators and external datasets to further validate its generalisation.

- **Fixed augmentation strategy:** R2 applies the same sequence of transformations during training. Future work could explore different transformation combinations, augmentation strengths, or adaptive augmentation strategies to further improve robustness.

- **Limited architecture exploration:** Our experiments showed that additional model complexity did not necessarily improve performance. Future work could investigate other lightweight complementary signals while maintaining the efficiency of the frozen SigLIP2 backbone and linear classifier.

- **Explainability via VLM reasoning:** The current detector outputs only a scalar probability, offering no justification for its decision. Future work could pair the classifier with a vision-language model to generate human-readable rationales (e.g. flagging implausible textures, lighting inconsistencies, or anatomical artefacts), improving user trust and making the system more useful in downstream moderation or forensic settings.

Given more time and computational resources, we would train on a larger and more diverse dataset, evaluate against additional AI generators and real-world distortions, and further optimise the augmentation strategy to improve robustness while maintaining strong clean-image and external performance.

---

## Team Member Contributions

- **Ong Yu Jie** — Baseline model and training infrastructure
- **Ng Jia Yi** — Generalisation experiments
- **Rajaram Sushmiithaa** — Robustness experiments
- **Rachel Yao Xin Ru** — Hybrid detection experiments
- **Li Meiyi** — Evaluation and error analysis
