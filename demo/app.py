"""Gradio demo for the AIGC Detection project.

Loads a trained AIGCDetector checkpoint (baseline + P3 augmentation strategy)
and lets a user upload an image, optionally apply one of the P3 post-processing
transforms, and see the model's authenticity score.
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor

# --- Repo path setup (so we can import from src/) ---------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from augmentation import (
    jpeg_compress,
    gaussian_blur,
    resize_downup,
    gaussian_noise,
    color_jitter,
    center_crop_pct,
)
from models.baseline import AIGCDetector, MODEL_NAME

# --- Config -----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINT = ROOT_DIR / "experiments" / "robustness" / "checkpoints" / "R2_best.pt"

# Preset transforms for the demo dropdown. Fixed params (not randomised)
# so the on-camera demo is reproducible.
TRANSFORMS = {
    "None": lambda im: im,
    "JPEG q=50": lambda im: jpeg_compress(im, 50),
    "JPEG q=30": lambda im: jpeg_compress(im, 30),
    "Gaussian blur (sigma=1.0)": lambda im: gaussian_blur(im, 1.0),
    "Gaussian blur (sigma=2.0)": lambda im: gaussian_blur(im, 2.0),
    "Resize (0.5x -> up)": lambda im: resize_downup(im, 0.5),
    "Resize (0.25x -> up)": lambda im: resize_downup(im, 0.25),
    "Gaussian noise (sigma=0.05)": lambda im: gaussian_noise(im, 0.05),
    "Color jitter": lambda im: color_jitter(im, 0.2),
    "Center crop 80%": lambda im: center_crop_pct(im, 0.8),
}

# --- Model loading ----------------------------------------------------------
def load_model(checkpoint_path: Path):
    print(f"Loading backbone {MODEL_NAME} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AIGCDetector(freeze_backbone=True).to(DEVICE)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Pass --checkpoint <path> or train a strategy first with "
            f"experiments/robustness/robust_augmentation_experiment.py"
        )

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    strategy = ckpt.get("strategy", "?")
    val_auc = ckpt.get("val_auc", None)
    print(f"Loaded strategy={strategy} val_auc={val_auc}")

    return model, processor, strategy, val_auc

# --- Inference --------------------------------------------------------------
@torch.no_grad()
def predict(model, processor, img: Image.Image) -> float:
    inputs = processor(images=img.convert("RGB"), return_tensors="pt").to(DEVICE)
    prob = model.predict_proba(inputs["pixel_values"]).item()
    return prob

def analyze(model, processor, img, transform_name):
    if img is None:
        return None, "### Upload an image to begin.", 0.0

    transform_fn = TRANSFORMS[transform_name]
    transformed = transform_fn(img.convert("RGB"))
    prob = predict(model, processor, transformed)

    label = "AI-generated" if prob >= 0.5 else "Authentic"
    emoji = "🤖" if prob >= 0.5 else "📷"
    confidence = prob if prob >= 0.5 else (1 - prob)

    verdict_md = (
        f"### {emoji} {label}\n"
        f"**AI probability:** `{prob:.4f}`  \n"
        f"**Confidence:** `{confidence:.1%}`  \n"
        f"**Transform applied:** {transform_name}"
    )
    return transformed, verdict_md, prob

# --- UI ---------------------------------------------------------------------
def build_ui(model, processor, strategy, val_auc):
    header_md = (
        "# AIGC Detection Demo\n"
        "**SigLIP2-Base (frozen) + Linear probe**, trained with augmentation "
        f"strategy **{strategy}**"
        + (f" (val AUC: {val_auc:.4f})" if val_auc is not None else "")
        + "\n\n"
        "Upload an image and optionally apply a post-processing transform "
        "to test the model's robustness to JPEG compression, blur, resizing, "
        "noise, and other common distortions."
    )

    with gr.Blocks(title="AIGC Detection Demo") as demo:
        gr.Markdown(header_md)

        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Input image", height=350)
                transform_sel = gr.Dropdown(
                    choices=list(TRANSFORMS.keys()),
                    value="None",
                    label="Post-processing transform (robustness test)",
                )
                run_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column():
                img_out = gr.Image(type="pil", label="What the model sees", height=350)
                verdict = gr.Markdown("### Upload an image to begin.")
                score_bar = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    label="AI probability (0 = authentic, 1 = AI-generated)",
                    interactive=False,
                )

        run_btn.click(
            fn=lambda img, t: analyze(model, processor, img, t),
            inputs=[img_in, transform_sel],
            outputs=[img_out, verdict, score_bar],
        )

        gr.Markdown(
            "---\n"
            "*Backbone: `google/siglip2-base-patch16-256` (frozen). "
            "Classifier: single linear layer on pooler output. "
            "See repo for full architecture and training details.*"
        )

    return demo

# --- Entrypoint -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gradio demo for AIGC detection.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to trained checkpoint (.pt). Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model, processor, strategy, val_auc = load_model(args.checkpoint)
    demo = build_ui(model, processor, strategy, val_auc)
    demo.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()