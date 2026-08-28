import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

import torch
from PIL import Image
from transformers import AutoProcessor

from models.baseline import (
    AIGCDetector,
    MODEL_NAME,
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AIGCDetector(
    freeze_backbone=True
)

model.eval()

image = Image.new(
    "RGB",
    (256, 256),
    "white"
)

inputs = processor(
    images=image,
    return_tensors="pt"
)

with torch.no_grad():
    logits = model(
        inputs["pixel_values"]
    )

    probabilities = torch.sigmoid(
        logits
    )

print("Logit:", logits)
print("Logit shape:", logits.shape)

print("P(AI):", probabilities)
print(
    "Probability shape:",
    probabilities.shape
)