import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel


MODEL_NAME = "google/siglip2-base-patch16-256"


processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

image = Image.new("RGB", (256, 256), "white")

inputs = processor(
    images=image,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.get_image_features(**inputs)

print(type(outputs))
print(outputs.pooler_output.shape)