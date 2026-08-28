import torch
import torch.nn as nn
from transformers import AutoModel


MODEL_NAME = "google/siglip2-base-patch16-256"


class AIGCDetector(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        # General-purpose pretrained backbone
        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME
        )

        # First baseline: freeze SigLIP2
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        feature_dim = (
            self.backbone.config.vision_config.hidden_size
        )

        # Our own Real-vs-AI classifier
        self.classifier = nn.Linear(
            feature_dim,
            1
        )

    def forward(self, pixel_values):
        outputs = self.backbone.get_image_features(
            pixel_values=pixel_values
        )

        features = outputs.pooler_output

        logits = self.classifier(
            features
        ).squeeze(-1)

        return logits

    def predict_proba(self, pixel_values):
        logits = self.forward(
            pixel_values
        )

        return torch.sigmoid(
            logits
        )