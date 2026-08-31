"""
Wraps NN classifier as a nn.Module with AIGCDetector-compatible interface.

Lets P5's evaluator run the NN classifier via load_model_for_eval(
    checkpoint, device, model_class=NNClassifierModel).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from models.baseline import MODEL_NAME  # noqa: E402


class NNClassifierModel(nn.Module):
    """
    NN classifier as a nn.Module. Feature bank stored as buffers so state_dict
    saves/loads them. forward returns pseudo-logits so downstream sigmoid
    recovers P(AI).
    """

    K = 5
    FEATURE_DIM = 768
    BANK_SIZE = 1600
    EPS = 1e-6

    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.register_buffer(
            "bank_features",
            torch.zeros(self.BANK_SIZE, self.FEATURE_DIM),
        )
        self.register_buffer(
            "bank_labels",
            torch.zeros(self.BANK_SIZE, dtype=torch.long),
        )

    @torch.no_grad()
    def forward(self, pixel_values):
        outputs = self.backbone.get_image_features(pixel_values=pixel_values)
        features = outputs.pooler_output
        features = F.normalize(features, p=2, dim=1)

        sims = features @ self.bank_features.T
        _, topk_idx = sims.topk(self.K, dim=1)
        neighbor_labels = self.bank_labels[topk_idx]
        probs = neighbor_labels.float().mean(dim=1)

        probs = probs.clamp(self.EPS, 1.0 - self.EPS)
        logits = torch.log(probs / (1.0 - probs))
        return logits