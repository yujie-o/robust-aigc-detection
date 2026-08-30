import os, sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from models.baseline import MODEL_NAME, AIGCDetector

class PatchShuffle(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(PatchShuffle, self).__init__()
        
        self.orig_backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.shuf_backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.feature_dim = self.orig_backbone.config.hidden_size
        
        if freeze_backbone:
            for param in self.orig_backbone.parameters():
                param.requires_grad = False
            for param in self.shuf_backbone.parameters():
                param.requires_grad = False

        self.orig_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.shuf_proj = nn.Linear(self.feature_dim, self.feature_dim)

        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, img_orig, img_shuf):
        # Forward pass through SigLIP 2 backbones; pooler_output yields the global image embedding vector [Batch, Hidden_Size]
        feat_orig = self.orig_backbone(pixel_values=img_orig).pooler_output
        feat_shuf = self.shuf_backbone(pixel_values=img_shuf).pooler_output
        
        # Extract linear embeddings
        emb_orig = self.orig_proj(feat_orig)
        emb_shuf = self.shuf_proj(feat_shuf)
        
        # Concatenate embeddings
        combined = torch.cat((emb_orig, emb_shuf), dim=1)
        
        # Classify 
        logits = self.classifier(combined)
        
        return logits