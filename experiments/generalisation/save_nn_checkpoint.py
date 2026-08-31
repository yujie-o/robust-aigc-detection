"""
Packages feature_bank.pt into an NNClassifierModel checkpoint compatible with
P5's load_model_for_eval.

Run once after nn_classifier.py has produced feature_bank.pt.

Usage:
    python experiments/generalisation/save_nn_checkpoint.py
"""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from experiments.generalisation.nn_classifier_model import NNClassifierModel  # noqa: E402


BANK_PATH = os.path.join(
    REPO_ROOT, "experiments", "generalisation", "results", "feature_bank.pt"
)
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints", "p2a_nn_best.pt")


def main():
    print(f"Loading feature bank from {BANK_PATH}...")
    bank_data = torch.load(BANK_PATH, map_location="cpu")
    bank_features = bank_data["features"]
    bank_labels = bank_data["labels"]
    print(f"Bank: {bank_features.shape[0]} features, dim={bank_features.shape[1]}")

    print("Creating NNClassifierModel and copying bank into buffers...")
    model = NNClassifierModel()
    model.bank_features.copy_(bank_features)
    model.bank_labels.copy_(bank_labels)

    print(f"Saving checkpoint to {CHECKPOINT_PATH}...")
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, CHECKPOINT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()