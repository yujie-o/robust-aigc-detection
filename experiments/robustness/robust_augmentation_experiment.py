import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "experiments" / "robustness"
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from augmentation import apply_augmentation_strategy
from data.sid_dataset import load_balanced_sid_subset
from models.baseline import AIGCDetector, MODEL_NAME

SEED = 42
VAL_RATIO = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
EPOCHS = 1


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)


def split_samples(samples, val_ratio: float = VAL_RATIO, seed: int = SEED):
    real = [x for x in samples if int(x["label"]) == 0]
    ai = [x for x in samples if int(x["label"]) == 1]

    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(ai)

    val_real_count = int(len(real) * val_ratio)
    val_ai_count = int(len(ai) * val_ratio)

    val_samples = real[:val_real_count] + ai[:val_ai_count]
    train_samples = real[val_real_count:] + ai[val_ai_count:]
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


class StrategyDataset(Dataset):
    """Dataset wrapper that applies a specific augmentation strategy to training items."""

    def __init__(self, samples, strategy: str, apply_to_val: bool = False):
        self.samples = samples
        self.strategy = strategy
        self.apply_to_val = apply_to_val

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        image = item["image"].convert("RGB")

        if self.strategy != "R0" and not self.apply_to_val:
            image = apply_augmentation_strategy(image, self.strategy)

        label = int(item["label"])
        return image, label


def make_collate_fn(processor):
    def collate_fn(batch):
        images, labels = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        labels = torch.tensor(labels, dtype=torch.float32)
        return inputs["pixel_values"], labels

    return collate_fn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        logits = model(pixel_values)
        probs = torch.sigmoid(logits)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)
    return auc


def evaluate_robustness(model, processor, samples, device, transform_name: str = "clean"):
    model.eval()
    all_labels = []
    all_probs = []

    for item in samples:
        image = item["image"].convert("RGB")

        if transform_name != "clean":
            image = apply_augmentation_strategy(image, transform_name)

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            logits = model(pixel_values)
            probs = torch.sigmoid(logits).cpu().squeeze(0).item()

        all_labels.append(int(item["label"]))
        all_probs.append(probs)

    return roc_auc_score(all_labels, all_probs)


def run_strategy(strategy: str, samples_per_class: int = 1000, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, val_ratio: float = VAL_RATIO, seed: int = SEED):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AIGCDetector(freeze_backbone=True).to(device)

    samples = load_balanced_sid_subset(samples_per_class=samples_per_class, seed=seed)
    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio, seed=seed)

    train_dataset = StrategyDataset(train_samples, strategy=strategy)
    val_dataset = StrategyDataset(val_samples, strategy="R0", apply_to_val=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_collate_fn(processor))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(processor))

    classifier_params = [p for p in model.parameters() if p.requires_grad]
    if not classifier_params:
        raise RuntimeError("No trainable parameters found in the model. Check the backbone freeze configuration.")

    optimizer = torch.optim.AdamW(classifier_params, lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, val_loader, device)
        print(f"Strategy={strategy} Epoch={epoch + 1}/{epochs} TrainLoss={train_loss:.4f} ValAUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint produced for strategy={strategy}")

    out_dir = RESULTS_DIR / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{strategy}_best.pt"
    torch.save({"model_state_dict": best_state, "strategy": strategy, "val_auc": best_auc, "epoch": best_epoch}, ckpt_path)

    clean_auc = evaluate_robustness(model, processor, val_samples, device, transform_name="clean")
    jpeg_auc = evaluate_robustness(model, processor, val_samples, device, transform_name="jpeg")
    blur_auc = evaluate_robustness(model, processor, val_samples, device, transform_name="blur")
    resize_auc = evaluate_robustness(model, processor, val_samples, device, transform_name="resize")
    noise_auc = evaluate_robustness(model, processor, val_samples, device, transform_name="noise")
    color_auc = evaluate_robustness(model, processor, val_samples, device, transform_name="color")
    crop_auc = evaluate_robustness(model, processor, val_samples, device, transform_name="crop")

    result = {
        "strategy": strategy,
        "samples_per_class": samples_per_class,
        "epochs": epochs,
        "batch_size": batch_size,
        "val_ratio": val_ratio,
        "best_val_auc": best_auc,
        "clean_auc": clean_auc,
        "jpeg_auc": jpeg_auc,
        "blur_auc": blur_auc,
        "resize_auc": resize_auc,
        "noise_auc": noise_auc,
        "color_auc": color_auc,
        "crop_auc": crop_auc,
        "best_epoch": best_epoch,
    }

    result_path = RESULTS_DIR / "results" / f"{strategy}_summary.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a controlled augmentation experiment against the baseline SigLIP2 classifier.")
    parser.add_argument("--strategy", choices=["R0", "R1", "R2", "R3"], default="R0", help="Which augmentation strategy to apply.")
    parser.add_argument("--all", action="store_true", help="Run R0/R1/R2/R3 sequentially.")
    parser.add_argument("--samples-per-class", type=int, default=1000, help="Balanced samples per class in the training subset.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs for the experiment.")
    parser.add_argument("--batch-size", type=int, default=4, help="DataLoader batch size.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.all:
        strategies = ["R0", "R1", "R2", "R3"]
        results = []
        for strategy in strategies:
            print(f"\n=== Running {strategy} ===")
            results.append(run_strategy(strategy=strategy, samples_per_class=args.samples_per_class, epochs=args.epochs, batch_size=args.batch_size, val_ratio=args.val_ratio, seed=args.seed))

        summary_path = RESULTS_DIR / "results" / "strategy_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved aggregate summary to {summary_path}")
        return

    result = run_strategy(strategy=args.strategy, samples_per_class=args.samples_per_class, epochs=args.epochs, batch_size=args.batch_size, val_ratio=args.val_ratio, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
