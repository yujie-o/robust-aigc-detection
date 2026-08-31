import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# --- Single source of truth for all data/split/training constants ---------
from train import (
    SEED,
    SAMPLES_PER_CLASS,
    VAL_RATIO,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    set_seed,
    split_samples,
    make_collate_fn,
    train_one_epoch,
    evaluate,
)

from augmentation import apply_augmentation_strategy
from data.sid_dataset import load_balanced_sid_subset
from models.baseline import AIGCDetector, MODEL_NAME

# --- Single source of truth for evaluation -------------------------------
EVAL_SCRIPT_DIR = SRC_DIR / "evaluation"
sys.path.insert(0, str(EVAL_SCRIPT_DIR))
from evaluation.run_evaluation import (
    run_full_combined_evaluation,
    RESULTS_DIR as EVAL_RESULTS_DIR,
)

CHECKPOINT_DIR = ROOT_DIR / "experiments" / "robustness" / "checkpoints"

class StrategyDataset(Dataset):
    """Applies a training-time augmentation strategy. Never applied to val data."""

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

def run_strategy(
    strategy: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    samples_per_class: int = SAMPLES_PER_CLASS,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
):
    """Train one strategy's model, then evaluate it with run_evaluation.py's
    combined evaluator so it lands in the shared summary table.

    samples_per_class/val_ratio/seed default to train.py's constants so the
    split matches the baseline out of the box, but can be overridden - just
    know that doing so means this run's val split will no longer match
    train.py's, so its numbers aren't directly comparable to the baseline
    row in summary_table.md.
    """

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AIGCDetector(freeze_backbone=True).to(device)

    # Same samples, same split as train.py by default -> R0-R3 and the
    # baseline model are validated on identical held-out data unless the
    # caller explicitly overrides samples_per_class/val_ratio/seed.
    samples = load_balanced_sid_subset(samples_per_class=samples_per_class, seed=seed)
    train_samples, val_samples = split_samples(samples, val_ratio)

    train_dataset = StrategyDataset(train_samples, strategy=strategy)
    val_dataset = StrategyDataset(val_samples, strategy="R0", apply_to_val=True)

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    classifier_params = [p for p in model.parameters() if p.requires_grad]
    if not classifier_params:
        raise RuntimeError("No trainable parameters found. Check the backbone freeze configuration.")

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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{strategy}_best.pt"
    torch.save(
        {"model_state_dict": best_state, "strategy": strategy, "val_auc": best_auc, "epoch": best_epoch},
        ckpt_path,
    )

    # Load the best checkpoint's weights back into the in-memory model before
    # evaluating, so evaluation matches exactly what was saved (not
    # whatever the model's weights happen to be after the final epoch).
    model.load_state_dict(best_state)
    model.eval()

    print(f"\nRunning combined evaluation (SID_Set + WildFake) for {strategy} via run_evaluation.py...")
    EVAL_RESULTS_DIR.mkdir(exist_ok=True)

    eval_args = SimpleNamespace(
        checkpoint=str(ckpt_path),
        model_name=strategy,
        samples_per_class=samples_per_class,
        batch_size=batch_size,
    )
    run_full_combined_evaluation(model, device, eval_args)

    # run_full_combined_evaluation already wrote
    # results/{strategy}_conditions.json and appended a row to
    # results/summary_table.md - read the JSON back so callers of
    # run_strategy() still get a return value if they want one.
    result_path = EVAL_RESULTS_DIR / f"{strategy}_conditions.json"
    return json.loads(result_path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(
        description="Train R0-R3 augmentation strategies and evaluate each with run_evaluation.py's combined evaluator."
    )
    parser.add_argument("--strategy", choices=["R0", "R1", "R2", "R3"], default="R0")
    parser.add_argument("--all", action="store_true", help="Run R0/R1/R2/R3 sequentially.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs (train.py's EPOCHS by default).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (train.py's BATCH_SIZE by default).")
    parser.add_argument(
        "--samples-per-class", type=int, default=SAMPLES_PER_CLASS,
        help="Balanced samples per class (train.py's SAMPLES_PER_CLASS by default). "
             "Override only if you know this run doesn't need to match the baseline's split.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=VAL_RATIO,
        help="Validation split ratio (train.py's VAL_RATIO by default).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Split/training seed (train.py's SEED by default).",
    )
    args = parser.parse_args()

    if (args.samples_per_class, args.val_ratio, args.seed) != (SAMPLES_PER_CLASS, VAL_RATIO, SEED):
        print(
            "WARNING: --samples-per-class/--val-ratio/--seed differ from train.py's constants "
            f"(train.py: {SAMPLES_PER_CLASS}, {VAL_RATIO}, {SEED} | this run: "
            f"{args.samples_per_class}, {args.val_ratio}, {args.seed}). "
            "This run's val split will NOT match the baseline model's, so its row in "
            "summary_table.md won't be directly comparable.\n"
        )

    run_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.all:
        for strategy in ["R0", "R1", "R2", "R3"]:
            print(f"\n=== Running {strategy} ===")
            run_strategy(strategy=strategy, **run_kwargs)
        print(f"\nAll strategies evaluated. See {EVAL_RESULTS_DIR / 'summary_table.md'} for the comparison table.")
        return

    result = run_strategy(strategy=args.strategy, **run_kwargs)
    print(json.dumps(result["summary"], indent=2))

if __name__ == "__main__":
    main()

