"""
CLI entry point for running the common evaluator on the P2-a NN classifier.

Copy of src/evaluation/run_evaluation.py with two changes:
  1. Imports NNClassifierModel
  2. Passes model_class=NNClassifierModel to load_model_for_eval

Usage:
    python experiments/generalisation/run_p2_evaluation.py \
        --checkpoint checkpoints/p2a_nn_best.pt \
        --model-name P2a-NN \
        --external-samples 200

Produces:
    - results/<model_name>_conditions.json
    - Appends row to results/summary_table.md
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor

# Add src/ and repo root to sys.path so imports resolve regardless of CWD
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train import split_samples, SAMPLES_PER_CLASS, VAL_RATIO, SEED
from data.sid_dataset import load_balanced_sid_subset
from experiments.generalisation.organiser_eval_loader import load_organiser_eval_samples
from experiments.generalisation.nn_classifier_model import NNClassifierModel

from evaluation.evaluator import (
    load_model_for_eval,
    run_full_evaluation,
    summarize_by_group,
    format_results_row,
    evaluate_condition,
    compute_degradation,
    find_biggest_degradation,
    jpeg_trend,
    internal_external_gap,
)
from evaluation.transforms import TRANSFORM_CONDITIONS
from models.baseline import MODEL_NAME


RESULTS_DIR = ROOT_DIR / "results"
SUMMARY_TABLE_PATH = RESULTS_DIR / "summary_table.md"

TABLE_HEADER = (
    "| Model | Clean | JPEG | Blur | Resize | Noise | Color | Crop | Robust Mean | External |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def get_internal_eval_samples():
    samples = load_balanced_sid_subset(
        samples_per_class=SAMPLES_PER_CLASS,
        seed=SEED,
    )
    _, val_samples = split_samples(samples, val_ratio=VAL_RATIO)

    return [
        {
            "image": s["image"].convert("RGB"),
            "label": int(s["label"]),
            "path": s.get("path", "sid_validation_stream"),
        }
        for s in val_samples
    ]


def get_external_eval_samples(samples_per_class: int = 200, seed: int = 42):
    """
    Organiser-provided WildFake external validation subset:
      - Real: COCO val2017 images
      - AI: DALLE3 images with IsAdvanced=1
    """
    return load_organiser_eval_samples(
        samples_per_class=samples_per_class,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model .pt weights")
    parser.add_argument("--model-name", required=True, help="Row label")
    parser.add_argument("--external-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation Device:", device)

    print(f"Loading NN classifier checkpoint: {args.checkpoint}")
    model = load_model_for_eval(args.checkpoint, device, model_class=NNClassifierModel)

    # 1. Internal Robustness Evaluation (SID_Set Held-out Split across conditions)
    print("\n--- Running Internal Robustness Evaluation (SID_Set) ---")
    internal_samples = get_internal_eval_samples()
    print(f"Loaded {len(internal_samples)} internal validation samples.")

    condition_results = run_full_evaluation(
        model, internal_samples, device, batch_size=args.batch_size
    )
    summary = summarize_by_group(condition_results)

    # 2. External Generalisation Benchmark
    print("\n--- Running External Generalisation Evaluation (WildFake) ---")
    external_samples = get_external_eval_samples(samples_per_class=args.external_samples)
    print(f"Loaded {len(external_samples)} external benchmark samples.")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    ext_result = evaluate_condition(
        model,
        processor,
        external_samples,
        TRANSFORM_CONDITIONS["clean"],
        device,
        args.batch_size,
    )
    external_auc = ext_result["auc"]
    external_predictions = ext_result["predictions"]

    # 2b. Error analysis
    degradation_by_group = compute_degradation(summary)
    biggest_degradation_group = find_biggest_degradation(summary)
    jpeg_trend_result = jpeg_trend(condition_results)
    gap = internal_external_gap(summary, external_auc)

    print("\n--- Error Analysis ---")
    print(f"Biggest degradation vs. clean: {biggest_degradation_group} "
          f"(-{degradation_by_group[biggest_degradation_group]:.4f})")
    print(f"JPEG trend (q90 -> q30): {jpeg_trend_result}")
    if gap is not None:
        print(f"Internal clean AUC - External AUC gap: {gap:.4f}")

    # 3. Save full JSON details
    output_path = RESULTS_DIR / f"{args.model_name}_conditions.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "conditions": {
                    name: {"auc": r["auc"], "predictions": r["predictions"]}
                    for name, r in condition_results.items()
                },
                "summary": summary,
                "external_auc": external_auc,
                "external_predictions": external_predictions,
                "error_analysis": {
                    "degradation_by_group": degradation_by_group,
                    "biggest_degradation_group": biggest_degradation_group,
                    "jpeg_trend": jpeg_trend_result,
                    "internal_external_gap": gap,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved detailed analysis to: {output_path}")

    # 4. Append row to Markdown summary table
    row = format_results_row(args.model_name, summary, external_auc)
    if not SUMMARY_TABLE_PATH.exists():
        SUMMARY_TABLE_PATH.write_text(TABLE_HEADER + "\n" + row + "\n")
    else:
        with open(SUMMARY_TABLE_PATH, "a") as f:
            f.write(row + "\n")

    print("\nSummary Table Row:")
    print(row)
    print(f"Summary table updated at: {SUMMARY_TABLE_PATH}")


if __name__ == "__main__":
    main()