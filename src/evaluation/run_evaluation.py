import argparse
import json
from pathlib import Path
from PIL import Image
import sys
import torch
from transformers import AutoProcessor

# Ensure src/ is in sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import split_samples, SAMPLES_PER_CLASS, VAL_RATIO, SEED
from data.sid_dataset import load_balanced_sid_subset
from data.wildfake_subset import load_balanced_wildfake_subset

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


RESULTS_DIR = Path("results")
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

    formatted_samples = []
    for s in val_samples:
        img = (
            s["image"].convert("RGB")
            if hasattr(s["image"], "convert")
            else Image.open(s["image"]).convert("RGB")
        )
        formatted_samples.append(
            {
                "image": img,
                "label": int(s["label"]),
                "path": s.get("path", "sid_validation_stream"),
            }
        )
    return formatted_samples


def get_external_eval_samples(samples_per_class: None, seed: int = 42):
    return load_balanced_wildfake_subset(
        samples_per_class=samples_per_class,
        seed=seed,
    )


# =========================================================================
# Function 1: External Dataset Only (All Transforms on WildFake)
# =========================================================================
def run_external_only_evaluation(model, device, args):
    print("\n=======================================================")
    print(" Running EXTERNAL-ONLY Evaluation (WildFake) ")
    print("=======================================================")

    external_samples = get_external_eval_samples(
        samples_per_class=args.samples_per_class
    )
    print(f"Loaded {len(external_samples)} external benchmark samples.")

    condition_results = run_full_evaluation(
        model, external_samples, device, batch_size=args.batch_size
    )
    summary = summarize_by_group(condition_results)
    clean_external_auc = summary.get("clean", None)

    degradation_by_group = compute_degradation(summary)
    biggest_degradation_group = find_biggest_degradation(summary)
    jpeg_trend_result = jpeg_trend(condition_results)

    print("\n--- Error Analysis (External Only) ---")
    if clean_external_auc is not None:
        print(f"Clean External AUC: {clean_external_auc:.4f}")
    if biggest_degradation_group:
        print(
            f"Biggest degradation vs. clean: {biggest_degradation_group} "
            f"(-{degradation_by_group[biggest_degradation_group]:.4f})"
        )
    print(f"JPEG trend (q90 -> q30): {jpeg_trend_result}")

    output_path = RESULTS_DIR / f"{args.model_name}_external_conditions.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "evaluation_type": "external_only",
                "dataset_evaluated": "WildFake",
                "conditions": {
                    name: {"auc": r["auc"], "predictions": r["predictions"]}
                    for name, r in condition_results.items()
                },
                "summary": summary,
                "error_analysis": {
                    "degradation_by_group": degradation_by_group,
                    "biggest_degradation_group": biggest_degradation_group,
                    "jpeg_trend": jpeg_trend_result,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved detailed analysis to: {output_path}")

    row = format_results_row(args.model_name, summary, external_auc=clean_external_auc)
    if not SUMMARY_TABLE_PATH.exists():
        SUMMARY_TABLE_PATH.write_text(TABLE_HEADER + "\n" + row + "\n")
    else:
        with open(SUMMARY_TABLE_PATH, "a") as f:
            f.write(row + "\n")

    print("\nSummary Table Row:")
    print(row)
    print(f"Summary table updated at: {SUMMARY_TABLE_PATH}")


# =========================================================================
# Function 2: Combined Internal Robustness + External Generalisation
# =========================================================================
def run_full_combined_evaluation(model, device, args):
    print("\n=======================================================")
    print(" Running COMBINED Evaluation (Internal + External) ")
    print("=======================================================")

    # 1. Internal Robustness Evaluation (SID_Set Held-out Split)
    print("\n--- 1. Running Internal Robustness Evaluation (SID_Set) ---")
    internal_samples = get_internal_eval_samples()
    print(f"Loaded {len(internal_samples)} internal validation samples.")

    condition_results = run_full_evaluation(
        model, internal_samples, device, batch_size=args.batch_size
    )
    summary = summarize_by_group(condition_results)

    # 2. External Generalisation Benchmark (WildFake Clean)
    print("\n--- 2. Running External Generalisation Benchmark (WildFake) ---")
    external_samples = get_external_eval_samples(
        samples_per_class=args.samples_per_class
    )
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

    # 3. Error Analysis
    degradation_by_group = compute_degradation(summary)
    biggest_degradation_group = find_biggest_degradation(summary)
    jpeg_trend_result = jpeg_trend(condition_results)
    gap = internal_external_gap(summary, external_auc)

    print("\n--- Error Analysis ---")
    if biggest_degradation_group:
        print(
            f"Biggest degradation vs. clean: {biggest_degradation_group} "
            f"(-{degradation_by_group[biggest_degradation_group]:.4f})"
        )
    print(f"JPEG trend (q90 -> q30): {jpeg_trend_result}")
    if gap is not None:
        print(f"Internal clean AUC - External AUC gap: {gap:.4f}")

    # 4. Save JSON details
    output_path = RESULTS_DIR / f"{args.model_name}_conditions.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "checkpoint": args.checkpoint,
                "evaluation_type": "combined",
                "dataset_evaluated": "SID_Set (Internal) + WildFake (External)",
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

    # 5. Append to markdown summary table
    row = format_results_row(args.model_name, summary, external_auc=external_auc)
    if not SUMMARY_TABLE_PATH.exists():
        SUMMARY_TABLE_PATH.write_text(TABLE_HEADER + "\n" + row + "\n")
    else:
        with open(SUMMARY_TABLE_PATH, "a") as f:
            f.write(row + "\n")

    print("\nSummary Table Row:")
    print(row)
    print(f"Summary table updated at: {SUMMARY_TABLE_PATH}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model .pt weights")
    parser.add_argument("--model-name", required=True, help="Row label: Baseline / P2 / P3 / P4 / Final")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Samples per class")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Evaluation Device:", device)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model_for_eval(args.checkpoint, device)

    # =========================================================================
    # Choose which evaluation function to run:
    # =========================================================================
    run_full_combined_evaluation(model, device, args)
    #run_external_only_evaluation(model, device, args)


if __name__ == "__main__":
    main()