import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Optional

from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.transforms import TRANSFORM_CONDITIONS, CONDITION_GROUPS

if TYPE_CHECKING:
    import torch
    from PIL import Image

Sample = Dict


def load_model_for_eval(
    checkpoint_path: str,
    device: "torch.device",
    model_class: Optional[Callable] = None,
):
    """
    Loads model checkpoint weights onto the target device.
    Defaults to baseline AIGCDetector if model_class is not specified.
    """
    import torch

    if model_class is None:
        from models.baseline import AIGCDetector
        model = AIGCDetector(freeze_backbone=True).to(device)
    else:
        model = model_class().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference_batched(
    model,
    processor,
    images: Sequence["Image.Image"],
    device: "torch.device",
    batch_size: int = 16,
) -> List[float]:
    """
    Runs batched inference and returns 1D probability scores P(AI).
    """
    import torch

    probs: List[float] = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            batch_rgb = [img.convert("RGB") for img in batch]

            inputs = processor(images=batch_rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            logits = model(pixel_values)
            batch_probs = torch.sigmoid(logits).view(-1).cpu().tolist()

            probs.extend(batch_probs)

    return probs


def evaluate_condition(
    model,
    processor,
    samples: Sequence[Sample],
    transform_fn: Callable[["Image.Image"], "Image.Image"],
    device: "torch.device",
    batch_size: int = 16,
) -> Dict:
    """
    Evaluates model predictions and ROC-AUC on a single transform condition.
    """
    transformed_images = [transform_fn(s["image"]) for s in samples]
    labels = [int(s["label"]) for s in samples]

    probs = run_inference_batched(
        model, processor, transformed_images, device, batch_size
    )

    auc = roc_auc_score(labels, probs)

    per_image = [
        {
            "path": samples[i].get("path"),
            "label": labels[i],
            "pred": probs[i],
        }
        for i in range(len(samples))
    ]

    return {"auc": auc, "predictions": per_image}


def run_full_evaluation(
    model,
    samples: Sequence[Sample],
    device: "torch.device",
    processor_name: Optional[str] = None,
    batch_size: int = 16,
) -> Dict:
    from transformers import AutoProcessor
    from models.baseline import MODEL_NAME

    processor = AutoProcessor.from_pretrained(processor_name or MODEL_NAME)

    results = {}
    for condition_name, transform_fn in TRANSFORM_CONDITIONS.items():
        print(f"Evaluating condition: {condition_name} ({len(samples)} images)...")
        results[condition_name] = evaluate_condition(
            model, processor, samples, transform_fn, device, batch_size
        )

    return results


def summarize_by_group(condition_results: Dict) -> Dict[str, float]:
    summary = {}
    for group_name, condition_names in CONDITION_GROUPS.items():
        aucs = [condition_results[c]["auc"] for c in condition_names]
        summary[group_name] = sum(aucs) / len(aucs)
    return summary


def compute_robust_mean(summary: Dict[str, float]) -> float:
    robust_groups = [v for k, v in summary.items() if k != "clean"]
    return sum(robust_groups) / len(robust_groups)


def format_results_row(
    model_name: str, summary: Dict[str, float], external_auc: Optional[float] = None
) -> str:
    robust_mean = compute_robust_mean(summary)
    ext = f"{external_auc:.4f}" if external_auc is not None else "-"

    return (
        f"| {model_name} "
        f"| {summary['clean']:.4f} "
        f"| {summary['jpeg']:.4f} "
        f"| {summary['blur']:.4f} "
        f"| {summary['resize']:.4f} "
        f"| {summary['noise']:.4f} "
        f"| {summary['color']:.4f} "
        f"| {summary['crop']:.4f} "
        f"| {robust_mean:.4f} "
        f"| {ext} |"
    )


def get_false_positives_negatives(
    predictions: List[Dict], threshold: float = 0.5
) -> Dict:
    false_positives = [
        p for p in predictions if p["label"] == 0 and p["pred"] >= threshold
    ]
    false_negatives = [
        p for p in predictions if p["label"] == 1 and p["pred"] < threshold
    ]

    false_positives.sort(key=lambda p: p["pred"], reverse=True)
    false_negatives.sort(key=lambda p: p["pred"])

    return {"false_positives": false_positives, "false_negatives": false_negatives}


def find_hardest_condition(condition_results: Dict) -> str:
    return min(condition_results, key=lambda name: condition_results[name]["auc"])


def compute_degradation(summary: Dict[str, float]) -> Dict[str, float]:
    """
    AUC drop of each non-clean group relative to clean.
    """
    clean = summary["clean"]
    return {group: clean - auc for group, auc in summary.items() if group != "clean"}


def find_biggest_degradation(summary: Dict[str, float]) -> str:
    """
    Name of the group with the largest AUC drop relative to clean.
    """
    degradation = compute_degradation(summary)
    return max(degradation, key=lambda group: degradation[group])


def jpeg_trend(condition_results: Dict) -> Dict[str, float]:
    """
    AUC at each JPEG quality level to show how detection
    degrades as compression gets more aggressive.
    """
    return {c: condition_results[c]["auc"] for c in CONDITION_GROUPS["jpeg"]}


def internal_external_gap(
    summary: Dict[str, float], external_auc: Optional[float]
) -> Optional[float]:
    """
    Gap between internal clean AUC and external (WildFake) AUC. A large gap
    suggests the model may be relying on dataset-specific signals rather
    than generalisable AIGC cues. Returns None if no external_auc was run.
    """
    if external_auc is None:
        return None
    return summary["clean"] - external_auc