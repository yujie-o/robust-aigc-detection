import argparse

import torch

from PIL import Image

from transformers import (
    AutoProcessor
)

from models.baseline import (
    AIGCDetector,
    MODEL_NAME,
)


def load_detector(
    checkpoint_path,
    device,
):
    model = AIGCDetector(
        freeze_backbone=True
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


def predict_probability(
    model,
    processor,
    image,
    device,
):
    image = image.convert(
        "RGB"
    )

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(device)
    )

    with torch.no_grad():

        logits = model(
            pixel_values
        )

        probability = (
            torch.sigmoid(
                logits
            )
        )

    return probability.item()


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "baseline_best.pt"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    model = load_detector(
        args.checkpoint,
        device,
    )

    image = Image.open(
        args.image
    )

    probability = (
        predict_probability(
            model,
            processor,
            image,
            device,
        )
    )

    print(
        f"P(AI): "
        f"{probability:.4f}"
    )


if __name__ == "__main__":
    main()