import os
import random
from pathlib import Path
from PIL import Image


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WILDFAKE_ROOT = Path(project_root) / 'raw_data' / 'wildfake'

REAL_DIR = WILDFAKE_ROOT / 'val2017'   # label 0
FAKE_DIR = WILDFAKE_ROOT / 'dalle'     # label 1 

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def _list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return [str(p) for p in folder.rglob('*') if p.suffix.lower() in IMAGE_EXTS]


def load_balanced_wildfake_subset(
    samples_per_class: int = None, seed: int = 42
):
    rng = random.Random(seed)

    real_paths = _list_images(REAL_DIR)
    fake_paths = _list_images(FAKE_DIR)

    if not real_paths:
        raise FileNotFoundError(f"No images found in {REAL_DIR}")
    if not fake_paths:
        raise FileNotFoundError(f"No images found in {FAKE_DIR}")

    rng.shuffle(real_paths)
    rng.shuffle(fake_paths)

    # Slice to the requested samples_per_class limit
    if samples_per_class is not None:
        real_paths = real_paths[:samples_per_class]
        fake_paths = fake_paths[:samples_per_class]

    samples = [{"image_path": p, "label": 0} for p in real_paths] + [
        {"image_path": p, "label": 1} for p in fake_paths
    ]

    rng.shuffle(samples)

    formatted_samples = []
    for item in samples:
        img = Image.open(item["image_path"]).convert("RGB")
        formatted_samples.append(
            {
                "image": img,
                "label": int(item["label"]),
                "path": str(item["image_path"]),
            }
        )

    return formatted_samples

if __name__ == "__main__":
    real_paths = _list_images(REAL_DIR)
    fake_paths = _list_images(FAKE_DIR)
    print(f"real (val2017): {len(real_paths)}")
    print(f"fake (dalle): {len(fake_paths)}")