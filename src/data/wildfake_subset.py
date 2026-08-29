"""
Downloads only the files needed for the COCO vs DALL-E WildFake subset,
instead of the full dataset via MsDataset.load(). 

Files fetched (from WildFake's "Files and versions" layout):
  - label_csv_files/dalle2.csv
  - label_csv_files/dalle3.csv
  - label_csv_files/real_coco.csv
  - Images/Diffusion_based/DALLE.zip   
  - Images/Real/coco.zip               
"""

import argparse
import random
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from modelscope.hub.snapshot_download import dataset_snapshot_download
from torch.utils.data import Dataset

DATASET_ID = "hy2628982280/WildFake"

CSV_FILES = {
    "dalle2": "label_csv_files/dalle2.csv",
    "dalle3": "label_csv_files/dalle3.csv",
    "real_coco": "label_csv_files/real_coco.csv",
}

ZIP_FILES = {
    "dalle": "Images/Diffusion_based/DALLE.zip",
    "coco": "Images/Real/coco.zip",
}

FILENAME_COLUMNS = ["Image_path", "filename", "file_name", "image", "path"]


class WildFakeSubset(Dataset):
    """
    Simple PyTorch dataset containing selected WildFake samples.
    Each sample is a dict with 'image_path' and 'label'.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        image = Image.open(item["image_path"]).convert("RGB")
        label = int(item["label"])
        return image, label


def download_minimal_files(local_dir: str = "data/wildfake_raw") -> Path:
    cache_dir = Path(local_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = list(CSV_FILES.values()) + list(ZIP_FILES.values())

    print("Downloading minimal WildFake subset (COCO + DALL-E only)...")
    print(f"Patterns: {allow_patterns}")

    downloaded_dir = dataset_snapshot_download(
        dataset_id=DATASET_ID,
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
    )

    print(f"Downloaded to: {downloaded_dir}")
    return Path(downloaded_dir)


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"Already extracted: {extract_to}")
        return extract_to

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    return extract_to


def _find_filename_column(df: pd.DataFrame) -> str:
    for col in FILENAME_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Couldn't find a filename column in {list(df.columns)}. "
        f"Update FILENAME_COLUMNS with the correct column name."
    )


def _load_labeled_filenames(csv_path: Path, expected_label: int) -> list:
    df = pd.read_csv(csv_path)
    fname_col = _find_filename_column(df)

    if "IsFake" in df.columns:
        df = df[df["IsFake"].astype(int) == expected_label]

    return df[fname_col].astype(str).tolist()


def _resolve_image_path(images_root: Path, filename: str) -> Path:
    
    direct = images_root / filename
    if direct.exists():
        return direct

    basename = Path(filename).name
    matches = list(images_root.rglob(basename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not locate image for '{filename}' under {images_root}")


def load_balanced_wildfake_subset(
    samples_per_class: int = 100,
    seed: int = 42,
    local_dir: str = "data/wildfake_raw",
) -> list:
    """
    Returns a list of dicts: {"image_path": str, "label": int}
    """
    root = download_minimal_files(local_dir)

    dalle_zip = root / ZIP_FILES["dalle"]
    coco_zip = root / ZIP_FILES["coco"]

    dalle_images_dir = extract_zip(dalle_zip, root / "extracted" / "dalle")
    coco_images_dir = extract_zip(coco_zip, root / "extracted" / "coco")

    dalle2_names = _load_labeled_filenames(root / CSV_FILES["dalle2"], expected_label=1)
    dalle3_names = _load_labeled_filenames(root / CSV_FILES["dalle3"], expected_label=1)
    coco_names = _load_labeled_filenames(root / CSV_FILES["real_coco"], expected_label=0)

    rng = random.Random(seed)
    rng.shuffle(dalle2_names)
    rng.shuffle(dalle3_names)
    rng.shuffle(coco_names)

    # Split the AI-generated quota across dalle2 + dalle3 (roughly evenly)
    half = samples_per_class // 2
    dalle_names = dalle2_names[:half] + dalle3_names[: samples_per_class - half]

    ai_samples = []
    for name in dalle_names:
        try:
            path = _resolve_image_path(dalle_images_dir, name)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        ai_samples.append({"image_path": str(path), "label": 1})
        if len(ai_samples) >= samples_per_class:
            break

    real_samples = []
    for name in coco_names:
        try:
            path = _resolve_image_path(coco_images_dir, name)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        real_samples.append({"image_path": str(path), "label": 0})
        if len(real_samples) >= samples_per_class:
            break

    print(f"Final subset: {len(real_samples)} real (COCO) | {len(ai_samples)} AI (DALL-E)")

    if len(real_samples) < samples_per_class:
        print(f"Warning: Requested {samples_per_class} real images but only found {len(real_samples)}.")
    if len(ai_samples) < samples_per_class:
        print(f"Warning: Requested {samples_per_class} AI images but only found {len(ai_samples)}.")

    samples = real_samples + ai_samples
    rng.shuffle(samples)
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and cache only the COCO/DALL-E WildFake subset."
    )
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-dir", default="data/wildfake_raw")
    args = parser.parse_args()

    load_balanced_wildfake_subset(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        local_dir=args.local_dir,
    )