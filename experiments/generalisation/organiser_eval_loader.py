"""
Direct organiser eval loader for WildFake.

Reads dalle3.csv and real_coco.csv (both in this folder), filters for:
- Real: COCO val2017 images
- AI: dalle3 images with IsAdvanced=1

Images are read directly from the local coco.zip / DALLE.zip archives
(also in this folder) rather than downloaded from ModelScope. The CSV
`Image_path` values are prefixed with the dataset's own root folder, which
must be stripped to match the member names actually stored inside each zip:

    Real:   "./Real/..."            -> strip "./Real/"            -> matches coco.zip
    DALLE3: "./Diffusion_based/..." -> strip "./Diffusion_based/" -> matches DALLE.zip
"""

import csv
import io
import os
import random
import zipfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = <repo_root>/experiments/generalisation, so two levels up is the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
WILDFAKE_ZIPS_DIR = os.path.join(REPO_ROOT, "wildfake_zips", "Images")

COCO_CSV = os.path.join(HERE, "data", "real_coco.csv")
DALLE3_CSV = os.path.join(HERE, "data", "dalle3.csv")

COCO_ZIP = os.path.join(WILDFAKE_ZIPS_DIR, "Real", "coco.zip")
DALLE_ZIP = os.path.join(WILDFAKE_ZIPS_DIR, "Diffusion_based", "DALLE.zip")

COCO_PREFIX = "./Real/"
DALLE_PREFIX = "./Diffusion_based/"


def _strip_prefix(path, prefix):
    """Map a CSV Image_path to the member name stored inside the zip."""
    if path.startswith(prefix):
        return path[len(prefix):]
    # Already stripped / unexpected format - use as-is.
    return path.lstrip("./")


def _load_image_from_zip(zf, member_name):
    """Read a single image straight out of an open ZipFile."""
    with zf.open(member_name) as f:
        data = f.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_organiser_eval_samples(samples_per_class=200, seed=42):
    """
    Returns list of samples:
        [{"image": PIL.Image, "label": 0 or 1, "path": str}, ...]

    Real: samples_per_class from real_coco.csv (val2017 only)
    AI: samples_per_class from dalle3.csv where IsAdvanced=1

    Images are pulled directly from coco.zip / DALLE.zip - no network access.
    """
    rng = random.Random(seed)

    # Load COCO CSV (comma-delimited), restrict to val2017 split only
    print(f"Reading {COCO_CSV}...")
    with open(COCO_CSV, encoding="utf-8") as f:
        coco_rows = [
            r for r in csv.DictReader(f)
            if r.get("IsFake") == "0" and "val2017" in r.get("Image_path", "")
        ]
    print(f"COCO rows (val2017 only): {len(coco_rows)}")

    # Load DALLE3 CSV, filter IsAdvanced=1 (comma-delimited)
    print(f"Reading {DALLE3_CSV}...")
    with open(DALLE3_CSV, encoding="utf-8") as f:
        dalle3_rows = [
            r for r in csv.DictReader(f)
            if r.get("IsFake") == "1" and r.get("IsAdvanced") == "1"
        ]
    print(f"DALLE3 IsAdvanced=1 rows: {len(dalle3_rows)}")

    # Shuffle and sample
    rng.shuffle(coco_rows)
    rng.shuffle(dalle3_rows)
    coco_samples = coco_rows[:samples_per_class]
    dalle3_samples = dalle3_rows[:samples_per_class]

    print(f"\nLoading {len(coco_samples)} COCO + {len(dalle3_samples)} DALLE3 images "
          f"from local zips...")

    samples = []

    with zipfile.ZipFile(COCO_ZIP) as coco_zf:
        for i, row in enumerate(coco_samples):
            path = row["Image_path"]
            member_name = _strip_prefix(path, COCO_PREFIX)
            try:
                img = _load_image_from_zip(coco_zf, member_name)
                samples.append({"image": img, "label": 0, "path": path})
                if (i + 1) % 20 == 0:
                    print(f"  COCO: {i + 1}/{len(coco_samples)}")
            except KeyError:
                print(f"  Skip {path}: not found in coco.zip as '{member_name}'")
            except Exception as e:
                print(f"  Skip {path}: {e}")

    with zipfile.ZipFile(DALLE_ZIP) as dalle_zf:
        for i, row in enumerate(dalle3_samples):
            path = row["Image_path"]
            member_name = _strip_prefix(path, DALLE_PREFIX)
            try:
                img = _load_image_from_zip(dalle_zf, member_name)
                samples.append({"image": img, "label": 1, "path": path})
                if (i + 1) % 20 == 0:
                    print(f"  DALLE3: {i + 1}/{len(dalle3_samples)}")
            except KeyError:
                print(f"  Skip {path}: not found in DALLE.zip as '{member_name}'")
            except Exception as e:
                print(f"  Skip {path}: {e}")

    print(f"\nLoaded {len(samples)} total samples")
    print(f"  Real: {sum(1 for s in samples if s['label'] == 0)}")
    print(f"  AI: {sum(1 for s in samples if s['label'] == 1)}")

    return samples