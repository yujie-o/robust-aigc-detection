import argparse
import random
from PIL import Image
from modelscope.msdatasets import MsDataset
from torch.utils.data import Dataset

DATASET_NAME = "hy2628982280/WildFake"


class WildFakeSubset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        image_data = item.get("image") or item.get("Image_path")
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data.convert("RGB")

        label = int(item["IsFake"])

        return image, label


def load_balanced_wildfake_subset(
    samples_per_class=100,
    seed=42,
    target_real_arch=None,
    target_ai_arch=None,
):
    """
    Download a balanced WildFake subset via ModelScope, filtered to only
    the requested architectures (e.g. COCO for real, DALL-E for AI-generated).

    Labels:
        IsFake = 0 (Real)
        IsFake = 1 (AI-generated)

    Args:
        samples_per_class: Target number of images for each class.
        seed: Random seed for shuffling.
        target_real_arch: Filter specific real source (e.g. "COCO").
        target_ai_arch: Filter specific generator (e.g. "DALL E Advanced").

    Returns:
        Equal numbers of label 0 and label 1 samples.
    """
    print(f"Downloading {DATASET_NAME} via ModelScope...")

    #Dataset Download
    ds = MsDataset.load(DATASET_NAME, subset_name='default', split='train')
    #You can configure subset_name and split as needed, refer to the "Quick Use" sample code

    real_samples = []
    ai_samples = []

    for item in ds:
        raw_is_fake = str(item.get("IsFake", "")).strip()
        if not raw_is_fake.isdigit():
            continue

        is_fake = int(raw_is_fake)
        arch = str(item.get("Architecture", "")).strip()

        # Only keep rows matching the requested architectures (COCO / DALL-E)
        if is_fake == 0:
            if target_real_arch is not None and target_real_arch.lower() not in arch.lower():
                continue
            if len(real_samples) < samples_per_class:
                real_samples.append(item)

        elif is_fake == 1:
            if target_ai_arch is not None and target_ai_arch.lower() not in arch.lower():
                continue
            if len(ai_samples) < samples_per_class:
                ai_samples.append(item)

        total_collected = len(real_samples) + len(ai_samples)
        if total_collected > 0 and total_collected % 50 == 0:
            print(
                f"\rCollected: {len(real_samples)} real | {len(ai_samples)} AI",
                end="",
            )

        if len(real_samples) >= samples_per_class and len(ai_samples) >= samples_per_class:
            break

    print()
    print(f"Final subset: {len(real_samples)} real | {len(ai_samples)} AI")

    if len(real_samples) < samples_per_class:
        print(f"Warning: Requested {samples_per_class} real images but only found {len(real_samples)}.")

    if len(ai_samples) < samples_per_class:
        print(f"Warning: Requested {samples_per_class} AI images but only found {len(ai_samples)}.")

    samples = real_samples + ai_samples

    rng = random.Random(seed)
    rng.shuffle(samples)

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and cache the WildFake dataset (COCO/DALL-E subset)."
    )
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-real-arch", default="COCO")
    parser.add_argument("--target-ai-arch", default="DALL E Advanced")
    args = parser.parse_args()

    load_balanced_wildfake_subset(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        target_real_arch=args.target_real_arch,
        target_ai_arch=args.target_ai_arch,
    )