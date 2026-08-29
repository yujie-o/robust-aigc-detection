import random
import re
from PIL import Image
from modelscope.msdatasets import MsDataset
from torch.utils.data import Dataset

DATASET_NAME = "hy2628982280/WildFake"


class WildFakeSubset(Dataset):
    """
    Simple PyTorch dataset containing selected WildFake samples.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        # Handle PIL Image or image file path if loaded as a string
        image_data = item.get("image") or item.get("Image_path")
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data.convert("RGB")

        # Map IsFake column: 0 = Real, 1 = Fake
        label = int(item["IsFake"])

        return image, label


def load_balanced_wildfake_subset(
    samples_per_class=100,
    seed=42,
    target_real_arch=None,  
    target_ai_arch=None,    
):
    """
    Stream a balanced subset of WildFake using ModelScope.

    Labels:
        IsFake = 0 (Real)
        IsFake = 1 (AI-generated)

    Args:
        samples_per_class: Target number of images for each class.
        seed: Random seed for shuffling.
        target_real_arch: Filter specific real source (e.g. for organiser validation).
        target_ai_arch: Filter specific generator (e.g. for organiser validation).

    Returns:
        Equal numbers of label 0 and label 1 samples.
    """
    print(f"Streaming {DATASET_NAME} via ModelScope...")

    # Load streaming dataset from ModelScope
    dataset = MsDataset.load(
        DATASET_NAME,
        subset_name="default",
        split="train",
        use_streaming=True,
    )

    real_samples = []
    ai_samples = []

    # Iterate through the streaming generator
    for item in dataset:
        is_fake = int(item.get("IsFake", 0))
        arch = str(item.get("Architecture", "")).strip()

        # Filter and collect Real images (IsFake == 0)
        if is_fake == 0 and len(real_samples) < samples_per_class:
            if target_real_arch is None or re.sub(r"[^a-z0-9]", "", target_real_arch.lower()) in re.sub(r"[^a-z0-9]", "", arch.lower()):
                real_samples.append(item)

        # Filter and collect AI images (IsFake == 1)
        elif is_fake == 1 and len(ai_samples) < samples_per_class:
            if target_ai_arch is None or re.sub(r"[^a-z0-9]", "", target_ai_arch.lower()) in re.sub(r"[^a-z0-9]", "", arch.lower()):
                ai_samples.append(item)

        # Progress reporting
        total_collected = len(real_samples) + len(ai_samples)
        if total_collected > 0 and total_collected % 50 == 0:
            print(
                f"\rCollected: {len(real_samples)} real | {len(ai_samples)} AI",
                end="",
            )

        # Stop early once quota is satisfied
        if len(real_samples) >= samples_per_class and len(ai_samples) >= samples_per_class:
            break

    print()
    print(f"Final subset: {len(real_samples)} real | {len(ai_samples)} AI")

    if len(real_samples) < samples_per_class:
        print(f"Warning: Requested {samples_per_class} real images but only found {len(real_samples)}.")

    if len(ai_samples) < samples_per_class:
        print(f"Warning: Requested {samples_per_class} AI images but only found {len(ai_samples)}.")

    samples = real_samples + ai_samples

    # Shuffle the final balanced subset reproducibly
    rng = random.Random(seed)
    rng.shuffle(samples)

    return samples