import random

from datasets import load_dataset
from torch.utils.data import Dataset


DATASET_NAME = "saberzl/SID_Set"


class SIDSubset(Dataset):
    """
    Simple PyTorch dataset containing selected SID_Set samples.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        image = item["image"].convert("RGB")
        label = int(item["label"])

        return image, label


def load_balanced_sid_subset(
    samples_per_class=100,
    seed=42,
    shuffle_buffer=1000,
):
    """
    Stream a balanced subset of SID_Set.

    Labels:
        0 = Real
        1 = Fully AI-generated
        2 = Tampered (excluded)

    Returns:
        Equal numbers of label 0 and label 1 samples.
    """

    print("Streaming SID_Set...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Shuffle the streaming dataset using a bounded buffer.
    # This gives us a less order-dependent sample without
    # loading the entire dataset into memory.
    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer,
    )

    real_samples = []
    ai_samples = []

    for item in dataset:
        label = int(item["label"])

        # Real
        if (
            label == 0
            and len(real_samples) < samples_per_class
        ):
            real_samples.append(item)

        # Fully AI-generated
        elif (
            label == 1
            and len(ai_samples) < samples_per_class
        ):
            ai_samples.append(item)

        # Print progress occasionally
        total_collected = (
            len(real_samples)
            + len(ai_samples)
        )

        if (
            total_collected > 0
            and total_collected % 100 == 0
        ):
            print(
                f"\rCollected: "
                f"{len(real_samples)} real | "
                f"{len(ai_samples)} AI",
                end="",
            )

        # Stop once both classes are full
        if (
            len(real_samples) >= samples_per_class
            and len(ai_samples) >= samples_per_class
        ):
            break

    print()

    print(
        f"Final subset: "
        f"{len(real_samples)} real | "
        f"{len(ai_samples)} AI"
    )

    if len(real_samples) < samples_per_class:
        raise RuntimeError(
            f"Requested {samples_per_class} real images "
            f"but only collected {len(real_samples)}."
        )

    if len(ai_samples) < samples_per_class:
        raise RuntimeError(
            f"Requested {samples_per_class} AI images "
            f"but only collected {len(ai_samples)}."
        )

    samples = real_samples + ai_samples

    # Shuffle the final balanced subset reproducibly
    rng = random.Random(seed)
    rng.shuffle(samples)

    return samples