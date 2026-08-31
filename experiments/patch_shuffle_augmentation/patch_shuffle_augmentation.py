import os
import sys
import random
import torch
import torch.nn.functional as nnf
import torchvision.transforms as T
from transformers import AutoProcessor
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from models.baseline import MODEL_NAME, AIGCDetector
from data.sid_dataset import SIDSubset, load_balanced_sid_subset

SEED = 42

SAMPLES_PER_CLASS = 500

VAL_RATIO = 0.2

BATCH_SIZE = 4

EPOCHS = 3

LEARNING_RATE = 1e-3

SHUFFLE_PROB = 0.5

CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "patch_shuffle_aug_best.pt")


def patch_shuffle(img: torch.Tensor, patch_size: int = 16):
    '''
    Args:
        img (Tensor): Input image Tensor of shape (C, H, W) or (B, C, H, W)
        patch_size (int): Size of patches to split input image into
    '''
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    patches = nnf.unfold(img, kernel_size=patch_size, padding=0, stride=patch_size)

    shuffled = []
    for i in range(img.shape[0]):
        num_patches = patches.shape[-1]
        new_order = torch.randperm(num_patches)
        shuffled.append(patches[i:i+1, :, new_order])

    shuffled_patches = torch.cat(shuffled, dim=0)

    out = nnf.fold(shuffled_patches, img.shape[-2:], kernel_size=patch_size, padding=0, stride=patch_size)
    return out


class PatchShuffleDataset(Dataset):
    def __init__(self, samples, apply_to_val: bool = False, shuffle_prob: float = 0.5):
        self.samples = samples
        self.apply_to_val = apply_to_val
        self.shuffle_prob = shuffle_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        img = item["image"].convert("RGB")

        # Only shuffle during training, and only with probability shuffle_prob
        if not self.apply_to_val and random.random() < self.shuffle_prob:
            tensor = T.PILToTensor()(img).float()
            shuffled = patch_shuffle(tensor)
            img = T.ToPILImage()(shuffled.squeeze(0).to(torch.uint8))

        label = int(item["label"])

        return img, label


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def split_samples(samples, val_ratio=0.2):
    real = [x for x in samples if int(x["label"]) == 0]
    ai = [x for x in samples if int(x["label"]) == 1]

    rng = random.Random(SEED)
    rng.shuffle(real)
    rng.shuffle(ai)

    val_real_count = int(len(real) * val_ratio)
    val_ai_count = int(len(ai) * val_ratio)

    val_samples = (real[:val_real_count] + ai[:val_ai_count])
    train_samples = (real[val_real_count:] + ai[val_ai_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return (train_samples, val_samples)


def make_collate_fn(processor):
    def collate_fn(batch):
        images, labels = zip(*batch)

        inputs = processor(images=list(images), return_tensors="pt")

        labels = torch.tensor(labels, dtype=torch.float32)

        return (inputs["pixel_values"], labels)

    return collate_fn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(pixel_values)

        loss = criterion(logits.squeeze(-1), labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for pixel_values, labels in loader:

            pixel_values = pixel_values.to(device)

            logits = model(pixel_values)

            probs = torch.sigmoid(logits.squeeze(-1))

            all_labels.extend(labels.tolist())

            all_probs.extend(probs.cpu().tolist())

    auc = roc_auc_score(all_labels, all_probs)

    return auc


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading processor...")
    processor = (AutoProcessor.from_pretrained(MODEL_NAME))

    print("Loading SID_Set subset...")
    samples = load_balanced_sid_subset(samples_per_class=SAMPLES_PER_CLASS, seed=SEED)

    train_samples, val_samples = split_samples(samples, VAL_RATIO)
    print("Train samples:", len(train_samples))
    print("Validation samples:", len(val_samples))

    train_dataset = PatchShuffleDataset(train_samples, shuffle_prob=SHUFFLE_PROB)
    val_dataset = PatchShuffleDataset(val_samples, apply_to_val=True)

    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("Loading baseline detector...")

    model = AIGCDetector(freeze_backbone=True).to(device)

    criterion = (torch.nn.BCEWithLogitsLoss())

    optimizer = torch.optim.AdamW(
        filter(
            lambda p:
            p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True,)

    best_auc = -1.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch " + f"{epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, val_loader, device)

        print(f"Train Loss: " + f"{train_loss:.4f}")
        print(f"Validation ROC-AUC: " + f"{val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "model_name": MODEL_NAME,
                },
                CHECKPOINT_PATH,
            )

            print("Saved checkpoint:", CHECKPOINT_PATH)

    print("\nTraining complete.")

    print(
        f"Best validation AUC: "
        f"{best_auc:.4f}"
    )


if __name__ == "__main__":
    main()