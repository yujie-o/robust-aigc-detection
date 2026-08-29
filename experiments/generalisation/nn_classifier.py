import os, sys, random
import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from data.sid_dataset import SIDSubset, load_balanced_sid_subset
from models.baseline import MODEL_NAME

SEED, SAMPLES_PER_CLASS, VAL_RATIO, BATCH_SIZE, TOP_K = 42, 1000, 0.2, 4, 5


def split_samples(samples, val_ratio=VAL_RATIO, seed=SEED):
    """Replicates src/train.py's split. Keep in sync."""
    real = [x for x in samples if int(x["label"]) == 0]
    ai = [x for x in samples if int(x["label"]) == 1]
    rng = random.Random(seed)
    rng.shuffle(real); rng.shuffle(ai)
    n_val_r, n_val_a = int(len(real) * val_ratio), int(len(ai) * val_ratio)
    val = real[:n_val_r] + ai[:n_val_a]
    train = real[n_val_r:] + ai[n_val_a:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def make_collate_fn(processor):
    def collate(batch):
        images, labels = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        return inputs["pixel_values"], torch.tensor(labels)
    return collate


@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    for pixel_values, batch_labels in loader:
        out = backbone.get_image_features(pixel_values=pixel_values.to(device))
        feats.append(F.normalize(out.pooler_output, p=2, dim=1).cpu())
        labels.extend(batch_labels.tolist())
    return torch.cat(feats), torch.tensor(labels)


def nn_classify(query, bank, bank_labels, k=TOP_K):
    """Cosine sim -> top-k -> P(AI) = fraction of AI neighbors."""
    sims = query @ bank.T
    _, topk = sims.topk(k, dim=1)
    return bank_labels[topk].float().mean(dim=1)


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for p in backbone.parameters(): p.requires_grad = False

    samples = load_balanced_sid_subset(samples_per_class=SAMPLES_PER_CLASS, seed=SEED)
    train_samples, val_samples = split_samples(samples)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    collate = make_collate_fn(processor)
    train_loader = DataLoader(SIDSubset(train_samples), batch_size=BATCH_SIZE, collate_fn=collate)
    val_loader = DataLoader(SIDSubset(val_samples), batch_size=BATCH_SIZE, collate_fn=collate)

    print("Building feature bank from train set...")
    bank, bank_labels = extract_features(backbone, train_loader, device)
    print("Extracting val features...")
    val_feats, val_labels = extract_features(backbone, val_loader, device)

    probs = nn_classify(val_feats, bank, bank_labels, k=TOP_K)
    auc = roc_auc_score(val_labels.tolist(), probs.tolist())
    print(f"\nP2-a NN Classifier val ROC-AUC (k={TOP_K}): {auc:.4f}")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "p2a_nn_results.txt"), "w") as f:
        f.write(f"P2-a NN Classifier\nk={TOP_K}\nBank size: {len(bank_labels)}\n"
                f"Val size: {len(val_labels)}\nVal ROC-AUC: {auc:.4f}\n")


if __name__ == "__main__":
    main()