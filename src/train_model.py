import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np

# Dataset
class RemixPairDataset(Dataset):
    def __init__(self, pairs, label, mel_cache_dir):
        self.pairs = pairs
        self.label = label
        self.mel_cache_dir = mel_cache_dir

    def __len__(self):
        return len(self.pairs)

    def sanitize_filename(self, fname):
        return fname.strip().replace(".mp3", "").replace(".wav", "").replace("\u200b", "")

    def __getitem__(self, idx):
        fname1, fname2 = self.pairs[idx]
        print(f"Loading pair: {fname1}, {fname2}")
        base1 = self.sanitize_filename(fname1)
        base2 = self.sanitize_filename(fname2)
        path1 = os.path.join(self.mel_cache_dir, base1 + ".npy")
        path2 = os.path.join(self.mel_cache_dir, base2 + ".npy")

        if not os.path.exists(path1):
            raise FileNotFoundError(f"Missing: {path1}")
        if not os.path.exists(path2):
            raise FileNotFoundError(f"Missing: {path2}")

        mel1 = np.load(path1)
        mel2 = np.load(path2)

        return torch.tensor(mel1).unsqueeze(0), torch.tensor(mel2).unsqueeze(0), torch.tensor(self.label, dtype=torch.float32)

# Model
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 30 * 126, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        loss = (1 - label) * dist.pow(2) + label * F.relu(self.margin - dist).pow(2)
        return loss.mean()

# Training
def load_datasets(pos_json, neg_json, mel_cache_dir):
    with open(pos_json) as f:
        pos_pairs = json.load(f)
    with open(neg_json) as f:
        neg_pairs = json.load(f)

    pos_dataset = RemixPairDataset(pos_pairs, label=0, mel_cache_dir=mel_cache_dir)
    neg_dataset = RemixPairDataset(neg_pairs, label=1, mel_cache_dir=mel_cache_dir)
    return ConcatDataset([pos_dataset, neg_dataset])

def train_model(dataset, model, loss_fn, optimizer, device, epochs=10, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, label in loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out1, out2 = model(x1, x2)
            loss = loss_fn(out1, out2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {total_loss:.4f}")

if __name__ == '__main__':
    MEL_CACHE = "mel_cache"
    POS_JSON = "../metadata/positive_pairs.json"
    NEG_JSON = "../metadata/negative_pairs.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading datasets")
    dataset = load_datasets(POS_JSON, NEG_JSON, MEL_CACHE)

    print("Initializing model")
    model = SiameseNetwork().to(device)
    loss_fn = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training")
    train_model(dataset, model, loss_fn, optimizer, device, epochs=10, batch_size=16)

    print("Training complete. Saving model")
    torch.save(model.state_dict(), "siamese_model.pth")
