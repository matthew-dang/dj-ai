import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import librosa
import numpy as np
from pydub import AudioSegment

# Dataset 
class RemixPairDataset(Dataset):
    def __init__(self, pairs, label, root_dir):
        self.pairs = pairs
        self.label = label
        self.root_dir = root_dir

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fname1, fname2 = self.pairs[idx]
        print(f"Loading pair: {fname1}, {fname2}")
        path1 = os.path.join(self.root_dir, fname1)
        path2 = os.path.join(self.root_dir, fname2)

        mel1 = self.get_mel(path1)
        mel2 = self.get_mel(path2)

        return torch.tensor(mel1).unsqueeze(0), torch.tensor(mel2).unsqueeze(0), torch.tensor(self.label, dtype=torch.float32)

    def convert_to_wav(self, mp3_path):
        wav_path = mp3_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            try:
                audio = AudioSegment.from_file(mp3_path, format="mp3")
                audio.export(wav_path, format="wav")
            except Exception as e:
                print(f"Error converting {mp3_path}: {e}")
                return None
        return wav_path

    def get_mel(self, path):
        if path.endswith(".mp3"):
            path = self.convert_to_wav(path)
            if path is None or not os.path.exists(path):
                raise FileNotFoundError(f"Converted WAV not found for {path}")
        y, sr = librosa.load(path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=512, axis=1)
        return mel_db

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
def load_datasets(pos_json, neg_json, root_dir):
    with open(pos_json) as f:
        pos_pairs = json.load(f)
    with open(neg_json) as f:
        neg_pairs = json.load(f)

    pos_dataset = RemixPairDataset(pos_pairs, label=0, root_dir=root_dir)
    neg_dataset = RemixPairDataset(neg_pairs, label=1, root_dir=root_dir)
    return ConcatDataset([pos_dataset, neg_dataset])

def train_model(dataset, model, loss_fn, optimizer, device, epochs=10, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
    DATA_FOLDER = "../data"
    POS_JSON = "positive_pairs.json"
    NEG_JSON = "negative_pairs.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading datasets")
    dataset = load_datasets(POS_JSON, NEG_JSON, DATA_FOLDER)

    print("Initializing model")
    model = SiameseNetwork().to(device)
    loss_fn = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training")
    train_model(dataset, model, loss_fn, optimizer, device, epochs=10, batch_size=16)

    print("Training complete. Saving model")
    torch.save(model.state_dict(), "siamese_model.pth")