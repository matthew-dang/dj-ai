# recommend.py
import os
import torch
import numpy as np
import librosa
import torch.nn.functional as F
from train_model import SiameseNetwork  # assumes model class is in siamese_model.py

# Config
MEL_CACHE = "mel_cache"
MODEL_PATH = "siamese_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Mode
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Preprocess Query Track
def get_mel_from_path(path):
    y, sr = librosa.load(path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=512, axis=1)
    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()

# Load Remix Embeddings
def load_database_embeddings(model):
    remix_embeddings = {}
    for fname in os.listdir(MEL_CACHE):
        if not fname.endswith(".npy"):
            continue
        mel = np.load(os.path.join(MEL_CACHE, fname))
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            emb = model.forward_once(mel_tensor).squeeze().cpu().numpy()
        remix_embeddings[fname] = emb
    return remix_embeddings

# Recommend Matches
def recommend(query_path, remix_embeddings, top_k=5):
    query_mel = get_mel_from_path(query_path).to(DEVICE)
    with torch.no_grad():
        query_emb = model.forward_once(query_mel).squeeze()

    results = []
    for fname, emb in remix_embeddings.items():
        emb_tensor = torch.tensor(emb).to(query_emb.device)
        dist = F.pairwise_distance(query_emb.unsqueeze(0), emb_tensor.unsqueeze(0), p=2)
        results.append((fname, dist.item()))

    results.sort(key=lambda x: x[1])
    return results[:top_k]

# Run 
if __name__ == '__main__':
    query_file = input("Enter path to your query .mp3 or .wav file: ").strip()
    print("Loading database remix embeddings")
    remix_db = load_database_embeddings(model)
    print(f"Loaded {len(remix_db)} remix embeddings.")

    print("Finding top remix-compatible tracks...")
    matches = recommend(query_file, remix_db, top_k=5)

    print("\nTop Matches:")
    for fname, dist in matches:
        print(f"{fname.replace('.npy', '')} — distance: {dist:.4f}")
