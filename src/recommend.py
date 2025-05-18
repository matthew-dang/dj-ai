# recommend.py
import os
import torch
import numpy as np
import librosa
import torch.nn.functional as F
import json
from train_model import SiameseNetwork
from metadata.beatport_scraper import get_single_track_metadata

# Config
MEL_CACHE = "mel_cache"
MODEL_PATH = "../models/siamese_model.pth"
METADATA_PATH = "../metadata/beatport_metadata.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BPM_TOLERANCE = 5

# Load model
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Preprocess query
def get_mel_from_path(path):
    y, sr = librosa.load(path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=512, axis=1)
    return torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()

# Load embeddings
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

# Recommend
def recommend(query_path, remix_embeddings, metadata, top_k=5):
    query_mel = get_mel_from_path(query_path).to(DEVICE)
    query_name = os.path.splitext(os.path.basename(query_path))[0]

    query_info = get_single_track_metadata(query_name)
    if not query_info:
        print("Could not retrieve metadata from Beatport for query track.")
        return []

    query_bpm = query_info['bpm']
    query_key = normalize_key(query_info['key'])

    with torch.no_grad():
        query_emb = model.forward_once(query_mel).squeeze()

    results = []
    for fname, emb in remix_embeddings.items():
        base = fname.replace(".npy", "")
        if base == query_name or base not in metadata:
            continue

        bpm = metadata[base].get("bpm")
        key = normalize_key(metadata[base].get("key", ""))

        if key != query_key or abs(bpm - query_bpm) > BPM_TOLERANCE:
            continue

        emb_tensor = torch.tensor(emb).to(query_emb.device)
        dist = F.pairwise_distance(query_emb.unsqueeze(0), emb_tensor.unsqueeze(0), p=2)
        results.append((fname, dist.item()))

    results.sort(key=lambda x: x[1])
    return results[:top_k]


def normalize_key(raw_key):
    key_line = raw_key.strip().split("\n")[0]
    enharmonics = {
        "A#": "Bb", "Bb": "Bb",
        "C#": "Db", "Db": "Db",
        "D#": "Eb", "Eb": "Eb",
        "F#": "Gb", "Gb": "Gb",
        "G#": "Ab", "Ab": "Ab"
    }
    parts = key_line.split()
    if len(parts) == 2:
        note, mode = parts[0], parts[1]
        note = enharmonics.get(note.replace("♯", "#"), note)
        return f"{note} {mode}"
    return key_line

# Run
if __name__ == '__main__':
    query_file = input("Enter path to your query .mp3 or .wav file: ").strip()
    print("Loading database remix embeddings...")
    remix_db = load_database_embeddings(model)
    print(f"Loaded {len(remix_db)} remix embeddings.")

    print("Loading Beatport metadata for query...")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print("Finding top remix-compatible tracks...")
    matches = recommend(query_file, remix_db, metadata, top_k=5)

    print("\nTop Matches:")
    for fname, dist in matches:
        print(f"{fname.replace('.npy', '')} — distance: {dist:.4f}")
