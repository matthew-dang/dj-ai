import os
import json
import random
from tqdm import tqdm

# --- Configuration ---
BPM_TOLERANCE = 5
METADATA_PATH = "../metadata/beatport_metadata.json"

# Normalize enharmonic key equivalents
def normalize_key(raw_key):
    key_line = raw_key.strip().split("\n")[0]  # remove date & price
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

# Check compatibility
def are_compatible(f1, f2):
    bpm_match = abs(f1['bpm'] - f2['bpm']) <= BPM_TOLERANCE
    key_match = normalize_key(f1['key']) == normalize_key(f2['key'])
    return bpm_match and key_match

# Load metadata
print("Loading Beatport metadata...")
with open(METADATA_PATH) as f:
    features = json.load(f)

print("Generating remix pairs...")
files = list(features.keys())
positive_pairs = []
negative_pairs = []

for i in range(len(files)):
    for j in range(i + 1, len(files)):
        file1, file2 = files[i], files[j]
        f1, f2 = features[file1], features[file2]

        if are_compatible(f1, f2):
            positive_pairs.append([file1, file2])
        else:
            negative_pairs.append([file1, file2])

negative_pairs = random.sample(negative_pairs, min(len(positive_pairs), len(negative_pairs)))

with open("../metadata/positive_pairs.json", "w") as f:
    json.dump(positive_pairs, f)

with open("../metadata/negative_pairs.json", "w") as f:
    json.dump(negative_pairs, f)

print(f"\nSaved {len(positive_pairs)} positive and {len(negative_pairs)} negative remix pairs.")
