import os
import librosa
import numpy as np
import json
import random
from tqdm import tqdm
from pydub import AudioSegment

# --- Configuration ---
REMIX_FOLDER = "../data"
BPM_TOLERANCE = 5
CLEAN_UP_WAV = True 

def convert_mp3_to_wav(mp3_path):
    try:
        wav_path = mp3_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            sound = AudioSegment.from_file(mp3_path, format="mp3")
            sound.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting {mp3_path} to WAV: {e}")
        return None

def extract_bpm_and_key(filepath):
    try:
        y, sr = librosa.load(filepath, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_index = np.argmax(chroma_mean)
        key = librosa.midi_to_note(24 + key_index)

        return tempo, key
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None

def are_compatible(f1, f2):
    bpm_match = abs(f1['bpm'] - f2['bpm']) <= BPM_TOLERANCE
    key_match = f1['key'] == f2['key']
    return bpm_match and key_match

features = {}
print("Converting and analyzing audio files...")
for filename in tqdm(os.listdir(REMIX_FOLDER)):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(REMIX_FOLDER, filename)
        wav_path = convert_mp3_to_wav(mp3_path)
        if wav_path:
            bpm, key = extract_bpm_and_key(wav_path)
            if bpm is not None and key is not None:
                features[filename] = {'bpm': bpm, 'key': key}

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

with open("positive_pairs.json", "w") as f:
    json.dump(positive_pairs, f)

with open("negative_pairs.json", "w") as f:
    json.dump(negative_pairs, f)

print(f"\nSaved {len(positive_pairs)} positive and {len(negative_pairs)} negative remix pairs.")

if CLEAN_UP_WAV:
    print("Cleaning up intermediate WAV files")
    for file in os.listdir(REMIX_FOLDER):
        if file.endswith(".wav"):
            os.remove(os.path.join(REMIX_FOLDER, file))