import os
import librosa
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

DATA_FOLDER = "../data"
MEL_FOLDER = "mel_cache"
os.makedirs(MEL_FOLDER, exist_ok=True)


def convert_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):
        try:
            audio = AudioSegment.from_file(mp3_path, format="mp3")
            audio.export(wav_path, format="wav")
        except Exception as e:
            print(f"Error converting {mp3_path} to WAV: {e}")
            return None
    return wav_path


def extract_mel(path):
    y, sr = librosa.load(path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=512, axis=1)
    return mel_db


print("Preprocessing remixes to mel spectrograms...")
for fname in tqdm(os.listdir(DATA_FOLDER)):
    if not fname.endswith(".mp3") and not fname.endswith(".wav"):
        continue

    base = os.path.splitext(fname)[0]
    mel_path = os.path.join(MEL_FOLDER, base + ".npy")
    if os.path.exists(mel_path):
        continue

    try:
        path = os.path.join(DATA_FOLDER, fname)
        if fname.endswith(".mp3"):
            path = convert_to_wav(path)
        if path:
            mel = extract_mel(path)
            np.save(mel_path, mel)
    except Exception as e:
        print(f"Error on {fname}: {e}")
