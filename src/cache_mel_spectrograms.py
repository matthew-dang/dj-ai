import os
import librosa
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment

DATA_FOLDER = "../data"
CACHE_FOLDER = "mel_cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    if not os.path.exists(wav_path):
        try:
            audio = AudioSegment.from_file(mp3_path, format="mp3")
            audio.export(wav_path, format="wav")
        except Exception as e:
            print(f"Error converting {mp3_path}: {e}")
            return None
    return wav_path

def compute_and_cache_mel(file_path):
    if file_path.endswith(".mp3"):
        file_path = convert_mp3_to_wav(file_path)
    if not file_path:
        return

    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=512, axis=1)

        base_name = os.path.basename(file_path).replace(".wav", "").replace(".mp3", "")
        out_path = os.path.join(CACHE_FOLDER, base_name + ".npy")
        np.save(out_path, mel_db)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    print("Caching mel spectrograms...")
    for fname in tqdm(os.listdir(DATA_FOLDER)):
        if fname.endswith(".mp3") or fname.endswith(".wav"):
            full_path = os.path.join(DATA_FOLDER, fname)
            compute_and_cache_mel(full_path)

    print("Done caching mel spectrograms to mel_cache/")
