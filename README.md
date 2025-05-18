DJ Remix Compatibility Recommender

An AI-powered recommendation system for DJs that suggests remix-compatible tracks based on audio similarity and musical metadata (BPM, key).

Features

Trains a Siamese neural network using contrastive loss and mel spectrogram embeddings

Recommends EDM tracks (dubstep for now) that are remixable based on learned audio similarity

Filters output using Beatport-scraped BPM and key data for musical accuracy

Automatically generates positive/negative training pairs from over 480 curated SoundCloud remixes

Supports fast training using cached mel spectrograms and GPU acceleration

Demo (CLI)

python recommend.py
# → Enter path to your query .mp3 or .wav file

Returns top 5 remix-compatible tracks

Filters candidates by compatible BPM/key

Displays similarity distance between embeddings

How It Works

Training Pipeline

Extracts 30-second mel spectrograms from remix tracks

Generates training pairs based on BPM/key compatibility from Beatport

Trains a PyTorch Siamese network to minimize contrastive loss:

Positive pair → closer in embedding space

Negative pair → pushed apart

Inference Pipeline

Takes a query track and computes its audio embedding

Compares it to database embeddings using Euclidean distance

Filters top matches based on Beatport metadata

🛠 Tech Stack

Python 3.9+

PyTorch for model training

librosa for audio processing

Selenium for Beatport scraping

NumPy, tqdm, pydub

Results

Trained on ~6,000 positive/negative remix pairs

Reduced contrastive loss from 13,822 ➜ 12.6

Recommendations respect musical structure and remix potential

To Do / Future Work

Add t-SNE / PCA visualization of learned audio embeddings

Build a Streamlit or Flask web UI for upload-based recommendations

Use energy scores or loudness metadata to enhance vibe matching

Expand dataset to include 1k+ remixes across EDM subgenres

Author

Matthew DangDJ and Computer Science Student @ UC Irvine"Combining AI and music to remix the future."

Feel free to reach out or remix this project yourself!