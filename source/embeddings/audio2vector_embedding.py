"""
Extract AudioVector embeddings for the entire ESC-50 dataset in the same order as CLAP.

Output:
    esc50_audiotovector_embeddings.pt -> Tensor [N, D]
    esc50_labels.pt                   -> Tensor [N]  (same as CLAP)
    esc50_folds.pt                    -> Tensor [N]  (same as CLAP)
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from audio2vec import Audio2Vec
import soundfile as sf

# =========================
# Configuration
# =========================
ESC50_ROOT = "datasets/ESC-50-master"  # change to your path
AUDIO_DIR = os.path.join(ESC50_ROOT, "audio")
META_FILE = os.path.join(ESC50_ROOT, "meta", "esc50.csv")

OUTPUT_DIR = "embeddings"
BATCH_SIZE = 32  # optional batching
USE_CUDA = torch.cuda.is_available()

# =========================
# Load metadata
# =========================
def load_metadata():
    meta = pd.read_csv(META_FILE)

    file_paths = []
    labels = []
    folds = []

    for _, row in meta.iterrows():
        file_paths.append(os.path.join(AUDIO_DIR, row["filename"]))
        labels.append(row["target"])
        folds.append(row["fold"])

    return file_paths, labels, folds

# =========================
# Extract embeddings
# =========================
def extract_embeddings(file_paths):
    embedder = Audio2Vec()
    all_embeddings = []

    for i in tqdm(range(0, len(file_paths), BATCH_SIZE)):
        batch_paths = file_paths[i:i+BATCH_SIZE]
        batch_embeddings = []

        for path in batch_paths:
            audio, sr = sf.read(path)
            emb = embedder.audio2VectorProcessor(path)  # np.array
            # convert sparse to dense if needed
            if hasattr(emb, "toarray"):
                emb = emb.toarray().squeeze()
            emb_tensor = torch.tensor(emb, dtype=torch.float32)
            batch_embeddings.append(emb_tensor)

        batch_tensor = torch.stack(batch_embeddings)
        all_embeddings.append(batch_tensor)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading metadata...")
    file_paths, labels, folds = load_metadata()
    print(f"Total samples: {len(file_paths)}")

    print("Extracting AudioVector embeddings...")
    embeddings = extract_embeddings(file_paths)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    folds_tensor = torch.tensor(folds, dtype=torch.long)

    print("Embedding shape:", embeddings.shape)

    torch.save(embeddings, os.path.join(OUTPUT_DIR, "esc50_audiotovector_embeddings.pt"))
    # torch.save(labels_tensor, os.path.join(OUTPUT_DIR, "esc50_labels.pt"))
    # torch.save(folds_tensor, os.path.join(OUTPUT_DIR, "esc50_folds.pt"))

    print("Saved successfully!")

if __name__ == "__main__":
    main()