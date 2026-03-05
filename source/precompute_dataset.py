import os
import numpy as np
import torch
from tqdm import tqdm
from msclap import CLAP
from esc50_dataset import ESC50

# ==========================
# Config
# ==========================
ROOT_PATH = "datasets"
SAVE_DIR = "feature_bank/esc50"
USE_CUDA = False

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================
# Load dataset
# ==========================
dataset = ESC50(root=ROOT_PATH, download=False)

print(f"Total samples: {len(dataset)}")
print(f"Number of classes: {len(dataset.classes)}")

# ==========================
# Load CLAP
# ==========================
print("Loading CLAP...")
clap_model = CLAP(version="2023", use_cuda=USE_CUDA)

# ==========================
# Precompute embeddings
# ==========================
all_embeddings = []
all_class_indices = []

print("Precomputing audio embeddings...")

for i in tqdm(range(len(dataset))):
    x, class_idx, _ = dataset[i]

    with torch.no_grad():
        audio_embedding = clap_model.get_audio_embeddings([x], resample=True)

    all_embeddings.append(audio_embedding.cpu().numpy())
    all_class_indices.append(class_idx)

# ==========================
# Convert to numpy
# ==========================
all_embeddings = np.concatenate(all_embeddings, axis=0)
all_class_indices = np.array(all_class_indices)

print("Embedding shape:", all_embeddings.shape)
print("Label shape:", all_class_indices.shape)

# ==========================
# Save
# ==========================
np.save(os.path.join(SAVE_DIR, "audio_embeddings.npy"), all_embeddings)
np.save(os.path.join(SAVE_DIR, "class_indices.npy"), all_class_indices)
np.save(os.path.join(SAVE_DIR, "class_names.npy"), np.array(dataset.classes))

print("Saved to:", SAVE_DIR)