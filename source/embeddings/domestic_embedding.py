import os
import torch
import torchaudio
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold
from msclap import CLAP  # 确保你安装了 msclap

# =========================
# Paths
# =========================
DATASET_ROOT = "datasets/few_shot_vfy_zenodo"
PATTERN_DIR = os.path.join(DATASET_ROOT, "pattern_sounds")
UNWANTED_DIR = os.path.join(DATASET_ROOT, "unwanted_sounds")

SAVE_DIR = "embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

X_PATH = os.path.join(SAVE_DIR, "domestic_clap_embeddings.pt")
Y_PATH = os.path.join(SAVE_DIR, "domestic_labels.pt")
FOLDS_PATH = os.path.join(SAVE_DIR, "domestic_folds.pt")
CSV_PATH = os.path.join(SAVE_DIR, "domestic_5fold.csv")

# =========================
# Load CLAP model
# =========================
clap_model = CLAP(version='2023', use_cuda=False)

# =========================
# Gather all wav files
# =========================
wav_files = []
labels = []

for root_dir in [PATTERN_DIR, UNWANTED_DIR]:
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname.endswith(".wav"):
                wav_files.append(os.path.join(class_path, fname))
                labels.append(class_name)

# 映射 label -> id
labels_set = sorted(list(set(labels)))
label2id = {l: i for i, l in enumerate(labels_set)}
y_ids = [label2id[l] for l in labels]

print(f"Found {len(wav_files)} audio files from {len(labels_set)} classes")

# =========================
# Generate CLAP embeddings
# =========================
X_list = []

for wav_path in tqdm(wav_files, desc="Embedding audio"):
    waveform, sr = torchaudio.load(wav_path)  # waveform shape [C, T]

    embedding = clap_model.get_audio_embeddings([wav_path], resample=True)[0]  # 返回 [D]
    X_list.append(embedding)

X = torch.stack(X_list)
y = torch.tensor(y_ids)

print("Embedding done. X shape:", X.shape, "y shape:", y.shape)

# =========================
# Generate 5-fold splits
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = torch.zeros(len(y), dtype=torch.long)

for fold_idx, (_, test_idx) in enumerate(kf.split(X)):
    folds[test_idx] = fold_idx + 1  # fold 从 1 开始

print("Fold assignment done:", folds.unique())

# =========================
# Save .pt files
# =========================
torch.save(X, X_PATH)
torch.save(y, Y_PATH)
torch.save(folds, FOLDS_PATH)
print(f"Saved embeddings to {SAVE_DIR}")

# =========================
# Generate CSV for reference
# =========================
df = pd.DataFrame({
    "filename": [os.path.relpath(f, DATASET_ROOT) for f in wav_files],
    "label": labels,
    "fold": folds.tolist()
})
df.to_csv(CSV_PATH, index=False)
print(f"CSV saved to {CSV_PATH}")