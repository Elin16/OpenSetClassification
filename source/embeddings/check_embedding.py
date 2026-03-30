import torch

# =========================
# Load embeddings
# =========================
CLAP_DIR = "embeddings"
AUDIOVECTOR_DIR = "embeddings"

clap_X = torch.load(f"{CLAP_DIR}/esc50_clap_embeddings.pt")
clap_y = torch.load(f"{CLAP_DIR}/esc50_labels.pt")
clap_folds = torch.load(f"{CLAP_DIR}/esc50_folds.pt")

av_X = torch.load(f"{AUDIOVECTOR_DIR}/esc50_audiotovector_embeddings.pt")
av_y = torch.load(f"{AUDIOVECTOR_DIR}/esc50_labels.pt")
av_folds = torch.load(f"{AUDIOVECTOR_DIR}/esc50_folds.pt")

# =========================
# Check shapes
# =========================
print("CLAP X shape:", clap_X.shape)
print("AudioVector X shape:", av_X.shape)

assert clap_y.shape == av_y.shape, "Labels shape mismatch!"
assert clap_folds.shape == av_folds.shape, "Folds shape mismatch!"
assert clap_X.shape[0] == av_X.shape[0], "Number of samples mismatch!"

# =========================
# Check labels and folds
# =========================
labels_match = torch.all(clap_y == av_y)
folds_match = torch.all(clap_folds == av_folds)

print(f"Labels match: {labels_match}")
print(f"Folds match: {folds_match}")

if labels_match and folds_match:
    print("SUCCESS: Sample order is consistent between CLAP and AudioVector embeddings!")
else:
    # 找出不匹配的索引
    mismatch_indices = torch.where((clap_y != av_y) | (clap_folds != av_folds))[0]
    print(f"Mismatch at indices: {mismatch_indices}")