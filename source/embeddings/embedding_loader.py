import torch

EMBEDDINGS = {
    "clap": {
        "X": "embeddings/esc50_clap_embeddings.pt",
        "y": "embeddings/esc50_labels.pt",
        "folds": "embeddings/esc50_folds.pt",
    },
    "audiotovector": {
        "X": "embeddings/esc50_audiotovector_embeddings.pt",
        "y": "embeddings/esc50_labels.pt",
        "folds": "embeddings/esc50_folds.pt",
    }
}

def load_embedding(name):

    cfg = EMBEDDINGS[name]

    X = torch.load(cfg["X"])
    y = torch.load(cfg["y"])
    folds = torch.load(cfg["folds"])

    assert len(X) == len(y) == len(folds)

    if isinstance(X, torch.Tensor):
        print("X shape:", X.shape)
    else:
        print("X length:", len(X))

    if isinstance(y, torch.Tensor):
        print("y shape:", y.shape)
    else:
        print("y length:", len(y))

    return X, y, folds