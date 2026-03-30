DOME_BASIC_CONFIG = {

    "dataset": "domestic",

    "embeddings": [
        "clap"
    ],

    "classifiers": [
        "cosine",
        "euclidean",
        "mahalanobis",
        "linear",
        "energy",
        "openmax"
    ],
    "rejection_method": [
        "global_threshold",
        "class_threshold",
        "class_quantile"
    ],
    "num_known_list": [
        30,
        20,
        15,
        10
    ],

    "seeds": [0],

    "test_fold": 5
}


ESC50_BASIC_CONFIG = {

    "dataset": "esc50",

    "embeddings": [
        "clap",
        "audiotovector"
    ],

    "classifiers": [
        "cosine",
        "euclidean",
        "mahalanobis",
        "linear",
        "energy",
        "openmax"
    ],
    "rejection_method": [
        "global_threshold",
        "class_threshold",
        "class_percentile"
    ],
    "num_known_list": [
        40,
        30,
        20,
        15,
        10
    ],

    "seeds": [0],

    "test_fold": 5
}


BASIC_CONFIG = {

    "dataset": "esc50",

    "embeddings": [
        "clap",
        "audiotovector"
    ],

    "classifiers": [
        "cosine",
        "euclidean",
        "mahalanobis",
        "linear",
        "energy",
        "openmax"
    ],
    "rejection_method": [
        "global_threshold",
        "class_threshold"
    ],
    "num_known_list": [
        40,
        30,
        20,
        15,
        10
    ],

    "seeds": [0],

    "test_fold": 5
}

DRAW_CONFIG = {

    "dataset": "esc50",

    "embeddings": [
        "clap"
    ],

    "classifiers": [
        "cosine",
        "euclidean",
        "mahalanobis",
    ],
    "rejection_method": [
        "global_threshold",
        "class_threshold"
    ],
    "num_known_list": [
        30,
        20,
        15,
    ],

    "seeds": [0],

    "test_fold": 5
}