"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die agglomerative
Clustering-Demo."""

DEFAULT_N_POINTS = 90
DEFAULT_K = 3
DEFAULT_SPREAD = 0.2
DEFAULT_SIZE_IMBALANCE = 0.0
DEFAULT_BRIDGE_STRENGTH = 0.0
DEFAULT_SEED = 7
DEFAULT_LINKAGE = "average"
DEFAULT_TARGET_K = 3

N_POINTS_MIN, N_POINTS_MAX = 30, 200
K_MIN, K_MAX = 2, 6
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
SIZE_IMBALANCE_MIN, SIZE_IMBALANCE_MAX = 0.0, 1.0
BRIDGE_STRENGTH_MIN, BRIDGE_STRENGTH_MAX = 0.0, 1.0
TARGET_K_MIN, TARGET_K_MAX = 2, 10
MAX_BRIDGE_POINTS = 80

LINKAGES = ("single", "complete", "average", "ward")
LINKAGE_LABELS = {
    "single": "Single (Minimum)",
    "complete": "Complete (Maximum)",
    "average": "Average (Mittelwert)",
    "ward": "Ward (Varianzminimierung)",
}

PRESETS = {
    "Einfaches Beispiel (klar getrennte Gruppen)": {
        "n_points": 60, "k": 3, "spread": 0.15, "size_imbalance": 0.0, "bridge_strength": 0.0,
        "linkage": "average", "target_k": 3, "seed": 1,
    },
    "Schwerer Fall (Chaining bei Single-Linkage)": {
        "n_points": 100, "k": 2, "spread": 0.225, "size_imbalance": 0.0, "bridge_strength": 1.0,
        "linkage": "single", "target_k": 2, "seed": 2,
    },
    "Ungleiche Clustergrößen": {
        "n_points": 100, "k": 3, "spread": 0.2, "size_imbalance": 0.85, "bridge_strength": 0.0,
        "linkage": "single", "target_k": 3, "seed": 3,
    },
    "Viele Gruppen (Dendrogramm wächst)": {
        "n_points": 120, "k": 6, "spread": 0.2, "size_imbalance": 0.0, "bridge_strength": 0.0,
        "linkage": "average", "target_k": 6, "seed": 4,
    },
}
