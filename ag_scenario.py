"""Zufällige 2D-Punktwolken für die agglomerative Clustering-Demo: k Gauß-Cluster mit
einstellbarem Größen-Ungleichgewicht, plus eine optionale dünne Brücke aus Punkten
zwischen den ersten beiden Clustern - das Vehikel für den Single-Linkage-Chaining-Effekt."""

from dataclasses import dataclass

import numpy as np

from ag_constants import MAX_BRIDGE_POINTS

RING_RADIUS = 2.0
MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...)
    true_labels: tuple  # Gruppenindex, oder -1 für Brückenpunkte (gehören zu keiner Gruppe)
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_shares(k, size_imbalance):
    """Wie in kmeans-demo/km_scenario.py: Cluster 0 wird mit wachsendem size_imbalance
    groesser, die uebrigen entsprechend kleiner - hier bewusst Groesse (nicht Dichte wie
    in dbscan-demo), um den "Ungleiche Clustergroessen"-Effekt auf Linkage-Kriterien
    zu zeigen."""
    weights = np.ones(k)
    if k > 1:
        weights[0] = 1 + size_imbalance * 2 * (k - 1)
    return weights / weights.sum()


def _generate_blobs(n_points, k, spread, size_imbalance, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    std = max(spread, MIN_STD_FRACTION) * RING_RADIUS

    shares = _cluster_shares(k, size_imbalance)
    counts = np.maximum(1, np.round(shares * n_points).astype(int))
    counts[-1] += n_points - counts.sum()
    counts = np.maximum(counts, 1)

    points_per_cluster, labels_per_cluster = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=std, size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))
    return np.concatenate(points_per_cluster, axis=0), np.concatenate(labels_per_cluster, axis=0), centers


def _add_bridge(points, labels, center_a, center_b, bridge_strength, rng):
    n_bridge = int(round(bridge_strength * MAX_BRIDGE_POINTS))
    if n_bridge <= 0:
        return points, labels
    t = rng.uniform(0.0, 1.0, n_bridge)
    base = center_a[None, :] + t[:, None] * (center_b - center_a)[None, :]
    jitter_std = 0.08 * RING_RADIUS
    bridge_points = base + rng.normal(scale=jitter_std, size=base.shape)
    bridge_labels = np.full(n_bridge, -1)
    return np.concatenate([points, bridge_points]), np.concatenate([labels, bridge_labels])


def generate_instance(n_points, k, spread, size_imbalance, bridge_strength, seed):
    """bridge_strength (0 bis 1) erzeugt bis zu MAX_BRIDGE_POINTS zusaetzliche Punkte
    entlang der Verbindungslinie zwischen den Zentren von Cluster 0 und Cluster 1 (nur bei
    k>=2 wirksam) - das Vehikel, um Single-Linkage-Chaining sichtbar zu machen. Diese
    Bruecken-Punkte gehoeren zu keiner echten Gruppe (true_label -1), aehnlich den
    Ausreisserpunkten in dbscan-demo."""
    rng = np.random.default_rng(seed)
    points, labels, centers = _generate_blobs(n_points, k, spread, size_imbalance, rng)
    if k >= 2 and bridge_strength > 0:
        points, labels = _add_bridge(points, labels, centers[0], centers[1], bridge_strength, rng)
    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        k=k,
    )
