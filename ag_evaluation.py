"""Rand-Index (from scratch) sowie der Linkage-Vergleich bei festem Ziel-k - das
DBSCAN-Sweep-/kmeans-Multistart-Äquivalent für die "Warum scheitert Single-Linkage an
der Brücke?"-Sektion."""

import numpy as np

from ag_algorithm import labels_at_step, run, step_for_target_k
from ag_constants import LINKAGES


def rand_index(true_labels, pred_labels):
    """Anteil der Punktpaare, bei denen beide Partitionen uebereinstimmen (entweder
    beide im selben Cluster oder beide in unterschiedlichen). Punkte mit true_label -1
    (z. B. Brueckenpunkte ohne echte Gruppenzugehoerigkeit) werden ausgeschlossen -
    analog zu dbscan-demos per_group_noise_fraction, die echte Ausreisser ebenfalls
    nicht mitzaehlt."""
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    mask = true_arr != -1
    t, p = true_arr[mask], pred_arr[mask]
    n = len(t)
    if n < 2:
        return 1.0

    iu = np.triu_indices(n, k=1)
    same_true = (t[:, None] == t[None, :])[iu]
    same_pred = (p[:, None] == p[None, :])[iu]
    return float(np.mean(same_true == same_pred))


def linkage_comparison(data, true_labels, target_k):
    """Rand-Index je Linkage-Kriterium, alle beim GLEICHEN Ziel-k berechnet - macht
    direkt vergleichbar, wie stark das Ergebnis vom Kriterium abhaengt."""
    n = len(data)
    step = step_for_target_k(n, target_k)
    scores = {}
    for linkage in LINKAGES:
        result = run(data, linkage)
        labels = labels_at_step(n, result.merges, step)
        scores[linkage] = rand_index(true_labels, labels)
    return scores
