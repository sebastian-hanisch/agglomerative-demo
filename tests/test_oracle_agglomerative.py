"""Unabhängige Orakel für das agglomerative Clustering (Regressionstest der Orakelprüfung).

scipy.cluster.hierarchy.linkage (Fusionspaare, Cluster-IDs, Distanzen, Größen) auf zufälligen Instanzen aller Formen/Linkage-Kriterien; die Partition bei jedem Ziel-k gegen `fcluster(maxclust)`;
der Rand-Index gegen scikit-learns `rand_score`. Bei exakten Gleichständen (ganzzahliges Raster, doppelte Punkte) wählen die Bibliotheken verschiedene Fusionen: dort wird nur verglichen, was
davon unabhängig ist (Single-Linkage: Menge der Fusionsdistanzen = Minimalgerüst; alle Kriterien: n-1 Fusionen, genau k Cluster beim Schnitt auf k)."""

import numpy as np
import pytest

from ag_algorithm import labels_at_step, run, step_for_target_k
from ag_evaluation import rand_index
from ag_scenario import generate_instance

hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
pdist = pytest.importorskip("scipy.spatial.distance").pdist
LINKAGES = ("single", "complete", "average", "ward")


def _same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])


def test_tie_free_instances_match_scipy_pairs_ids_distances_sizes_and_partitions():
    rng = np.random.default_rng(1)
    for it in range(80):
        linkage = LINKAGES[it % 4]
        if it % 2:
            X = generate_instance(int(rng.integers(8, 45)), int(rng.integers(2, 6)), float(rng.choice([0.05, 0.2, 0.6])), float(rng.choice([0, 0.4, 1.0])),
                                  float(rng.choice([0, 0.5, 1.0])), int(rng.integers(0, 1000)), shape=str(rng.choice(["blobs", "moons"]))).as_array()
        else:
            X = rng.normal(size=(int(rng.integers(3, 35)), int(rng.integers(1, 4))))
        n = len(X)
        result = run(X, linkage)
        Z = hierarchy.linkage(pdist(X), method=linkage)
        assert result.n_merges == n - 1
        for i, m in enumerate(result.merges):
            assert frozenset((m.cluster_a, m.cluster_b)) == frozenset((int(Z[i, 0]), int(Z[i, 1])))
            assert m.new_cluster == n + i and len(m.members) == int(Z[i, 3])
            assert m.distance == pytest.approx(Z[i, 2], rel=1e-8, abs=1e-8)
        for k in range(2, n + 1):
            labels = labels_at_step(n, result.merges, step_for_target_k(n, k))
            assert len(set(labels)) == k
            ref = hierarchy.fcluster(Z, k, criterion="maxclust")
            if len(set(ref)) == k:                                                                  # scipy liefert bei gleichen Höhen evtl. weniger Cluster
                assert _same_partition(labels, ref)


def test_instances_with_exact_ties_still_give_a_valid_hierarchy():
    rng = np.random.default_rng(2)
    for it in range(60):
        X = rng.integers(0, 5, (int(rng.integers(3, 30)), 2)).astype(float)
        linkage = LINKAGES[it % 4]
        result = run(X, linkage)
        n = len(X)
        d = [m.distance for m in result.merges]
        if linkage == "single":
            assert np.allclose(sorted(d), sorted(hierarchy.linkage(pdist(X), method="single")[:, 2]), atol=1e-9)
        assert result.n_merges == n - 1 and sorted(result.merges[-1].members) == list(range(n))
        for k in range(1, n + 1):
            assert len(set(labels_at_step(n, result.merges, step_for_target_k(n, k)))) == k


def test_rand_index_matches_scikit_learn_without_the_bridge_points():
    rand_score = pytest.importorskip("sklearn.metrics").rand_score
    rng = np.random.default_rng(3)
    for _ in range(150):
        n = int(rng.integers(2, 40))
        true, pred = rng.integers(-1, 3, n), rng.integers(0, 4, n)
        keep = true != -1
        if keep.sum() >= 2:
            assert rand_index(true, pred) == pytest.approx(rand_score(true[keep], pred[keep]), abs=1e-12)
        else:
            assert rand_index(true, pred) == 1.0
