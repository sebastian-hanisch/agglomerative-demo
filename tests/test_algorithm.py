import numpy as np
import pytest

from ag_algorithm import labels_at_step, run, step_for_target_k
from ag_scenario import generate_instance


def test_hand_computed_single_vs_complete_diverge():
    """Vier Punkte, von Hand konstruiert: A=(0,0), B=(1,0) sind sich am naechsten und
    verschmelzen unter jedem Kriterium zuerst. Danach ist C naeher an A als an B
    (AC=1.3, BC=sqrt(3.69)=1.92), waehrend D naeher an B ist (BD=2.0, AD=sqrt(5)=2.24) -
    Single-Linkage nimmt fuer den zweiten Schritt das MINIMUM (C an {A,B}, ueber die
    kurze A-C-Distanz), Complete-Linkage das MAXIMUM (das erzwingt stattdessen die
    direkte C-D-Fusion bei 1.7, da Completes {A,B}-C-Distanz mit 1.92 groesser waere).
    Von Hand nachgerechnet, nicht nur gegen scipy verifiziert."""
    data = np.array([[0.0, 0.0], [1.0, 0.0], [-0.5, 1.2], [1.0, 2.0]])

    single = run(data, "single")
    assert (single.merges[0].cluster_a, single.merges[0].cluster_b) == (0, 1)
    assert single.merges[0].distance == pytest.approx(1.0)
    assert (single.merges[1].cluster_a, single.merges[1].cluster_b) == (2, 4)
    assert single.merges[1].distance == pytest.approx(1.3)

    complete = run(data, "complete")
    assert (complete.merges[0].cluster_a, complete.merges[0].cluster_b) == (0, 1)
    assert (complete.merges[1].cluster_a, complete.merges[1].cluster_b) == (2, 3)
    assert complete.merges[1].distance == pytest.approx(1.7)


def test_n_minus_one_merges_and_shrinking_cluster_count():
    instance = generate_instance(n_points=40, k=3, spread=0.2, size_imbalance=0.0, bridge_strength=0.0, seed=1)
    for linkage in ("single", "complete", "average", "ward"):
        result = run(instance.as_array(), linkage)
        assert result.n_merges == instance.n_points - 1
        for step in range(result.n_merges):
            n_clusters = len(set(labels_at_step(instance.n_points, result.merges, step)))
            assert n_clusters == instance.n_points - step - 1


def test_step_for_target_k_matches_labels_at_step_cluster_count():
    instance = generate_instance(n_points=50, k=4, spread=0.2, size_imbalance=0.0, bridge_strength=0.0, seed=2)
    result = run(instance.as_array(), "average")
    for target_k in (2, 3, 5, 8):
        step = step_for_target_k(instance.n_points, target_k)
        labels = labels_at_step(instance.n_points, result.merges, step)
        assert len(set(labels)) == target_k


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_matches_scipy_linkage(linkage):
    """Unabhaengiger Kreuzvergleich gegen scipy.cluster.hierarchy.linkage: dieselbe
    Fusionsreihenfolge (als ungeordnetes Paar je Schritt) und dieselben Distanzen.
    scipy ist ausschliesslich ein Test-Dependency, kein Laufzeit-Dependency der App."""
    scipy_hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    scipy_distance = pytest.importorskip("scipy.spatial.distance")

    for seed in range(5):
        instance = generate_instance(
            n_points=25, k=3, spread=0.3, size_imbalance=0.4, bridge_strength=0.3, seed=seed
        )
        data = instance.as_array()
        ours = run(data, linkage)

        condensed = scipy_distance.pdist(data)
        Z = scipy_hierarchy.linkage(condensed, method=linkage)

        for i, merge in enumerate(ours.merges):
            our_pair = frozenset((merge.cluster_a, merge.cluster_b))
            scipy_pair = frozenset((int(Z[i, 0]), int(Z[i, 1])))
            assert our_pair == scipy_pair, f"step {i}: {our_pair} != {scipy_pair} ({linkage}, seed {seed})"
            assert merge.distance == pytest.approx(Z[i, 2], rel=1e-4, abs=1e-6)
