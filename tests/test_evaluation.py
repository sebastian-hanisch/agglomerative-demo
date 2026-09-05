import pytest

from ag_evaluation import linkage_comparison, rand_index
from ag_scenario import generate_instance


def test_rand_index_identical_partitions_is_one():
    assert rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == 1.0  # Umbenennung darf nichts aendern


def test_rand_index_hand_computed_partial_agreement():
    """true=[0,0,1,1], pred=[0,1,0,1]: von 6 Punktpaaren stimmen nur (0,3) und (1,2)
    ueberein (beide sagen 'verschiedene Cluster') - von Hand nachgerechnet."""
    assert rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(1 / 3)


def test_rand_index_excludes_true_label_minus_one():
    with_bridge = rand_index([0, 0, 1, 1, -1], [0, 0, 1, 1, 2])
    without_bridge = rand_index([0, 0, 1, 1], [0, 0, 1, 1])
    assert with_bridge == without_bridge == 1.0


def test_single_linkage_chaining_fails_where_others_succeed():
    """Kern-Behauptung der Demo: bei einer duennen Bruecke zwischen zwei Gruppen
    verschmilzt Single-Linkage sie beim Ziel-k=2 praktisch zufaellig (Rand-Index nahe
    0.5), waehrend Complete/Average/Ward beide Gruppen perfekt trennen (Rand-Index 1.0)."""
    instance = generate_instance(
        n_points=100, k=2, spread=0.225, size_imbalance=0.0, bridge_strength=1.0, seed=2
    )
    scores = linkage_comparison(instance.as_array(), instance.true_labels, target_k=2)

    assert scores["single"] < 0.6
    assert scores["complete"] > 0.9
    assert scores["average"] > 0.9
    assert scores["ward"] > 0.9


def test_no_bridge_all_linkages_agree_on_clear_groups():
    instance = generate_instance(
        n_points=60, k=3, spread=0.15, size_imbalance=0.0, bridge_strength=0.0, seed=1
    )
    scores = linkage_comparison(instance.as_array(), instance.true_labels, target_k=3)
    assert all(score > 0.9 for score in scores.values())
