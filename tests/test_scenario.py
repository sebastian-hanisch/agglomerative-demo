import numpy as np

from ag_scenario import generate_instance


def test_point_count_matches_request_including_bridge():
    instance = generate_instance(
        n_points=100, k=2, spread=0.2, size_imbalance=0.0, bridge_strength=0.5, seed=1
    )
    assert instance.n_points == 100 + 40  # 50% von MAX_BRIDGE_POINTS (80) = 40
    assert instance.true_labels.count(-1) == 40


def test_no_bridge_by_default_strength_zero():
    instance = generate_instance(
        n_points=60, k=3, spread=0.2, size_imbalance=0.0, bridge_strength=0.0, seed=1
    )
    assert -1 not in instance.true_labels
    assert instance.n_points == 60


def test_reproducible_given_same_seed():
    kwargs = dict(n_points=80, k=3, spread=0.2, size_imbalance=0.3, bridge_strength=0.3, seed=42)
    a = generate_instance(**kwargs)
    b = generate_instance(**kwargs)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_size_imbalance_makes_group_zero_larger():
    instance = generate_instance(
        n_points=150, k=3, spread=0.2, size_imbalance=0.85, bridge_strength=0.0, seed=3
    )
    labels = np.array(instance.true_labels)
    counts = np.bincount(labels, minlength=3)
    assert counts[0] == counts.max()
    assert counts[0] > counts[1:].mean() * 2


def test_bridge_points_lie_between_the_first_two_cluster_centers():
    """Bruecken-Punkte sollen ungefaehr auf der Verbindungslinie zwischen den ersten
    beiden Cluster-Zentren liegen - geprueft ueber ihre Projektion auf diese Linie,
    die im Intervall [0, 1] (zwischen den Zentren) liegen sollte, nicht weit davor
    oder dahinter."""
    instance = generate_instance(
        n_points=100, k=2, spread=0.15, size_imbalance=0.0, bridge_strength=1.0, seed=2
    )
    points = np.array(instance.points)
    labels = np.array(instance.true_labels)
    bridge_points = points[labels == -1]

    center_a = points[labels == 0].mean(axis=0)
    center_b = points[labels == 1].mean(axis=0)
    direction = center_b - center_a
    t = ((bridge_points - center_a) @ direction) / (direction @ direction)

    assert bridge_points.shape[0] > 0
    assert np.mean((t >= -0.15) & (t <= 1.15)) > 0.9
