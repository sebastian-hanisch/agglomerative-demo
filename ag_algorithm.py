"""Agglomeratives hierarchisches Clustering from scratch, über die Lance-Williams-
Rekursionsformel (eine Formel für alle vier Linkage-Kriterien), mit vollständigem
Fusions-Protokoll, damit die App Fusion für Fusion (nicht nur das Endergebnis)
durchblättern kann - analog zu den Schritt-Protokollen in kmeans-demo/dbscan-demo.

Bewusst ohne scipy zur Laufzeit implementiert. scipy.cluster.hierarchy.linkage dient in
tests/ nur als unabhängiger Kreuzvergleich für die eigene Implementierung.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Merge:
    step: int  # Reihenfolge der Fusion (0-indiziert)
    cluster_a: int
    cluster_b: int
    distance: float  # Fusionsdistanz (fuer "ward" bereits zurueckgewurzelt, siehe unten)
    new_cluster: int  # neue Cluster-ID (n, n+1, ... - wie scipy.cluster.hierarchy.linkage)
    members: tuple  # ALLE urspruenglichen Punktindizes in diesem neuen Cluster


@dataclass(frozen=True)
class RunResult:
    merges: tuple  # n-1 Merge-Eintraege in chronologischer Reihenfolge
    n_points: int
    linkage: str

    @property
    def n_merges(self):
        return len(self.merges)


def _key(i, j):
    return (i, j) if i < j else (j, i)


def _lance_williams_coeffs(linkage, n_i, n_j, n_k):
    if linkage == "single":
        return 0.5, 0.5, 0.0, -0.5
    if linkage == "complete":
        return 0.5, 0.5, 0.0, 0.5
    if linkage == "average":
        total = n_i + n_j
        return n_i / total, n_j / total, 0.0, 0.0
    if linkage == "ward":
        total = n_i + n_j + n_k
        return (n_i + n_k) / total, (n_j + n_k) / total, -n_k / total, 0.0
    raise ValueError(f"unbekanntes Linkage-Kriterium: {linkage}")


def run(data, linkage):
    """Fuehrt agglomeratives Clustering vollstaendig protokolliert aus. Fuer "ward"
    rechnet die Lance-Williams-Rekursion auf QUADRIERTEN Distanzen (die etablierte
    ward.D2-Konvention, die auch scipy.cluster.hierarchy.linkage(method="ward")
    verwendet) - die berichtete Merge.distance ist dann die Wurzel daraus, fuer die
    uebrigen drei Kriterien wird direkt mit den rohen euklidischen Distanzen gerechnet."""
    n = len(data)
    diffs = data[:, None, :] - data[None, :, :]
    raw_dist = np.sqrt((diffs ** 2).sum(axis=2))
    squared_space = linkage == "ward"
    base = raw_dist ** 2 if squared_space else raw_dist

    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = float(base[i, j])

    active = set(range(n))
    size = {i: 1 for i in range(n)}
    members = {i: (i,) for i in range(n)}
    next_id = n
    merges = []

    for step in range(n - 1):
        a, b = min(dist, key=dist.get)
        d_ab = dist[(a, b)]
        others = active - {a, b}

        old_da = {c: dist[_key(a, c)] for c in others}
        old_db = {c: dist[_key(b, c)] for c in others}

        new_id = next_id
        next_id += 1
        new_vals = {}
        for c in others:
            alpha_i, alpha_j, beta, gamma = _lance_williams_coeffs(linkage, size[a], size[b], size[c])
            da, db = old_da[c], old_db[c]
            new_vals[c] = alpha_i * da + alpha_j * db + beta * d_ab + gamma * abs(da - db)

        for key_ in [k for k in dist if a in k or b in k]:
            del dist[key_]
        for c, v in new_vals.items():
            dist[_key(new_id, c)] = v

        size[new_id] = size[a] + size[b]
        members[new_id] = members[a] + members[b]
        active.discard(a)
        active.discard(b)
        active.add(new_id)

        merge_distance = math.sqrt(max(d_ab, 0.0)) if squared_space else d_ab
        merges.append(Merge(step, a, b, merge_distance, new_id, members[new_id]))

    return RunResult(merges=tuple(merges), n_points=n, linkage=linkage)


def labels_at_step(n_points, merges, step):
    """Partition nach genau `step + 1` angewendeten Fusionen (0-indiziert), als fortlaufend
    nummerierte Labels 0..k-1 (Reihenfolge nach erstem Auftreten, nur fuer stabile
    Einfaerbung - keine inhaltliche Bedeutung)."""
    label = list(range(n_points))
    for merge in merges[: step + 1]:
        for p in merge.members:
            label[p] = merge.new_cluster

    seen = {}
    result = []
    for l in label:
        if l not in seen:
            seen[l] = len(seen)
        result.append(seen[l])
    return tuple(result)


def step_for_target_k(n_points, target_k):
    """Schrittindex (0-indiziert, inklusive), bei dem genau target_k Cluster aktiv sind."""
    target_k = max(1, min(target_k, n_points))
    return n_points - target_k - 1
