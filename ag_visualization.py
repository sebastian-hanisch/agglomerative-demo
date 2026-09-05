"""Plotly-Visualisierungen: Dendrogramm (mit stabiler Blattreihenfolge, damit sich
Fusionslinien nicht kreuzen), Scatter-Plot beim aktuellen Schnitt (schrittanimierbar),
Kleinmultiples je Linkage-Kriterium und Rand-Index-Balkendiagramm."""

import numpy as np

from ag_algorithm import labels_at_step
from ag_constants import LINKAGE_LABELS, LINKAGES

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
MAX_LEGEND_CLUSTERS = 12


def _compute_dendrogram_layout(merges, n_points):
    """x-Position jedes Blatts und jedes Fusionsknotens - einmal ueber den VOLLSTAENDIGEN
    Fusionsverlauf berechnet (wie branch-bound-demos Baum-Layout), damit Positionen beim
    Durchblaettern nicht springen, nur welche Linien sichtbar sind aendert sich."""
    children = {m.new_cluster: (m.cluster_a, m.cluster_b) for m in merges}
    x_of = {}
    next_leaf_slot = [0]

    def assign(node_id):
        if node_id not in children:
            x_of[node_id] = next_leaf_slot[0]
            next_leaf_slot[0] += 1
            return x_of[node_id]
        a, b = children[node_id]
        xa, xb = assign(a), assign(b)
        x_of[node_id] = (xa + xb) / 2
        return x_of[node_id]

    root = n_points + len(merges) - 1
    assign(root)
    return x_of


def build_dendrogram_figure(n_points, merges, step):
    import plotly.graph_objects as go

    x_of = _compute_dendrogram_layout(merges, n_points)
    height_of = {i: 0.0 for i in range(n_points)}

    edge_x, edge_y = [], []
    for merge in merges[: step + 1]:
        ya = height_of.get(merge.cluster_a, 0.0)
        yb = height_of.get(merge.cluster_b, 0.0)
        xa, xb = x_of[merge.cluster_a], x_of[merge.cluster_b]
        y = merge.distance
        edge_x += [xa, xa, None, xb, xb, None, xa, xb, None]
        edge_y += [ya, y, None, yb, y, None, y, y, None]
        height_of[merge.new_cluster] = y

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#1f77b4", width=1.5), hoverinfo="skip")
    )
    fig.update_layout(
        template="plotly_white", height=320,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(title="Fusionsdistanz", fixedrange=True, rangemode="tozero"),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def _cluster_traces(data, labels, legend):
    import plotly.graph_objects as go

    traces = []
    cluster_ids = sorted(set(labels.tolist()))
    show_legend = legend and len(cluster_ids) <= MAX_LEGEND_CLUSTERS
    for cid in cluster_ids:
        mask = labels == cid
        color = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        traces.append(
            go.Scatter(
                x=data[mask, 0], y=data[mask, 1], mode="markers", name=f"Cluster {cid + 1}",
                showlegend=show_legend,
                marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
                hoverinfo="skip",
            )
        )
    return traces


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def build_scatter_figure(instance, merges, step):
    import plotly.graph_objects as go

    data = np.array(instance.points)
    labels = np.array(labels_at_step(instance.n_points, merges, step))

    fig = go.Figure()
    for trace in _cluster_traces(data, labels, legend=True):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    fig.update_layout(
        template="plotly_white", height=460,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    return fig


def build_mini_scatter_figure(instance, merges, step):
    import plotly.graph_objects as go

    data = np.array(instance.points)
    labels = np.array(labels_at_step(instance.n_points, merges, step))

    fig = go.Figure()
    for trace in _cluster_traces(data, labels, legend=False):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    fig.update_layout(
        template="plotly_white", height=200,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        margin=dict(t=5, l=5, r=5, b=5), showlegend=False,
    )
    return fig


def build_rand_index_bar_chart(scores):
    import plotly.graph_objects as go

    linkages = list(LINKAGES)
    fig = go.Figure(
        go.Bar(
            x=[LINKAGE_LABELS[l] for l in linkages], y=[scores[l] for l in linkages],
            marker_color=[CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(len(linkages))],
        )
    )
    fig.update_layout(
        template="plotly_white", height=280,
        yaxis=dict(title="Rand-Index", range=[0, 1.05], fixedrange=True),
        xaxis=dict(fixedrange=True), margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig
