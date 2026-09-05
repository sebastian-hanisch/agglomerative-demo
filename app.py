"""Agglomeratives Clustering für schrittweise Depot-Konsolidierung - interaktive
Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - agglomeratives hierarchisches Clustering -
und lässt stattdessen die Empfindlichkeit gegenüber dem Linkage-Kriterium wachsen: von
klar getrennten Gruppen, bei denen alle vier Kriterien übereinstimmen, bis zu Gruppen, die
durch eine dünne Punktbrücke verbunden sind - dort verschmilzt Single-Linkage sie viel zu
früh ("Chaining"), während die übrigen Kriterien sie sauber getrennt halten. Viertes Stück
der "Konzepte"-Reihe, zweiter (unabhängiger) Vorläufer zu HDBSCAN neben dbscan-demo (siehe
README für die Einordnung).

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import ag_constants as C
from ag_algorithm import labels_at_step, run, step_for_target_k
from ag_evaluation import linkage_comparison
from ag_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ag_scenario import generate_instance
from ag_visualization import (
    build_dendrogram_figure,
    build_mini_scatter_figure,
    build_rand_index_bar_chart,
    build_scatter_figure,
)

st.set_page_config(page_title="Agglomeratives Clustering – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, size_imbalance, bridge_strength, seed, linkage):
    instance = generate_instance(n_points, k, spread, size_imbalance, bridge_strength, seed)
    result = run(instance.as_array(), linkage)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, linkage):
    return run(instance.as_array(), linkage)


@st.cache_data(show_spinner=False)
def _compute_linkage_comparison(instance, target_k):
    return linkage_comparison(instance.as_array(), instance.true_labels, target_k)


st.title("🔗 Agglomeratives Clustering: schrittweise Depot-Konsolidierung")
st.markdown(
    """
Kleinere Depots sollen schrittweise zu größeren Verteilzentren **konsolidiert** werden -
in welcher Reihenfolge, und wie viele sollten am Ende übrig bleiben? **Agglomeratives
hierarchisches Clustering** beantwortet genau das: es startet mit jedem Punkt als eigenem
Cluster und verschmilzt wiederholt das jeweils "nächstgelegene" Clusterpaar, bis nur noch
eines übrig ist - jede Fusion hat einen konkreten Konsolidierungs-Aufwand (die
Fusionsdistanz), und ein Schnitt bei einer gewünschten Zielanzahl zeigt exakt, welche
Depots zu diesem Zeitpunkt zusammengelegt wären. Genau **wie** das funktioniert, erklärt
der aufgeklappte Abschnitt direkt darunter - bevor weiter unten die Fusionen live ablaufen
und die Frage "📐 Warum scheitert Single-Linkage an der Brücke?" live beantwortet wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** "
    "Verfahren an einem wachsenden Beispiel: nicht Zufall (wie bei k-Means) oder ein "
    "Suchradius (wie bei DBSCAN) ist hier die Schwierigkeitsachse, sondern das gewählte "
    "**Linkage-Kriterium** - ein Parameter des einen gezeigten Verfahrens, keine vier "
    "verglichenen Methoden."
)

with st.expander("So funktioniert agglomeratives Clustering", expanded=True):
    st.markdown(
        """
Das Verfahren startet damit, dass jeder Punkt sein eigenes Cluster ist, und wiederholt
dann einen einzigen Schritt, bis nur noch ein Cluster übrig ist:

- **Fusion**: die beiden Cluster mit der kleinsten Distanz zueinander werden zu einem
  neuen Cluster verschmolzen.

Was "Distanz zwischen zwei Clustern" bedeutet, legt das **Linkage-Kriterium** fest:

- **Single (Minimum)**: die kleinste Distanz zwischen irgendeinem Punkt des einen und
  irgendeinem Punkt des anderen Clusters - anfällig für **Chaining**: eine dünne Kette
  naher Punkte kann zwei eigentlich getrennte Gruppen fälschlich verbinden, weil dafür nur
  EIN nahes Punktpaar reicht.
- **Complete (Maximum)**: die größte paarweise Distanz - das Gegenteil, tendenziell
  vorsichtiger.
- **Average (Mittelwert)**: die mittlere paarweise Distanz.
- **Ward (Varianzminimierung)**: die Fusion, die den geringsten Anstieg der
  cluster-internen Varianz verursacht.

Jede Fusion wird protokolliert (Reihenfolge, welche zwei Cluster, welche Distanz) - das
vollständige Protokoll ist das **Dendrogramm**: ein Baum, der die gesamte Fusionsgeschichte
zeigt. Ein Schnitt bei einer gewünschten Zielanzahl k liefert die Clusterzuordnung zu
diesem Zeitpunkt. Die Grafiken weiter unten zeigen genau das live: das Dendrogramm wächst
Fusion für Fusion, die Punktwolke daneben zeigt die aktuelle Aufteilung beim gewählten
Ziel-k.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel (klar getrennte Gruppen)": "Keine Brücke - alle vier Linkage-Kriterien liefern beim Schnitt auf die wahre Gruppenzahl dasselbe Ergebnis.",
    "Schwerer Fall (Chaining bei Single-Linkage)": "Zwei Gruppen plus eine dünne Punktbrücke dazwischen - Single-Linkage verschmilzt sie viel zu früh, die übrigen Kriterien bleiben sauber getrennt.",
    "Ungleiche Clustergrößen": "Zeigt eine zweite Eigenart: Single-Linkage neigt zu einem großen Ketten-Cluster plus Einzelpunkten, Ward zu ausgeglicheneren Größen.",
    "Viele Gruppen (Dendrogramm wächst)": "Mehr wahre Gruppen zeigen ein deutlich komplexeres Dendrogramm.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Standorte", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider("Anzahl wahrer Gruppen", *bounds("k_slider"), key="k_slider")
    spread = st.slider(
        "Streuung", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Klein = Gruppen klar getrennt. Groß = Gruppen überlappen sich spürbar.",
    )
    size_imbalance = st.slider(
        "Größen-Ungleichgewicht", *bounds("size_imbalance_slider"), key="size_imbalance_slider", step=0.05,
        help="0 = alle Gruppen gleich groß. 1 = eine Gruppe wird deutlich größer als die übrigen.",
    )
    bridge_strength = st.slider(
        "Brücken-Stärke (zwischen den ersten beiden Gruppen)", *bounds("bridge_strength_slider"),
        key="bridge_strength_slider", step=0.05,
        help="0 = keine Brücke. Höher = mehr verbindende Punkte zwischen Gruppe 1 und 2 - "
        "das Vehikel für Single-Linkage-Chaining.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Clustering**")
    linkage = st.radio(
        "Linkage-Kriterium", options=C.LINKAGES, key="linkage_radio",
        format_func=lambda l: C.LINKAGE_LABELS[l],
        help="Steuert die Primäransicht unten - der '📐'-Vergleich weiter unten prüft "
        "ohnehin immer alle vier Kriterien parallel.",
    )
    target_k = st.slider(
        "Ziel-Clusteranzahl (Schnitt)", *bounds("target_k_slider"), key="target_k_slider",
        help="Bei wie vielen Clustern das Dendrogramm für die Primäransicht geschnitten wird.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Standorte.",
    )

sync_query_params(n_points, k, spread, size_imbalance, bridge_strength, seed, linkage, target_k)

with st.spinner("Führe agglomeratives Clustering aus..."):
    instance, result = _compute_run(
        int(n_points), int(k), spread, size_imbalance, bridge_strength, int(seed), linkage
    )

max_step = result.n_merges - 1
target_step = min(max(step_for_target_k(instance.n_points, int(target_k)), 0), max_step)
run_key = (n_points, k, spread, size_imbalance, bridge_strength, seed, linkage, target_k)
if "ag_step" not in st.session_state or st.session_state.get("ag_step_owner") != run_key:
    st.session_state["ag_step"] = target_step
    st.session_state["ag_step_owner"] = run_key

st.markdown("## 🎯 Agglomeratives Clustering in Aktion")

step_col, play_col = st.columns([5, 1])
with step_col:
    step = st.slider(
        "Schritt (Fusion)", 0, max_step, key="ag_step",
        help="Ein Schritt = eine Fusion. Reglerposition entspricht standardmäßig der "
        "eingestellten Ziel-Clusteranzahl - frei verschiebbar, um die gesamte "
        "Fusionsgeschichte zu erkunden.",
    )
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

dendro_col, scatter_col = st.columns(2)
dendro_slot = dendro_col.empty()
scatter_slot = scatter_col.empty()


def _render(current_step):
    dendro_slot.plotly_chart(
        build_dendrogram_figure(instance.n_points, result.merges, current_step),
        width="stretch", key=f"dendro_{current_step}",
    )
    scatter_slot.plotly_chart(
        build_scatter_figure(instance, result.merges, current_step),
        width="stretch", key=f"scatter_{current_step}",
    )


if auto_play:
    n_frames = min(max_step + 1, 40)
    frame_skip = max(1, (max_step + 1) // n_frames)
    for s in list(range(0, max_step, frame_skip)) + [max_step]:
        _render(s)
        time.sleep(0.05)
    step = max_step
else:
    _render(step)

current_labels = labels_at_step(instance.n_points, result.merges, step)
lm1, lm2, lm3 = st.columns(3)
lm1.metric("Fusionen bisher", f"{step + 1}/{max_step + 1}")
lm2.metric("Cluster aktuell", len(set(current_labels)))
lm3.metric(
    "Fusionsdistanz (dieser Schritt)", f"{result.merges[step].distance:,.3f}",
    help="Distanz zwischen den beiden Clustern, die in diesem Schritt verschmolzen wurden.",
)

st.markdown("**Und mit anderen Linkage-Kriterien?**")
st.caption(
    "Gleiche Standorte, gleiche Ziel-Clusteranzahl wie oben - nur das Linkage-Kriterium "
    "unterscheidet sich."
)
example_cols = st.columns(len(C.LINKAGES))
for col, example_linkage in zip(example_cols, C.LINKAGES):
    with col:
        example_result = _compute_mini_run(instance, example_linkage)
        example_step = min(step_for_target_k(instance.n_points, int(target_k)), example_result.n_merges - 1)
        st.plotly_chart(
            build_mini_scatter_figure(instance, example_result.merges, example_step),
            width="stretch", key=f"mini_{example_linkage}",
        )
        st.caption(C.LINKAGE_LABELS[example_linkage])

st.markdown("---")

st.subheader("📐 Warum scheitert Single-Linkage an der Brücke?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet, nicht nur behauptet: der **Rand-Index** (Anteil
der Punktpaare, bei denen die berechnete Aufteilung mit der tatsächlichen
Gruppenzugehörigkeit übereinstimmt - 1.0 = perfekte Übereinstimmung, ~0.5 = kaum besser als
Zufall) für alle vier Linkage-Kriterien, jeweils beim selben Ziel-k geschnitten:
"""
)

scores = _compute_linkage_comparison(instance, int(target_k))
st.plotly_chart(build_rand_index_bar_chart(scores), width="stretch")

gap = max(scores.values()) - min(scores.values())
if gap > 0.3:
    worst = min(scores, key=scores.get)
    best = max(scores, key=scores.get)
    st.success(
        f"✅ Bei diesem Szenario erreicht {C.LINKAGE_LABELS[worst]} nur einen Rand-Index "
        f"von {scores[worst]:.2f}, während {C.LINKAGE_LABELS[best]} bei "
        f"{scores[best]:.2f} liegt - dasselbe Datenset, derselbe Ziel-k, aber ein "
        f"komplett anderes Ergebnis, nur weil sich das Linkage-Kriterium unterscheidet."
    )
else:
    st.info(
        "Bei diesem Szenario liegen alle vier Kriterien noch nah beieinander - eine "
        "stärkere Brücke (Regler links) macht den Unterschied deutlicher."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Algorithmus**: starte mit $n$ Clustern (jeder Punkt für sich), wiederhole - finde das
Clusterpaar $(A, B)$ mit minimaler Distanz $d(A,B)$, verschmelze es zu einem neuen Cluster -
bis nur noch ein Cluster übrig ist. Das Ergebnis ist ein **Dendrogramm**: $n-1$ Fusionen,
jede mit ihrer Fusionsdistanz.

**Linkage-Kriterien** (Distanz zwischen zwei Clustern $A$, $B$):

$$
d_{\text{single}}(A,B) = \min_{a \in A, b \in B} \lVert a-b \rVert, \qquad
d_{\text{complete}}(A,B) = \max_{a \in A, b \in B} \lVert a-b \rVert
$$

$$
d_{\text{average}}(A,B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} \lVert a-b \rVert,
\qquad
d_{\text{ward}}(A,B) = \sqrt{\frac{2|A||B|}{|A|+|B|}} \, \lVert \bar{a} - \bar{b} \rVert
$$

wobei $\bar{a}, \bar{b}$ die Schwerpunkte von $A$ und $B$ sind - Ward minimiert den Zuwachs
an cluster-interner Varianz, nicht direkt eine geometrische Distanz.

**Lance-Williams-Rekursion**: nach einer Fusion von $A,B$ zu $A \cup B$ lässt sich die neue
Distanz zu jedem anderen Cluster $C$ direkt aus den ALTEN Distanzen berechnen, ohne bei Null
neu zu rechnen:

$$
d(A \cup B, C) = \alpha_A \, d(A,C) + \alpha_B \, d(B,C) + \beta \, d(A,B) + \gamma \,
|d(A,C) - d(B,C)|
$$

mit Koeffizienten $\alpha_A, \alpha_B, \beta, \gamma$, die je nach Kriterium unterschiedlich
sind (implementiert in `ag_algorithm.py`, `_lance_williams_coeffs`) - eine Formel für alle
vier Kriterien, für Ward angewendet auf quadrierte statt rohe Distanzen (die etablierte
"ward.D2"-Konvention).

Naive Laufzeit $O(n^3)$ - $n-1$ Fusionen, je $O(n^2)$ um das naechste Paar zu finden. Mit
effizienteren Verfahren (z. B. SLINK für Single-Linkage) sinkt das auf $O(n^2)$ - hier
bewusst nicht implementiert.

**Rand-Index**: Anteil der Punktpaare, bei denen zwei Partitionen übereinstimmen (beide
sagen "gleiches Cluster" oder beide sagen "unterschiedliches Cluster"):

$$
\text{RI} = \frac{a + b}{\binom{n}{2}}
$$

wobei $a$ die Anzahl der Paare ist, die in beiden Partitionen im selben Cluster liegen, und
$b$ die Anzahl der Paare, die in beiden Partitionen in unterschiedlichen Clustern liegen.

**Und wenn die Dichte noch stärker variiert?** HDBSCANs Kern ist exakt Single-Linkage-
Hierarchie - aber auf einer "Mutual-Reachability-Distanz" statt der rohen euklidischen
Distanz. Das unterdrückt genau das oben live gezeigte Chaining-Problem, weil Abstände durch
dünn besiedelte ("Brücken"-)Regionen künstlich vergrößert werden, bevor überhaupt fusioniert
wird - ein zweiter, unabhängiger Vorläufer neben dbscan-demos Dichte-Ungleichgewicht-
Beobachtung, der auf dasselbe Verfahren zuläuft.

Implementiert in `ag_algorithm.py` (Lance-Williams-Fusionen) und `ag_evaluation.py`
(Rand-Index, Linkage-Vergleich).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
