# Agglomeratives Clustering für schrittweise Depot-Konsolidierung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-agglomerative-demo.streamlit.app/)**

Viertes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **zweiter, unabhängiger Vorläufer zu HDBSCAN** neben
[dbscan-demo](../dbscan-demo) - kein drittes Glied einer Kette. Die Lineage ist damit ein
echter DAG mit einem Zusammenfluss:

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo
agglomerative-demo ──────────┘
kmeans-demo ─┐
             ├──> divisive-demo
agglomerative-demo ─┘
```

[hdbscan-demo](../hdbscan-demo) ist inzwischen gebaut und weist live nach, dass HDBSCAN
sowohl das Chaining-Problem dieser Demo als auch dbscan-demos Dichte-Ungleichgewicht-
Problem löst. [divisive-demo](../divisive-demo) (Bisecting k-Means) ist dagegen kein Fix,
sondern ein bewusster Kontrast: dieselbe Art Hierarchie, top-down statt bottom-up gebaut -
Single-Linkages Chaining-Neigung, hier die Schwäche bei einer Brücke, ist dort - ohne
Brücke - genau der Mechanismus, der nicht-konvexe Formen erfolgreich trennt.

HDBSCAN baut buchstäblich auf einer Hierarchie via Single-Linkage-artigem Merging auf -
nur mit einer dichte-angepassten ("mutual reachability") statt der rohen Distanz. Diese
Demo zeigt live, welches konkrete Problem das behebt: **Chaining bei Single-Linkage** -
eine dünne Brücke aus Punkten kann zwei eigentlich getrennte Gruppen fälschlich zu einer
verschmelzen lassen, weil Single-Linkage nur EIN nahes Punktpaar zwischen zwei Clustern
braucht, um sie zu vereinen.

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo **ein** Verfahren – agglomeratives hierarchisches Clustering
– und lässt stattdessen die Empfindlichkeit gegenüber dem **Linkage-Kriterium** wachsen.
Vehikel-Problem: kleinere Depots sollen schrittweise zu größeren Verteilzentren
konsolidiert werden - in welcher Reihenfolge, und wie viele sollten am Ende übrig
bleiben? Das Dendrogramm beantwortet genau das, jede Fusion mit ihrem konkreten
Konsolidierungs-Aufwand (der Fusionsdistanz).

## Warum diese Demo anders aufgebaut ist

Bei k-Means war die Schwierigkeitsachse der Zufall der Startpunkte, bei DBSCAN die Wahl
von eps/min_samples. Hier ist es die Wahl des **Linkage-Kriteriums** - single/complete/
average/Ward sind ein Parameter des einen gezeigten Verfahrens, keine vier verglichenen
Methoden:

- **Einfaches Beispiel**: keine Brücke - alle vier Kriterien liefern beim Schnitt auf die
  wahre Gruppenzahl dasselbe Ergebnis.
- **Schwerer Fall**: zwei Gruppen plus eine dünne Punktbrücke dazwischen - Single-Linkage
  verschmilzt sie beim Ziel-k=2 praktisch zufällig (Rand-Index ≈ 0.5), während
  Complete/Average/Ward beide Gruppen perfekt trennen (Rand-Index 1.0). Live in der
  "📐"-Sektion nachgewiesen, nicht nur behauptet.
- **Ungleiche Clustergrößen**: Single-Linkage neigt zu einem großen "Ketten"-Cluster plus
  vielen Einzelpunkten, Ward zu ausgeglicheneren Größen.
- **Viele Gruppen**: zeigt wachsende Dendrogramm-Komplexität.

## Visualisierung

Dendrogramm und Punktwolke laufen synchron über denselben Schritt-Regler (ein Schritt =
eine Fusion) - das Dendrogramm-Layout wird einmal über den vollständigen Fusionsverlauf
berechnet (wie branch-bound-demos Baum-Layout), damit Positionen beim Durchblättern nicht
springen. Die "Und mit anderen Linkage-Kriterien?"-Kleinmultiples zeigen dieselben Daten
beim selben Ziel-k unter allen vier Kriterien nebeneinander - ein echter
Eins-zu-eins-Vergleich bei sonst identischen Bedingungen.

## Sicherheitsgrenzen

Keine eigene Iterationsgrenze nötig - das Verfahren terminiert immer nach exakt $n-1$
Fusionen. `N_POINTS_MAX` (200) hält die naive $O(n^3)$-Suche schnell genug für eine
flüssige Animation.

## Verifikation

- **Handgerechnetes Beispiel**: vier Punkte, bei denen Single- und Complete-Linkage im
  zweiten Schritt nachweislich unterschiedliche Cluster verschmelzen (von Hand
  nachgerechnet, nicht nur gegen scipy verifiziert).
- **Kreuzvergleich mit scipy**: identische Fusionsreihenfolge und -distanzen wie
  `scipy.cluster.hierarchy.linkage` für alle vier Kriterien, über mehrere Szenarien und
  Seeds (scipy ist ausschließlich ein Test-Dependency, kein Laufzeit-Dependency der App).
- **Struktur-Invarianten**: genau $n-1$ Fusionen, jede reduziert die Clusterzahl um genau 1.
- **Kern-Behauptung der Demo direkt getestet**: bei aktiver Brücke liegt der Rand-Index
  von Single-Linkage bei Ziel-k=2 unter 0.6, während Complete/Average/Ward über 0.9 liegen
  (`test_single_linkage_chaining_fails_where_others_succeed`).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Fusions-Animation, Kleinmultiples, Rand-Index-Vergleich, Formulierungs-Expander |
| `ag_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS`, `LINKAGES` |
| `ag_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `ag_scenario.py` | Zufällige Punktwolken: Gauß-Gruppen mit Größen-Ungleichgewicht, plus optionale Punktbrücke zwischen den ersten beiden Gruppen |
| `ag_algorithm.py` | Agglomeratives Clustering from scratch über die Lance-Williams-Rekursionsformel, mit vollständigem Fusions-Protokoll |
| `ag_evaluation.py` | Rand-Index (from scratch), Linkage-Vergleich bei festem Ziel-k |
| `ag_visualization.py` | Dendrogramm-, Punktwolken- und Rand-Index-Diagramm (Plotly) |
| `tests/` | Handinstanz, scipy-Kreuzvergleich, Struktur-Invarianten, Szenario-Reproduzierbarkeit, Chaining-Effekt |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
