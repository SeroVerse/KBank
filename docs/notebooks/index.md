---
title: Benchmarks & Notebooks
subtitle: Interactive 16:9 Marimo Slide Decks
description: Reproducible Marimo notebooks and interactive 16:9 slide decks covering multi-database bacterial glycan curation, 10-head 1D-CNN enzyme classifiers, and Relational MPNN structure prediction.
author: Tom Stanton
icon: lucide/presentation
comments: true
tags:
  - benchmarks
  - marimo
  - notebooks
---

# Benchmarks & Interactive Slide Decks

KBank maintains reproducible [Marimo](https://marimo.io) notebooks in [`scripts/`](https://github.com/SeroVerse/KBank/tree/main/scripts) covering every stage of the pipeline: multi-database bacterial glycan and biosynthetic locus curation, 10-head factorized 1D-CNN enzyme specificity classification, and `RelationalMPNN` + constrained spanning arborescence *de novo* structure prediction.

Each notebook is configured with a native **16:9 Marimo slide layout** (`scripts/layouts/*.slides.json`) and a cached session snapshot (`scripts/__marimo__/session/*.py.json`) that exports directly into the interactive slide decks embedded in this section.

<div class="grid cards" markdown>

-   :lucide-database:{ .lg .middle } **Glycan & Biosynthetic Locus Curation (`curate_glycans.py`)**

    ---

    End-to-end streaming, canonicalization, and cross-linking of **1,530** bacterial surface polysaccharide structures across **CSDB**, **KEGG**, **ECODAB**, **EPS-DB**, **EK3D**, and **GlyTouCan**, paired *de novo* with **565** `SeroGen` biosynthetic gene clusters (`src/kbank/data/glycans.npz`).

    [:octicons-arrow-right-24: Explore Curation Deck](curate-glycans.md)

-   :lucide-cpu:{ .lg .middle } **10-Head 1D-CNN Cluster Classifiers (`train_cluster_classifier.py`)**

    ---

    2-pass Expectation-Maximization (EM) self-training and 5-fold cross-validation of [`BiosyntheticClassifierSuite`][kbank.biosynthesis.classifier.BiosyntheticClassifierSuite] across all 10 enzymatic specificity heads (`donor_linkage`, `donor`, `linkage`, `acceptor_sugar`, `acceptor_motif`, `is_backbone`, `mechanism`, `initiator_sugar`, `wzy_linkage`, `modifier_linkage`).

    [:octicons-arrow-right-24: Explore Classifier Deck](cluster-classifier.md)

-   :lucide-network:{ .lg .middle } **MPNN Glycan Structure Predictor (`train_glycan_predictor.py`)**

    ---

    Two-Phase Optuna TPE optimization of [`RelationalMPNN`][serogen.core.graph.gnn.RelationalMPNN] and [`GlycanPredictorParams`][kbank.biosynthesis.predictor.GlycanPredictorParams] (`glycan_predictor_mpnn.npz`), achieving **>92% Monosaccharide Composition F1**, **>82% Disaccharide Bigram F1**, and **100% topological + stereochemical isomorphism** on held-out *K. pneumoniae* `KL106` (`LT174576.1`) and `KL107` (`KF793263.1`) capsules.

    [:octicons-arrow-right-24: Explore Predictor Deck](glycan-predictor.md)

</div>

---

## Running Notebooks Locally

All notebooks can be launched interactively as reactive web apps, slide presentations, or editable Marimo notebooks via [`uv`](https://docs.astral.sh/uv/):

=== "Present as Slideshow"

    ```bash
    # Launch any notebook directly in Marimo's fullscreen slide presentation mode
    uv run --with marimo --with altair marimo run scripts/curate_glycans.py
    uv run --with marimo --with altair marimo run scripts/train_cluster_classifier.py
    uv run --with marimo --with altair --with optuna marimo run scripts/train_glycan_predictor.py
    ```

=== "Interactive Editor"

    ```bash
    # Open in the reactive Marimo notebook editor
    uv run --with marimo --with altair marimo edit scripts/train_glycan_predictor.py
    ```

=== "Refresh Cached Session Snapshot"

    ```bash
    # Re-execute a notebook and update its cached session snapshot in scripts/__marimo__/session/
    uv run --with marimo --with altair marimo export session --no-sandbox --force-overwrite scripts/train_glycan_predictor.py
    ```
