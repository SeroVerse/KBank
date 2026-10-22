---
title: MPNN Glycan Predictor
subtitle: Relational MPNN & Arborescence Solver
description: Two-Phase Optuna TPE optimization, held-out multi-regime structural validation, and Klebsiella pneumoniae KL106/KL107 de novo glycan structure benchmarks.
author: Tom Stanton
icon: lucide/git-fork
status: new
comments: true
tags:
  - benchmarks
  - mpnn
  - arborescence
  - klebsiella
---

# MPNN Glycan Structure Predictor (`GlycanPredictor`)

This presentation documents the **Two-Phase Optuna TPE hyperparameter optimization**, held-out multi-regime structural validation, and out-of-sample *Klebsiella pneumoniae* `KL106` (`LT174576.1`) and `KL107` (`KF793263.1`) capsule benchmarks of [`GlycanPredictor`][kbank.biosynthesis.predictor.GlycanPredictor] and [`RelationalMPNN`][serogen.core.graph.gnn.RelationalMPNN] (`src/kbank/data/glycan_predictor_mpnn.npz`).

<div class="serogen-slide-frame">
  <iframe src="assets/train_glycan_predictor.html" title="KBank MPNN Glycan Structure Predictor Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **Two-Phase Optuna TPE Hyperparameter Optimization**:
  - **Phase 1 (`RelationalMPNN`)**: Tunes message-passing hidden dimension, layer depth, factored bilinear edge-scoring rank, 2-hop syntenic edges, learning rate, weight decay, and single-free-GT clamping probability on pre-extracted biosynthetic arborescences.
  - **Phase 2 (`GlycanPredictorParams`)**: Jointly tunes all 14 Stage 3/4 node and pairwise arborescence potential blending weights across Cross-Validated (unseen record) and Pure Sequence De Novo (`cluster_id=0`) evaluation regimes.
- **Exact Out-of-Sample *K. pneumoniae* `KL106 -> K106` & `KL107 -> K107` Recovery**: Fetches public NCBI GenBank accessions `LT174576.1` (`KL106`) and `KF793263.1` (`KL107`), segments their CPS loci de novo via `SeroGen`'s [`LocusPredictor`][serogen.loci.predictor.LocusPredictor], and recovers both complete capsule structures with **100% topological and stereochemical isomorphism (`WWL = 0.0000`)**.
- **Automatic Production Model Export**: Persists both the Optuna-selected `RelationalMPNN` weights and the 14 tuned `GlycanPredictorParams` potentials in `src/kbank/data/glycan_predictor_mpnn.npz`.

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair --with optuna marimo run scripts/train_glycan_predictor.py

# Edit notebook cells
uv run --with marimo --with altair --with optuna marimo edit scripts/train_glycan_predictor.py
```
