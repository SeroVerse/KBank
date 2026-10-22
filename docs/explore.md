---
title: Explore
subtitle: Interactive Geno-Glycomic Database & Graph Explorer
description: Interactive Marimo & Altair explorer for browsing KBank's curated bacterial surface polysaccharides, SNFG chemical graphs, SeroGen biosynthetic gene clusters, and pan-graphs.
author: Tom Stanton
icon: lucide/compass
comments: false
hide:
  - navigation
  - toc
tags:
  - explore
  - snfg
  - altair
  - marimo
  - layout
---

# Explore the KBank Geno-Glycomic Database

Interactive browser for exploring [`src/kbank/data/glycans.npz`](https://github.com/SeroVerse/KBank/blob/main/src/kbank/data/glycans.npz), built with [`LayoutEngine`][serogen.core.graph.layout.LayoutEngine] in [`scripts/explore_kbank.py`](https://github.com/SeroVerse/KBank/blob/main/scripts/explore_kbank.py):

- **`LocusPlotter(GraphPlotter)`**: Renders [`Locus`][serogen.loci.models.Locus] biosynthetic gene clusters using strand-aware directional arrows coloured by [`GeneRole`][serogen.genes.roles.GeneRole] or MCL homology cluster, plus 2D Fruchterman-Reingold Pan-Locus synteny networks ([`Loci.as_pan_graph()`][serogen.core.graph.models.GraphBatch.as_pan_graph]).
- **`GlycanPlotter(GraphPlotter)`**: Renders [`Glycan`][kbank.glycan.Glycan] repeating units according to **SNFG (Symbol Nomenclature for Glycans)** shapes and stereochemical colours, plus 2D Fruchterman-Reingold Pan-Glycan monosaccharide transition networks ([`Glycans.as_pan_graph()`][serogen.core.graph.models.GraphBatch.as_pan_graph]).
- **Cross-Domain Gene $\leftrightarrow$ Glycan Highlighting**: Select any paired record in the dropdown and hover over a gene arrow or SNFG monosaccharide to highlight its bipartite [`GeneGlycanAssignment`][kbank.biosynthesis.align.GeneGlycanAssignment].

<div class="sv-app-toolbar">
  <span><strong>Interactive Marimo App</strong> · Powered by <code>SeroGen LayoutEngine</code>, <code>LocusPlotter</code> &amp; <code>GlycanPlotter</code> (SNFG)</span>
  <div class="sv-app-links">
    <a class="sv-deck-btn" href="../notebooks/assets/explore_kbank.html" target="_blank" rel="noopener">
      Open Fullscreen Explorer ↗
    </a>
  </div>
</div>

<iframe class="sv-app-frame" src="../notebooks/assets/explore_kbank.html" title="KBank Interactive Geno-Glycomic Database Explorer" loading="lazy" allowfullscreen></iframe>

---

## Run or Customize Locally

You can launch [`scripts/explore_kbank.py`](https://github.com/SeroVerse/KBank/blob/main/scripts/explore_kbank.py) locally as an interactive Marimo web app or open it in the reactive notebook editor to plot custom [`Glycan`][kbank.glycan.Glycan] and [`Locus`][serogen.loci.models.Locus] objects with `GlycanPlotter` and `LocusPlotter`:

```bash
# Launch interactive web app
uv run --with marimo --with altair marimo run scripts/explore_kbank.py

# Open in the reactive Marimo editor
uv run --with marimo --with altair marimo edit scripts/explore_kbank.py
```
