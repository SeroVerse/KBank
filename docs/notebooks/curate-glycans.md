---
title: Glycan & Locus Curation
subtitle: 1,530 CSDB/IUPAC/KEGG Glycan-Locus Pairs
description: Multi-database streaming, chemical graph canonicalization, taxonomy normalization, and 4-tier NCBI/PubMed SeroGen biosynthetic locus pairing pipeline.
author: Tom Stanton
icon: lucide/database
comments: true
tags:
  - benchmarks
  - curation
  - csdb
  - iupac
---

# Bacterial Surface Glycan & Biosynthetic Locus Curation

This presentation documents the multi-database streaming, chemical graph canonicalization, taxonomy normalization, and 4-tier NCBI/PubMed `SeroGen` biosynthetic locus pairing pipeline in [`kbank.io.sources`][kbank.io.sources] and [`kbank.db.curation`][kbank.db.curation], producing the unified Structure-of-Arrays [`GlycanRecords`][kbank.db.records.GlycanRecords] archive (`src/kbank/data/glycans.npz`).

<div class="serogen-slide-frame">
  <iframe src="assets/curate_glycans.html" title="KBank Bacterial Surface Glycan & Biosynthetic Locus Curation Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **5 Integrated Surface-Antigen Databases**: Consolidates **CSDB**, **KEGG Glycan**, **ECODAB**, **CERMAV EPS-DB**, and **EK3D** with **GlyTouCan** accession cross-references into **1,530** non-redundant bacterial surface polysaccharide graphs (`7,788` residue/modifier nodes and `7,660` glycosidic linkages across `154` species).
- **De Novo `SeroGen` Locus Pairing**: Resolves NCBI Genome Assemblies and nucleotide BGC accessions via `cassetter`-cached Entrez queries, segments candidate polysaccharide loci with [`LocusPredictor`][serogen.loci.predictor.LocusPredictor], and aligns `7,940` classified BGC genes across **565** paired records via [`BiosyntheticAssigner`][kbank.biosynthesis.align.BiosyntheticAssigner].
- **Zero-Pickle SoA Archive**: Compiles all graphs, loci, and bipartite `GeneGlycanAssignments` into `src/kbank/data/glycans.npz` (`allow_pickle=False`).

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair marimo run scripts/curate_glycans.py

# Edit notebook cells
uv run --with marimo --with altair marimo edit scripts/curate_glycans.py
```
