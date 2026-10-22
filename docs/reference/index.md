---
title: API Reference
subtitle: Public Modules & Classes
description: Complete API reference for KBank's CSR chemical glycan graphs (kbank.glycan & kbank.sugars), curated database records (kbank.db & kbank.io), and biosynthetic alignment & MPNN prediction (kbank.biosynthesis).
author: Tom Stanton
icon: lucide/book-open
comments: true
tags:
  - api
  - reference
---

# API Reference

The **KBank** API is organized into three main areas:

- **[`kbank.glycan` & `kbank.sugars`](glycan.md)** — CSR directed polysaccharide graphs ([`Glycan`][kbank.glycan.Glycan], [`Glycans`][kbank.glycan.Glycans]), monosaccharide and substituent chemical registries ([`Sugars`][kbank.sugars.Sugars]), and stereochemical enumerations ([`Anomer`][kbank.glycan.Anomer], [`Chirality`][kbank.glycan.Chirality], [`LinkageType`][kbank.glycan.LinkageType], [`ResidueType`][kbank.glycan.ResidueType]).
- **[`kbank.db` & `kbank.io`](db.md)** — Curated bacterial glycan–locus records ([`GlycanRecord`][kbank.db.models.GlycanRecord], [`GlycanRecords`][kbank.db.records.GlycanRecords]), multi-database curation ([`build_curated_glycan_records`][kbank.db.curation.build_curated_glycan_records]), and structural parsers ([`CSDBParser`][kbank.io.csdb.CSDBParser], [`IUPACParser`][kbank.io.iupac.IUPACParser], [`KEGGParser`][kbank.io.kegg.KEGGParser]).
- **[`kbank.biosynthesis`](biosynthesis.md)** — Bipartite enzyme-to-linkage alignment ([`BiosyntheticAssigner`][kbank.biosynthesis.align.BiosyntheticAssigner], [`PanGraphConsensusPrior`][kbank.biosynthesis.align.PanGraphConsensusPrior]), factorized 1D-CNN enzyme classifiers ([`BiosyntheticClassifierSuite`][kbank.biosynthesis.classifier.BiosyntheticClassifierSuite]), and MPNN *de novo* structure prediction ([`GlycanPredictor`][kbank.biosynthesis.predictor.GlycanPredictor]).
