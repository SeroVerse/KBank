---
title: Architecture & Algorithms
subtitle: CSR Glycans, EM Aligner & Relational MPNN
description: Deep dive into KBank's Compressed Sparse Row (CSR) chemical polysaccharide graphs, pan-graph EM enzyme-to-linkage alignment, 10-head 1D-CNN classifiers, and Relational MPNN arborescence solver.
author: Tom Stanton
icon: lucide/layers
status: new
comments: true
tags:
  - architecture
  - algorithms
  - glycomics
  - mpnn
  - alignment
---

# Architecture & Algorithms

**KBank** is architected on the same zero-copy Structure-of-Arrays (SoA) and Compressed Sparse Row (CSR) principles as [`SeroGen`](https://seroverse.github.io/SeroGen/), extending genomic interval and sequence containers to chemical polysaccharide graphs and biochemically constrained structure prediction.

## 1. Chemical Monosaccharide Registry & CSR Glycan Graphs

- **`kbank.sugars.Sugars`**: Immutable SoA chemical registry encoding monosaccharide base rings, stereochemical configurations (`Anomer`, `Chirality`), ring forms, and non-stochiometric substituents ($O$-acetyl, pyruvyl, lactyl, formyl, phosphate, sulfate).
- **`kbank.glycan.Glycan` & `Glycans`**: Packed CSR directed multigraphs representing repeating-unit oligosaccharides, side-chain branches, and polymer backbone (`Wzy`) linkages, supporting canonical root-invariant topological hashing and Weisfeiler-Lehman subtree kernels.

## 2. Pan-Graph Expectation-Maximization Enzyme-to-Linkage Alignment

- **`kbank.biosynthesis.align.BiosyntheticAssigner`**: Formulates enzyme-to-linkage assignment as a minimum-cost bipartite matching problem (`LOCUS_SUGAR_EXT`) coupling:
  - Biochemical rule compatibility (`BiosyntheticRuleRegistry`),
  - Multi-record pan-graph EM homology priors (`PanGraphConsensusPrior`),
  - Syntenic operon transition penalties, and
  - Biological Und-PP reducing-end frame rotation.

## 3. *De Novo* Polysaccharide Structure Prediction (`GlycanPredictor`)

1. **Operon Precursor Pool Gating**: Identifies nucleotide-sugar biosynthesis pathways present in the locus (e.g., rhamnose `rmlABCD`, fucose `gmd`/`fcl`, uronic acid `ugd`).
2. **Factorized 1D-CNN Sequence Classifiers (`BiosyntheticClassifierSuite`)**: Predicts GT donor specificity, stereochemical mechanism (retaining vs. inverting), `Wzy` polymerase linkage, and initiating transferase directly from protein sequence.
3. **Relational Message-Passing Neural Network (`RelationalMPNN`)**: Refines pairwise donor-acceptor linkage probabilities over the candidate sugar-enzyme graph.
4. **Constrained Spanning Arborescence Solver**: Extracts the top-$K$ biochemically valid repeating-unit trees via Numba-accelerated branch-and-bound search.
