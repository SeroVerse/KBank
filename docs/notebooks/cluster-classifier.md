---
title: 10-Head Cluster Classifiers
subtitle: EM Self-Training & Sequence Embeddings
description: Two-pass Expectation-Maximization self-training, 5-fold cross-validation, and homology-cluster log-probability pooling for BiosyntheticClassifierSuite.
author: Tom Stanton
icon: lucide/cpu
status: new
comments: true
tags:
  - benchmarks
  - cnn
  - glycosyltransferases
  - em
---

# 10-Head 1D-CNN Cluster Classifiers (`BiosyntheticClassifierSuite`)

This presentation documents the 2-pass Expectation-Maximization (EM) self-training, 5-fold sequence-stratified cross-validation, homology-cluster log-probability pooling, and 1D-CNN sequence embedding geometry of [`ClusterClassifier`][kbank.biosynthesis.classifier.ClusterClassifier] and [`BiosyntheticClassifierSuite`][kbank.biosynthesis.classifier.BiosyntheticClassifierSuite] (`src/kbank/data/cluster_classifiers.npz`).

<div class="serogen-slide-frame">
  <iframe src="assets/train_cluster_classifier.html" title="KBank 10-Head 1D-CNN Cluster Classifiers Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **10 Factorized Enzymatic Specificity Heads**: Jointly predicts glycosyltransferase `donor_linkage`, `donor`, `linkage`, `acceptor_sugar`, `acceptor_motif`, `is_backbone`, catalytic `mechanism` (retaining vs. inverting), `initiator_sugar` (`WbaP`/`WcaJ`/`WecA`/`Itr`), `wzy_linkage` polymerase specificity, and `modifier_linkage` regioselectivity directly from amino acid sequences.
- **2-Pass EM Self-Training & Warm-Started Multi-Scale 1D-CNN**: Warm-starts multi-scale `(5, 9, 13)` convolutional filters from `SeroGen`'s [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] backbone and refines non-locked record alignments in an E-step before refitting all 10 heads (`src/kbank/data/cluster_classifiers.npz`).
- **Homology-Cluster Pooling (`predict_cluster_proba`)**: Combining sequence-level CNN log-probabilities across MCL homology clusters boosts reaction specificity accuracy over single-sequence CV across all 10 enzymatic heads.

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair marimo run scripts/train_cluster_classifier.py

# Edit notebook cells
uv run --with marimo --with altair marimo edit scripts/train_cluster_classifier.py
```
