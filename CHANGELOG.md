# Release Notes

All notable changes to **`KBank`** are documented on this page. This project adheres to [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) specification.

---

## [Unreleased]

### Added

- **CSR Chemical Glycan Graphs & Monosaccharide Registry (`kbank.glycan`, `kbank.sugars`)**:
  - Structure-of-Arrays (SoA) `Sugars` chemical registry encoding monosaccharide base rings, stereochemistry (`Anomer`, `Chirality`), ring forms, and non-stoichiometric substituents.
  - Compressed Sparse Row (CSR) directed multigraphs (`Glycan`, `Glycans`) with canonical root-invariant topological hashing and Weisfeiler-Lehman graph kernels.
- **Structural Parsers & Multi-Database Curation (`kbank.io`, `kbank.db`)**:
  - `CSDBParser`, `IUPACParser`, and `KEGGParser` for bidirectional chemical graph parsing and serialization.
  - Multi-database curation pipeline (`build_curated_glycan_records`) uniting **1,530** bacterial surface polysaccharide structures across CSDB, KEGG, ECODAB, EPS-DB, EK3D, and GlyTouCan paired *de novo* with `SeroGen` biosynthetic gene clusters (`GlycanRecords`).
- **Biosynthetic Alignment & De Novo Structure Prediction (`kbank.biosynthesis`)**:
  - `BiosyntheticAssigner` and `PanGraphConsensusPrior` Expectation-Maximization homology priors for bipartite enzyme-to-linkage assignment.
  - 10-head factorized 1D-CNN `BiosyntheticClassifierSuite` (`ClusterClassifier`) predicting donor, acceptor, anomer, linkage position, and substituent specificity from protein sequences.
  - `GlycanPredictor` combining `RelationalMPNN` message passing with a branch-and-bound Constrained Spanning Arborescence solver for *de novo* repeating-unit structure reconstruction.
- **Interactive Benchmarks & Documentation**:
  - Reproducible 16:9 Marimo slide decks (`curate_glycans.py`, `train_cluster_classifier.py`, `train_glycan_predictor.py`) and unified SeroVerse Zensical documentation.
