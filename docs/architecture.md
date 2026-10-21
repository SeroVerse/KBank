# Architecture & Algorithms

**KBank** is architected on the same zero-copy Structure-of-Arrays (SoA) and Compressed Sparse Row (CSR) principles as [`SeroGen`](https://seroverse.github.io/SeroGen/), extending genomic interval and sequence containers to chemical polysaccharide graphs and neuro-symbolic biosynthetic inference.

## 1. Chemical Monosaccharide Registry & CSR Glycan Graphs

- **`kbank.sugars.Sugars`**: Immutable SoA chemical registry encoding monosaccharide base rings, stereochemical configurations (`Anomer`, `Chirality`), ring forms, and non-stochiometric substituents ($O$-acetyl, pyruvyl, lactyl, formyl, phosphate, sulfate).
- **`kbank.glycan.Glycan` & `Glycans`**: Packed CSR directed multigraphs representing repeating-unit oligosaccharides, side-chain branches, and polymer backbone (`Wzy`) linkages, supporting canonical root-invariant topological hashing and Weisfeiler-Lehman subtree kernels.

## 2. Pan-Graph Expectation-Maximization Enzyme-to-Linkage Alignment

- **`kbank.biosynthesis.BiosyntheticAligner`**: Formulates enzyme-to-linkage assignment as a minimum-cost bipartite matching problem (`LOCUS_SUGAR_EXT`) coupling:
  - Biochemical rule compatibility (`BiosyntheticRuleRegistry`),
  - Multi-record pan-graph EM homology priors (`PanGraphConsensusPrior`),
  - Syntenic operon transition penalties, and
  - Biological Und-PP reducing-end frame rotation.

## 3. Neuro-Symbolic *De Novo* Structure Prediction (`GlycanPredictor`)

1. **Operon Precursor Pool Gating**: Identifies nucleotide-sugar biosynthesis pathways present in the locus (e.g., rhamnose `rmlABCD`, fucose `gmd`/`fcl`, uronic acid `ugd`).
2. **Factorized 1D-CNN Sequence Classifiers (`BiosyntheticClassifierSuite`)**: Predicts GT donor specificity, stereochemical mechanism (retaining vs. inverting), `Wzy` polymerase linkage, and initiating transferase directly from protein sequence.
3. **Relational Message-Passing Neural Network (`RelationalMPNN`)**: Refines pairwise donor-acceptor linkage probabilities over the candidate sugar-enzyme graph.
4. **Constrained Spanning Arborescence Solver**: Extracts the top-$K$ biochemically valid repeating-unit trees via Numba-accelerated branch-and-bound search.
