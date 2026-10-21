# API Reference

The **KBank** API is organized into three main areas:

- **[`kbank.glycan` & `kbank.sugars`](glycan.md)** — CSR directed polysaccharide graphs (`Glycan`, `Glycans`), monosaccharide and substituent chemical registries (`Sugars`), and stereochemical enumerations (`Anomer`, `Chirality`, `LinkageType`, `ResidueType`).
- **[`kbank.db` & `kbank.io`](db.md)** — Curated bacterial glycan–locus records (`GlycanRecord`, `GlycanRecords`) and structural parsers (`CSDBParser`, `IUPACParser`).
- **[`kbank.biosynthesis`](biosynthesis.md)** — Bipartite enzyme-to-linkage alignment (`BiosyntheticAligner`, `GeneGlycanAligner`), factorized 1D-CNN enzyme classifiers (`BiosyntheticClassifierSuite`), and neuro-symbolic MPNN *de novo* structure prediction (`GlycanPredictor`).

---

::: kbank
    options:
      members: false
