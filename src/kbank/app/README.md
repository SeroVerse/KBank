# KBank
Geno-glycomic database for bacterial surface polysaccharides

## Antigen Similarity Analysis

Our systematic scan across the 285 curated antigen records revealed several distinct clusters and pairs of highly similar glycans.
These pairs fall into 5 major evolutionary and genetic mechanisms:

1. Enzymatic Loss/Gain of Function (Pseudogenization & Mobile Phenotype Islands)
2. Regioselective Linkage Shifts (Single Amino Acid Polymorphisms in Glycosyltransferases)
3. Anomeric Inversions (Inverting vs. Retaining Glycosyltransferase Gene Swaps)
4. Wzy Polymerization Frame Permutations (Same Oligosaccharide, Different Polymerase)
5. Divergent Loci Synthesizing Identical Glycans (Modular Recombination & HGT)

──────

### 1. Enzymatic Loss/Gain of Function (Phenotypes & Modifications)
These pairs possess an identical repeating unit backbone, but differ by a single decorative modification or sugar addition.

| Organism | Pair | Structural Difference | Inferred Genetic Mechanism | Kaptive Phenotype Logic |
| --- | --- | --- | --- | --- |
| K. pneumoniae (CPS) | K22 (KL22) ↔ K37 (KL22) | ± OAc(1 → 6) on β-D-Galp | Single nucleotide frameshift inactivating the O-acetyltransferase. | `{'K37': {'loci': ['KL22'], 'inactive_genes': ['atr12']}}` |
| K. pneumoniae (LPS) | O2a (OL2a) ↔ O1 (OL1) capping repeat (→ 3-α-D-Galp-(1 → 3)-β-D-Galp-(1 →) | Addition of D-galactan II | Acquisition of the wbbY and wbbZ operon on a mobile element outside the main rfb locus. | `{'O1α,2α': {'loci': ['OL2α*'], 'extra_genes': ['wbbY']}}` |
| A. baumannii (CPS) | K1 (KL1) ↔ K1-Ac (KL1) | ± OAc modification | Acquisition/retention of prophage-borne or extra acetyltransferase atr25. | `{'K1-Ac': {'loci': ['KL1'], 'extra_genes': ['atr25']}}` |
| A. baumannii (CPS) | K3 (KL3) ↔ K3-v1 (KL3) | Truncation of terminal branch | Inactivation of the glycosyltransferase gtr6. | `{'K3-v1': {'loci': ['KL3'], 'inactive_genes': ['gtr6']}}` |

──────

### 2. Regioselective Linkage Shifts (Single Amino Acid Polymorphisms)
These pairs possess the exact same monosaccharide composition, but differ by the specific hydroxyl position on the 
acceptor ring targeted by a glycosyltransferase.

#### A. Streptococcus pneumoniae 6A vs. 6B
```
6A:  -2)aDGalp(1-3)aDGlcp(1-3)aLRhap(1-3)xDRib-ol(5-P-
6B:  -2)aDGalp(1-3)aDGlcp(1-3)aLRhap(1-4)xDRib-ol(5-P-
                                      ▲
                           Linkage shift: 3 vs 4
```

Both serotypes share the exact same tetrasaccharide-alditol phosphate backbone.

Mavroidi et al. (2004) proved that the rhamnosyltransferase wciP differs by a single missense mutation (G863A), resulting in a Ser288Asn amino acid substitution. This single
residue change in the catalytic cleft shifts the regioselectivity of the transfer from the C3-OH (in 6A) to the C4-OH (in 6B) of D-ribitol-5-phosphate.

#### B. Streptococcus pneumoniae 19A vs. 19F
```
19A: -4)[Ac(1-2)]bDManpN(1-4)aDGlcp(1-3)aLRhap(1-P-
19F: -4)[Ac(1-2)]bDManpN(1-4)aDGlcp(1-2)aLRhap(1-P-
                                      ▲
                           Linkage shift: 3 vs 2
```

19A and 19F have identical monosaccharide units (`[β-D-ManpNAc, α-D-Glcp, α-L-Rhap-1-P]`). 
The glucosyltransferase wchU and rhamnosyltransferase wciP alleles specify an α(1 → 3)
linkage in 19A versus an α(1 → 2) linkage in 19F, which also requires an allele-specific wzy polymerase to 
recognize the altered repeat.

──────

### 3. Anomeric Inversions (Inverting vs. Retaining Glycosyltransferases)

In these pairs, the position and linkage carbons are identical, but the stereochemistry (α vs. β) is flipped.

#### Acinetobacter baumannii K20 vs. K21
```
K20: -2)[xRPyr(2-6:2-4)]bDGalp(1-3)[Ac(1-2)]bDGlcpN(1-4)bDGlcpA(1-3)[lS3HOBut(1-4),Ac(1-2)]bDQuipN4N(1-
K21: -2)[xRPyr(2-6:2-4)]bDGalp(1-3)[Ac(1-2)]aDGlcpN(1-4)bDGlcpA(1-3)[lS3HOBut(1-4),Ac(1-2)]bDQuipN4N(1-
                                             ▲
                                 Anomeric flip: β vs α
```

Both serotypes share a complex, highly decorated pentasaccharide containing 3-hydroxybutanoyl-substituted QuiN4N, 
D-GlcA, pyruvate-ether-linked Gal, and GlcNAc.

The GlcNAc residue linked to GlcA is β-linked in K20, but α-linked in K21. In the BGC, the locus has exchanged an 
inverting glycosyltransferase family (GT2) for a retaining glycosyltransferase family (GT4) while retaining the 
remainder of the locus synteny intact.

──────

### 4. Wzy Polymerase Frame Permutations (Same Unit, Different Polymerase)

These pairs contain 100% identical oligosaccharide repeating subunits, but are cyclically permuted because the Wzy 
polymerase links a different residue in the periplasm.

#### Klebsiella pneumoniae K14 vs. K29

```
K14: [-3)bDGalf(1-3)bDGlcp(1-4)[aLRhap(1-3),xSPyr(2-6:2-4)bDGlcp(1-2)]bDManp(1-4)bDGlcpA(1-]n
K29: [-4)bDGlcpA(1-3)bDGalf(1-3)bDGlcp(1-4)[aLRhap(1-3),xSPyr(2-6:2-4)bDGlcp(1-2)]bDManp(1-]n
```

Both assemble the exact same branched pentasaccharide precursor on undecaprenyl phosphate in the cytoplasm.

In KL14, the periplasmic polymerase wzy links the reducing-end sugar to the C3 position of Galf. In KL29, the locus 
encodes a non-homologous wzy that links to the C4 position of GlcA. The cytosolic GTs are shared homologs, but the 
divergent polymerases result in distinct serological epitopes.

──────

### 5. Divergent Loci Producing Identical Glycans (Modular Recombination in A. baumannii)

In A. baumannii, our scan identified distinct Kaptive loci that produce 100% identical chemical structures:

KL5 vs. KL90: Both synthesize the identical legionaminic acid-containing CPS:
`-3)[Ac(1-4)[Ac(1-7),Ac(1-5)]aXLegp(2-6)]aDGalp(1-6)[Ac(1-2)]aDGlcpN(1-3)[Ac(1-2)]bDGalpN(1-`

KL2 vs. KL6: Both synthesize the identical pseudaminic acid-containing CPS:
`-3)[Ac(1-5)[Ac(1-7)]aXPsep(2-6)bDGlcp(1-6)]bDGalp(1-3)[Ac(1-2)]bDGalpN(1-`

KL43 vs. KL47: Both synthesize the identical tetrasaccharide:
`-6)[Ac(1-2)]aDGlcpN(1-6)[Ac(1-2)]aDGlcpN(1-4)[bDGlcp(1-3),Ac(1-2)]aDGalpN(1-3)[Ac(1-2)]aDGlcpN(1-`

Kenyon et al. demonstrated that extensive homologous recombination and insertion sequence (IS) transposition in A. baumannii frequently exchanges flanking regulatory regions or
flippase/polymerase alleles while retaining the same enzymatic module, resulting in distinct genomic loci that produce the same antigenic glycan.
──────
### 6. Bifunctional Polymerase Switching (E. coli K1 vs. K92)

K1: Homopolymer of α(2 → 8)-linked Neu5Ac (polysialic acid).
K92: Copolymer of alternating α(2 → 8) and α(2 → 9)-linked Neu5Ac.

A single polysialyltransferase gene, neuS. In K1, NeuS is strict and exclusively forms α(2 → 8) bonds (mimicking human neural cell adhesion molecule). In K92, the evolved NeuS
allele switches linkage geometry on alternate catalytic cycles, generating the mixed α(2 → 8)/α(2 → 9) capsule.