"""Data models for multi-database bacterial glycan records and de novo SeroGen locus pairings.

Provides immutable dataclasses for individual glycan entries integrated from
CSDB, ECODAB, EK3D, EPSDB, and KEGG Glycan, paired with de novo predicted
biosynthetic gene clusters (`Locus`, `LocusArchetype`) from `serogen`.
"""

from dataclasses import dataclass

from serogen.loci import Locus, LocusArchetype

from kbank.biosynthesis.align import GeneGlycanAssignment
from kbank.glycan import Glycan


@dataclass(frozen=True, slots=True)
class GlycanRecord:
    r"""Immutable record for a bacterial surface glycan entry across integrated databases.

    Consolidates taxonomic metadata, literature provenance, structural notations, parsed
    [`Glycan`][kbank.glycan.Glycan] graphs, de novo matched [`Locus`][serogen.loci.Locus]
    gene clusters from `serogen`, and resolved [`GeneGlycanAssignment`][kbank.biosynthesis.GeneGlycanAssignment]
    bipartite edges. Uses byte strings for textual fields to ensure zero-copy compatibility with
    Structure-of-Arrays (SoA) containers.

    Attributes:
        source_db: Source database identifier (`b"CSDB"`, `b"ECODAB"`, `b"EK3D"`, `b"EPSDB"`, `b"KEGG"`).
        record_id: Primary identifier within the source database (e.g. `b"70"`, `b"K1"`, `b"G00525"`).
        organism: Canonical bacterial organism name (e.g. `b"Escherichia coli"`).
        strain: Specific bacterial strain designation (e.g. `b"K12"`, `b"ATCC 43816"`).
        serotype: Serotype or antigen designation (e.g. `b"O1"`, `b"K2"`).
        antigen_type: Surface polysaccharide antigen class (`b"CPS"`, `b"O-antigen"`, `b"EPS"`, `b"WTA"`, `b"LPS"`).
        assembly_id: Linked genome assembly or nucleotide accession (e.g. `b"GCF_000005845.2"`, `b"AB811615"`).
        archetype: Biosynthetic locus archetype (`LocusArchetype`) matched or inferred for this glycan.
        locus: Matched [`Locus`][serogen.loci.Locus] biosynthetic gene cluster, or `None` if unpaired.
        alignment_score: Combined geno-glycomic alignment score in `[0.0, 1.0]` from `BiosyntheticAssigner`.
        assignments: Resolved gene-to-glycan component assignments (`GRAPH_ALIGNED` or `RULE_INFERRED`).
        structure_raw: Raw structure notation from the source database.
        structure_linear: Normalized linear sequence notation (CSDB linear code or IUPAC condensed).
        glycan: Parsed [`Glycan`][kbank.glycan.Glycan] core graph instance, or `None` if unparsed.
        is_repeating: Whether the structure represents a repeating polymer unit.
        pmid: PubMed ID integer, or `0` if unknown.
        publication_year: Four-digit year of primary publication, or `0` if unknown.
        doi: Digital Object Identifier byte string.
        glytoucan_id: GlyTouCan accession identifier (e.g. `b"G12345AB"`), or empty byte string.
        notes: Additional curation notes or cross-reference annotations.
    """

    source_db: bytes
    record_id: bytes
    organism: bytes
    strain: bytes = b""
    serotype: bytes = b""
    antigen_type: bytes = b""
    assembly_id: bytes = b""
    archetype: LocusArchetype = LocusArchetype.UNKNOWN
    locus: Locus | None = None
    alignment_score: float = 0.0
    assignments: tuple[GeneGlycanAssignment, ...] = ()
    structure_raw: bytes = b""
    structure_linear: bytes = b""
    glycan: Glycan | None = None
    is_repeating: bool = True
    pmid: int = 0
    publication_year: int = 0
    doi: bytes = b""
    glytoucan_id: bytes = b""
    notes: bytes = b""

    @property
    def compound_key(self) -> bytes:
        r"""Global unique identifier combining source database and primary record ID.

        Returns:
            bytes: Byte string formatted as `b"<source_db>:<record_id>"` (e.g. `b"ECODAB:70"`).
        """
        return b"%s:%s" % (self.source_db, self.record_id)

    @property
    def has_locus(self) -> bool:
        r"""True if this glycan record is paired with a non-empty biosynthetic `Locus`."""
        return self.locus is not None and len(self.locus.nodes) > 0

    @property
    def num_assignments(self) -> int:
        r"""Total number of gene-to-glycan component assignments on this record."""
        return len(self.assignments)

    @property
    def num_aligned_assignments(self) -> int:
        r"""Number of graph-aligned ORF-to-glycan component assignments (`gene_index >= 0`)."""
        return sum(1 for a in self.assignments if a.is_aligned)

    @property
    def num_pan_consensus_assignments(self) -> int:
        r"""Number of assignments resolved by multi-record pan-graph homology or synteny consensus."""
        return sum(1 for a in self.assignments if a.is_pan_consensus)

    @property
    def num_rule_assignments(self) -> int:
        r"""Number of stoichiometric rule-inferred `GeneRole`-to-glycan component assignments."""
        return sum(1 for a in self.assignments if not a.is_aligned)

    def format_assignments(self) -> str:
        r"""Compact human-readable summary string of all gene-to-glycan assignments."""
        return "; ".join(a.format_short() for a in self.assignments)
