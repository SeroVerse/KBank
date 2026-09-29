"""Data models for multi-db bacterial glycan records.

Provides immutable dataclasses for individual glycan entries integrated from
CSDB, ECODAB, EK3D, EPSDB, and KEGG Glycan.
"""

from dataclasses import dataclass

from kbank.glycan.glycan import Glycan


@dataclass(frozen=True, slots=True)
class GlycanRecord:
    r"""Immutable record for a bacterial glycan entry across integrated databases.

    Consolidates metadata, literature provenance, structural notations, and parsed core
    representations into a standard schema. Uses byte strings for all textual fields
    to ensure zero-copy compatibility and alignment with KBank SoA containers.

    Attributes:
        source_db: Source db identifier (e.g. `b"CSDB"`, `b"ECODAB"`, `b"EK3D"`, `b"EPSDB"`, `b"KEGG"`).
        record_id: Primary identifier within the source db (e.g. `b"70"`, `b"EK26"`, `b"G00177"`).
        organism: Canonical bacterial organism name (e.g. `b"Escherichia coli"`).
        strain: Specific bacterial strain designation (e.g. `b"K12"`, `b"0-1"`).
        serotype: Serotype or antigen designation (e.g. `b"O1"`, `b"K26"`).
        antigen_type: Kaptive db keyword identifier (e.g. `b"ecoli_kps"`, `b"kpsc_k"`, `b"ab_k"`).
        kaptive_locus: Locus identifier within the Kaptive db (e.g. `b"KL1"`, `b"OL1"`).
        structure_raw: Raw structure notation from source db (linear sequence or ASCII diagram).
        structure_linear: Normalized linear sequence notation (CSDB linear code or IUPAC).
        glycan: Parsed [`Glycan`][kbank.glycan.glycan.Glycan] core instance, or `None` if unparsed.
        is_repeating: Whether the structure represents an oligo/polysaccharide repeating unit.
        pmid: PubMed ID integer, or 0 if unknown.
        publication_year: Four-digit year of primary publication, or 0 if unknown.
        doi: Digital Object Identifier byte string.
        glytoucan_id: GlyTouCan accession identifier (e.g. `b"G12345AB"`), or empty byte string.
        notes: Additional curation notes or source annotations.
    """

    source_db: bytes
    record_id: bytes
    organism: bytes
    strain: bytes = b""
    serotype: bytes = b""
    antigen_type: bytes = b""
    kaptive_locus: bytes = b""
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
        r"""Global unique identifier combining source db and primary record ID.

        Returns:
            bytes: Byte string formatted as `b"<source_db>:<record_id>"` (e.g. `b"ECODAB:70"`).
        """
        return b"%s:%s" % (self.source_db, self.record_id)

    @property
    def kaptive_key(self) -> bytes:
        r"""Composite Kaptive db locus keyword.

        Returns:
            bytes: Byte string formatted as `b"<antigen_type>:<kaptive_locus>"` if both are present,
                otherwise `kaptive_locus` or `b""`.
        """
        if self.antigen_type and self.kaptive_locus:
            return b"%s:%s" % (self.antigen_type, self.kaptive_locus)
        return self.kaptive_locus or b""
