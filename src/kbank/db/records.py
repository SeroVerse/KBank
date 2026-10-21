"""Structure-of-Arrays (SoA) batch container for bacterial glycan records and paired biosynthetic loci."""

from collections.abc import AsyncIterable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from serogen.core.collections import BatchedContainer
from serogen.core.interval import Intervals
from serogen.core.seq import GeneticCode, Sequences
from serogen.genes.models import Genes, Indels
from serogen.loci import Loci, Locus, LocusArchetype

from kbank.biosynthesis.align import BiosyntheticAssigner, GeneGlycanAssignments, PanGraphConsensusPrior
from kbank.db.models import GlycanRecord
from kbank.glycan import Glycan, Glycans
from kbank.sugars import Sugars

DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "glycans.npz"


def _loci_to_npz_dict(loci: Loci, prefix: str = "locus_") -> dict[str, np.ndarray]:
    """Serialize a `Loci` SoA batch (and its embedded `Genes`) into a flat `.npz`-ready dictionary."""
    g = loci.genes
    return {
        f"{prefix}node_types": loci.node_types,
        f"{prefix}node_offsets": loci.node_offsets,
        f"{prefix}edge_sources": loci.edge_sources,
        f"{prefix}edge_targets": loci.edge_targets,
        f"{prefix}edge_types": loci.edge_types,
        f"{prefix}edge_offsets": loci.edge_offsets,
        f"{prefix}starts": loci.intervals.starts,
        f"{prefix}ends": loci.intervals.ends,
        f"{prefix}strands": loci.intervals.strands,
        f"{prefix}contig_indices": loci.contig_indices,
        f"{prefix}archetypes": loci.archetypes,
        f"{prefix}completeness": loci.completeness,
        f"{prefix}scores": loci.scores,
        f"{prefix}is_tandem_split": loci.is_tandem_split,
        f"{prefix}names": np.char.encode(loci.names, "utf-8") if len(loci.names) > 0 else np.empty(0, dtype="S1"),
        f"{prefix}gene_starts": g.intervals.starts,
        f"{prefix}gene_ends": g.intervals.ends,
        f"{prefix}gene_strands": g.intervals.strands,
        f"{prefix}gene_contig_indices": g.contig_indices,
        f"{prefix}gene_is_partial": g.is_partial,
        f"{prefix}gene_genetic_code_id": np.array([g.genetic_code.table_id], dtype=np.int32),
        f"{prefix}gene_seq_ids": g.seqs.ids,
        f"{prefix}gene_seq_bytes": g.seqs.seqs,
        f"{prefix}gene_seq_offsets": g.seqs.offsets,
        f"{prefix}gene_seq_lengths": g.seqs.lengths,
        f"{prefix}gene_indel_positions": g.indels.positions,
        f"{prefix}gene_indel_kinds": g.indels.kinds,
        f"{prefix}gene_indel_offsets": g.indels.offsets,
        f"{prefix}gene_indel_lengths": g.indels.lengths,
    }


def _loci_from_npz_dict(data: Any, prefix: str = "locus_") -> Loci:
    """Deserialize a `Loci` SoA batch from a flat `.npz` archive dictionary."""
    files = set(data.files) if hasattr(data, "files") else set(data.keys())
    g_starts = data[f"{prefix}gene_starts"]
    n_genes = len(g_starts)
    gene_intervals = Intervals(
        starts=g_starts,
        ends=data[f"{prefix}gene_ends"],
        strands=data[f"{prefix}gene_strands"],
    )
    seq_lengths = data[f"{prefix}gene_seq_lengths"]
    gene_seqs = (
        Sequences(
            ids=data[f"{prefix}gene_seq_ids"],
            seqs=data[f"{prefix}gene_seq_bytes"],
            offsets=data[f"{prefix}gene_seq_offsets"],
            lengths=seq_lengths,
        )
        if len(seq_lengths) > 0
        else Sequences.empty()
    )
    gene_indels = Indels(
        positions=data[f"{prefix}gene_indel_positions"],
        kinds=data[f"{prefix}gene_indel_kinds"],
        offsets=data[f"{prefix}gene_indel_offsets"],
        lengths=data[f"{prefix}gene_indel_lengths"],
    )
    is_partial = (
        data[f"{prefix}gene_is_partial"].astype(np.bool_)
        if f"{prefix}gene_is_partial" in files
        else np.zeros(n_genes, dtype=np.bool_)
    )
    gc_id = int(data[f"{prefix}gene_genetic_code_id"][0]) if f"{prefix}gene_genetic_code_id" in files else 11
    genes = Genes(
        intervals=gene_intervals,
        seqs=gene_seqs,
        contig_indices=data[f"{prefix}gene_contig_indices"],
        indels=gene_indels,
        is_partial=is_partial,
        genetic_code=GeneticCode.from_table(gc_id),
    )
    node_offsets = data[f"{prefix}node_offsets"]
    n_loci = max(0, len(node_offsets) - 1)
    names_raw = data[f"{prefix}names"] if f"{prefix}names" in files else np.full(n_loci, "", dtype="U1")
    return Loci(
        node_types=data[f"{prefix}node_types"],
        node_offsets=node_offsets,
        edge_sources=data[f"{prefix}edge_sources"],
        edge_targets=data[f"{prefix}edge_targets"],
        edge_types=data[f"{prefix}edge_types"],
        edge_offsets=data[f"{prefix}edge_offsets"],
        genes=genes,
        intervals=Intervals(
            starts=data[f"{prefix}starts"],
            ends=data[f"{prefix}ends"],
            strands=data[f"{prefix}strands"],
        ),
        contig_indices=data[f"{prefix}contig_indices"],
        archetypes=data[f"{prefix}archetypes"],
        completeness=data[f"{prefix}completeness"],
        scores=data[f"{prefix}scores"],
        is_tandem_split=data[f"{prefix}is_tandem_split"],
        names=names_raw,
    )


@dataclass(slots=True, frozen=True)
class GlycanRecords(BatchedContainer[GlycanRecord, "GlycanRecords"]):
    r"""Pure Structure-of-Arrays (SoA) batch container for consolidated bacterial glycan records.

    Inherits from [`BatchedContainer`][serogen.core.collections.BatchedContainer] to provide
    vectorized operations, contiguous flat NumPy arrays, embedded [`Glycans`][kbank.glycan.Glycans],
    [`Loci`][serogen.loci.Loci], and [`GeneGlycanAssignments`][kbank.biosynthesis.GeneGlycanAssignments]
    batch representations, zero-pickle `.npz` serialization, and seamless slice / boolean / integer mask
    indexing. Individual [`GlycanRecord`][kbank.db.models.GlycanRecord] scalar views are materialized on
    demand when indexing a single row `records[i]`.

    Attributes:
        source_dbs: 1D array of source database identifier byte strings (`"S"`).
        record_ids: 1D array of primary record identifiers (`"S"`).
        organisms: 1D array of canonical bacterial organism names (`"S"`).
        strains: 1D array of bacterial strain designations (`"S"`).
        serotypes: 1D array of serotype designations (`"S"`).
        antigen_types: 1D array of surface antigen categories (`"S"`).
        assembly_ids: 1D array of linked genome assembly or nucleotide accessions (`"S"`).
        compound_keys: 1D array of compound keys (`b"<source_db>:<record_id>"`) (`"S"`).
        archetypes: 1D `int8` array of `LocusArchetype` integer IDs (`0..10`).
        alignment_scores: 1D `float32` array of geno-glycomic alignment scores in `[0.0, 1.0]`.
        pmids: 1D `int64` array of PubMed IDs.
        years: 1D `int16` array of publication years.
        is_repeating: 1D boolean array indicating whether the glycan represents a polymer repeat unit.
        glycans: Embedded [`Glycans`][kbank.glycan.Glycans] batch of all parsed glycan graphs.
        loci: Embedded [`Loci`][serogen.loci.Loci] batch of all paired biosynthetic gene clusters.
        assignments: Embedded [`GeneGlycanAssignments`][kbank.biosynthesis.GeneGlycanAssignments] ragged batch of cross-domain gene-to-glycan edges.
        structure_raws: 1D array of raw structure strings from the source database (`"S"`).
        structure_linears: 1D array of normalized linear structure strings (`"S"`).
        dois: 1D array of Digital Object Identifier byte strings (`"S"`).
        glytoucan_ids: 1D array of GlyTouCan accession byte strings (`"S"`).
        notes: 1D array of curation notes and provenance tags (`"S"`).
    """

    source_dbs: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    record_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    organisms: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    strains: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    serotypes: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    antigen_types: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    assembly_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    compound_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    archetypes: np.ndarray[tuple[int], np.dtype[np.int8]]
    alignment_scores: np.ndarray[tuple[int], np.dtype[np.float32]]
    pmids: np.ndarray[tuple[int], np.dtype[np.int64]]
    years: np.ndarray[tuple[int], np.dtype[np.int16]]
    is_repeating: np.ndarray[tuple[int], np.dtype[np.bool_]]
    glycans: Glycans
    loci: Loci
    assignments: GeneGlycanAssignments
    structure_raws: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(
        default_factory=lambda: np.empty(0, dtype="S1")
    )
    structure_linears: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(
        default_factory=lambda: np.empty(0, dtype="S1")
    )
    dois: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(
        default_factory=lambda: np.empty(0, dtype="S1")
    )
    glytoucan_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(
        default_factory=lambda: np.empty(0, dtype="S1")
    )
    notes: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(
        default_factory=lambda: np.empty(0, dtype="S1")
    )

    def __post_init__(self) -> None:
        n = len(self.source_dbs)
        if n > 0:
            for attr in ("structure_raws", "structure_linears", "dois", "glytoucan_ids", "notes"):
                arr = getattr(self, attr)
                if len(arr) == 0:
                    object.__setattr__(self, attr, np.zeros(n, dtype="S1"))

    def __len__(self) -> int:
        r"""Returns the total number of records in the batch container."""
        return len(self.source_dbs)

    def __iter__(self) -> Iterator[GlycanRecord]:
        r"""Iterates over the container, yielding each materialized `GlycanRecord` scalar view."""
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: object) -> bool:
        r"""Checks membership by `GlycanRecord`, compound key, or assembly ID."""
        if isinstance(item, GlycanRecord):
            return bool(np.any(self.compound_keys == item.compound_key))
        if isinstance(item, str):
            item = item.encode("utf-8")
        if isinstance(item, bytes):
            return bool(np.any(self.compound_keys == item) or np.any(self.assembly_ids == item))
        return False

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> GlycanRecord: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | Sequence[Any]) -> Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | Sequence[Any],
    ) -> GlycanRecord | Self:
        r"""Access a materialized `GlycanRecord` by scalar index, or slice/mask the SoA container."""
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"GlycanRecords index {item} out of range for batch of length {n}.")

            g_obj = self.glycans[idx]
            loc_obj = self.loci[idx]
            return GlycanRecord(
                source_db=bytes(self.source_dbs[idx]),
                record_id=bytes(self.record_ids[idx]),
                organism=bytes(self.organisms[idx]),
                strain=bytes(self.strains[idx]),
                serotype=bytes(self.serotypes[idx]),
                antigen_type=bytes(self.antigen_types[idx]),
                assembly_id=bytes(self.assembly_ids[idx]),
                archetype=LocusArchetype(int(self.archetypes[idx])),
                locus=loc_obj if len(loc_obj.nodes) > 0 else None,
                alignment_score=float(self.alignment_scores[idx]),
                assignments=self.assignments[idx],
                structure_raw=bytes(self.structure_raws[idx]),
                structure_linear=bytes(self.structure_linears[idx]),
                glycan=g_obj if len(g_obj.nodes) > 0 else None,
                is_repeating=bool(self.is_repeating[idx]),
                pmid=int(self.pmids[idx]),
                publication_year=int(self.years[idx]),
                doi=bytes(self.dois[idx]),
                glytoucan_id=bytes(self.glytoucan_ids[idx]),
                notes=bytes(self.notes[idx]),
            )

        if isinstance(item, slice):
            return GlycanRecords(
                source_dbs=self.source_dbs[item],
                record_ids=self.record_ids[item],
                organisms=self.organisms[item],
                strains=self.strains[item],
                serotypes=self.serotypes[item],
                antigen_types=self.antigen_types[item],
                assembly_ids=self.assembly_ids[item],
                compound_keys=self.compound_keys[item],
                archetypes=self.archetypes[item],
                alignment_scores=self.alignment_scores[item],
                pmids=self.pmids[item],
                years=self.years[item],
                is_repeating=self.is_repeating[item],
                glycans=self.glycans[item],
                loci=self.loci[item],
                assignments=self.assignments[item],
                structure_raws=self.structure_raws[item],
                structure_linears=self.structure_linears[item],
                dois=self.dois[item],
                glytoucan_ids=self.glytoucan_ids[item],
                notes=self.notes[item],
            )

        arr_item = np.asarray(item)
        if arr_item.dtype == bool:
            if len(arr_item) != len(self):
                raise ValueError(f"Mask length {len(arr_item)} does not match batch length {len(self)}.")
            return GlycanRecords(
                source_dbs=self.source_dbs[arr_item],
                record_ids=self.record_ids[arr_item],
                organisms=self.organisms[arr_item],
                strains=self.strains[arr_item],
                serotypes=self.serotypes[arr_item],
                antigen_types=self.antigen_types[arr_item],
                assembly_ids=self.assembly_ids[arr_item],
                compound_keys=self.compound_keys[arr_item],
                archetypes=self.archetypes[arr_item],
                alignment_scores=self.alignment_scores[arr_item],
                pmids=self.pmids[arr_item],
                years=self.years[arr_item],
                is_repeating=self.is_repeating[arr_item],
                glycans=self.glycans[arr_item],
                loci=self.loci[arr_item],
                assignments=self.assignments[arr_item],
                structure_raws=self.structure_raws[arr_item],
                structure_linears=self.structure_linears[arr_item],
                dois=self.dois[arr_item],
                glytoucan_ids=self.glytoucan_ids[arr_item],
                notes=self.notes[arr_item],
            )

        if np.issubdtype(arr_item.dtype, np.integer) or len(arr_item) == 0:
            indices = self._normalize_indices(arr_item)
            if len(indices) == 0:
                return self.empty()
            return GlycanRecords(
                source_dbs=self.source_dbs[indices],
                record_ids=self.record_ids[indices],
                organisms=self.organisms[indices],
                strains=self.strains[indices],
                serotypes=self.serotypes[indices],
                antigen_types=self.antigen_types[indices],
                assembly_ids=self.assembly_ids[indices],
                compound_keys=self.compound_keys[indices],
                archetypes=self.archetypes[indices],
                alignment_scores=self.alignment_scores[indices],
                pmids=self.pmids[indices],
                years=self.years[indices],
                is_repeating=self.is_repeating[indices],
                glycans=self.glycans[indices],
                loci=self.loci[indices],
                assignments=self.assignments[indices],
                structure_raws=self.structure_raws[indices],
                structure_linears=self.structure_linears[indices],
                dois=self.dois[indices],
                glytoucan_ids=self.glytoucan_ids[indices],
                notes=self.notes[indices],
            )

        raise TypeError(f"Unsupported index type: {type(item)}")

    # Vectorized filtering methods relying directly on __getitem__ masking
    def where_organism(self, organism: bytes | str) -> Self:
        r"""Filters records by exact bacterial organism name."""
        org_b = organism.encode("utf-8") if isinstance(organism, str) else organism
        return self[self.organisms == org_b]

    def where_serotype(self, serotype: bytes | str) -> Self:
        r"""Filters records by serotype designation."""
        sero_b = serotype.encode("utf-8") if isinstance(serotype, str) else serotype
        return self[self.serotypes == sero_b]

    def where_source_db(self, source: bytes | str) -> Self:
        r"""Filters records by source database name (e.g. `b"CSDB"`, `b"ECODAB"`)."""
        src_b = source.encode("utf-8") if isinstance(source, str) else source
        return self[self.source_dbs == src_b]

    def where_assembly_id(self, assembly_id: bytes | str) -> Self:
        r"""Filters records by genome assembly or nucleotide accession."""
        asm_b = assembly_id.encode("utf-8") if isinstance(assembly_id, str) else assembly_id
        return self[self.assembly_ids == asm_b]

    def where_archetype(self, archetype: LocusArchetype | int | str | bytes) -> Self:
        r"""Filters records by `LocusArchetype`."""
        arch = LocusArchetype.from_name(archetype)
        return self[self.archetypes == int(arch)]

    def where_has_locus(self) -> Self:
        r"""Filters records to those paired with a non-empty biosynthetic `Locus`."""
        has_loc = np.diff(self.loci.node_offsets) > 0
        return self[has_loc]

    # Property aliases for convenience
    @property
    def organism(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `organisms`."""
        return self.organisms

    @property
    def serotype(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `serotypes`."""
        return self.serotypes

    @property
    def source_db(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `source_dbs`."""
        return self.source_dbs

    @property
    def assembly_id(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `assembly_ids`."""
        return self.assembly_ids

    @property
    def compound_key(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `compound_keys`."""
        return self.compound_keys

    @property
    def structure_raw(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `structure_raws`."""
        return self.structure_raws

    @property
    def structure_linear(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `structure_linears`."""
        return self.structure_linears

    @property
    def doi(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `dois`."""
        return self.dois

    @property
    def glytoucan_id(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `glytoucan_ids`."""
        return self.glytoucan_ids

    @property
    def has_locus(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        r"""Boolean mask indicating which records have a non-empty paired `Locus`."""
        return np.diff(self.loci.node_offsets) > 0

    @classmethod
    def empty(cls) -> Self:
        r"""Constructs an empty, zero-length `GlycanRecords` container."""
        return cls(
            source_dbs=np.empty(0, dtype="S16"),
            record_ids=np.empty(0, dtype="S32"),
            organisms=np.empty(0, dtype="S64"),
            strains=np.empty(0, dtype="S64"),
            serotypes=np.empty(0, dtype="S32"),
            antigen_types=np.empty(0, dtype="S32"),
            assembly_ids=np.empty(0, dtype="S32"),
            compound_keys=np.empty(0, dtype="S48"),
            archetypes=np.empty(0, dtype=np.int8),
            alignment_scores=np.empty(0, dtype=np.float32),
            pmids=np.empty(0, dtype=np.int64),
            years=np.empty(0, dtype=np.int16),
            is_repeating=np.empty(0, dtype=bool),
            glycans=Glycans.empty(),
            loci=Loci.empty(),
            assignments=GeneGlycanAssignments.empty(),
            structure_raws=np.empty(0, dtype="S1"),
            structure_linears=np.empty(0, dtype="S1"),
            dois=np.empty(0, dtype="S1"),
            glytoucan_ids=np.empty(0, dtype="S1"),
            notes=np.empty(0, dtype="S1"),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenates multiple `GlycanRecords` batches into a single container."""
        batch_list = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty()
        if len(batch_list) == 1:
            return batch_list[0]

        return cls(
            source_dbs=np.concatenate([b.source_dbs for b in batch_list]),
            record_ids=np.concatenate([b.record_ids for b in batch_list]),
            organisms=np.concatenate([b.organisms for b in batch_list]),
            strains=np.concatenate([b.strains for b in batch_list]),
            serotypes=np.concatenate([b.serotypes for b in batch_list]),
            antigen_types=np.concatenate([b.antigen_types for b in batch_list]),
            assembly_ids=np.concatenate([b.assembly_ids for b in batch_list]),
            compound_keys=np.concatenate([b.compound_keys for b in batch_list]),
            archetypes=np.concatenate([b.archetypes for b in batch_list]),
            alignment_scores=np.concatenate([b.alignment_scores for b in batch_list]),
            pmids=np.concatenate([b.pmids for b in batch_list]),
            years=np.concatenate([b.years for b in batch_list]),
            is_repeating=np.concatenate([b.is_repeating for b in batch_list]),
            glycans=Glycans.concat([b.glycans for b in batch_list]),
            loci=Loci.concat([b.loci for b in batch_list]),
            assignments=GeneGlycanAssignments.concat([b.assignments for b in batch_list]),
            structure_raws=np.concatenate([b.structure_raws for b in batch_list]),
            structure_linears=np.concatenate([b.structure_linears for b in batch_list]),
            dois=np.concatenate([b.dois for b in batch_list]),
            glytoucan_ids=np.concatenate([b.glytoucan_ids for b in batch_list]),
            notes=np.concatenate([b.notes for b in batch_list]),
        )

    @classmethod
    def build(cls, records: Iterable[GlycanRecord]) -> Self:
        r"""Compiles an iterable of `GlycanRecord` instances into a pure SoA `GlycanRecords` batch."""
        raw_recs = list(records)
        n = len(raw_recs)
        if n == 0:
            return cls.empty()

        assigner = BiosyntheticAssigner()

        source_dbs = np.empty(n, dtype="S16")
        record_ids = np.empty(n, dtype="S32")
        organisms = np.empty(n, dtype="S64")
        strains = np.empty(n, dtype="S64")
        serotypes = np.empty(n, dtype="S32")
        antigen_types = np.empty(n, dtype="S32")
        assembly_ids = np.empty(n, dtype="S32")
        compound_keys = np.empty(n, dtype="S48")
        archetypes = np.empty(n, dtype=np.int8)
        alignment_scores = np.empty(n, dtype=np.float32)
        pmids = np.empty(n, dtype=np.int64)
        years = np.empty(n, dtype=np.int16)
        is_repeating = np.empty(n, dtype=bool)

        structure_raw_list: list[bytes] = []
        structure_linear_list: list[bytes] = []
        doi_list: list[bytes] = []
        glytoucan_id_list: list[bytes] = []
        notes_list: list[bytes] = []

        empty_g = Glycan(b"", (), ())
        empty_loc = Locus(nodes=(), edges=())
        glycan_objs: list[Glycan] = []
        locus_objs: list[Locus] = []
        assign_lists: list[tuple[Any, ...]] = []

        for i, r in enumerate(raw_recs):
            if r.locus is not None and len(r.locus.nodes) > 0:
                loc = r.locus.canonicalize_strand()
                if not loc.name:
                    label = (r.serotype or r.strain or r.record_id).decode("utf-8", errors="replace").strip()
                    if label:
                        loc = loc.with_name(label)
                if loc is not r.locus:
                    r = replace(r, locus=loc)
            raw_recs[i] = r

            source_dbs[i] = r.source_db
            record_ids[i] = r.record_id
            organisms[i] = r.organism
            strains[i] = r.strain
            serotypes[i] = r.serotype
            antigen_types[i] = r.antigen_type
            assembly_ids[i] = r.assembly_id
            compound_keys[i] = r.compound_key
            archetypes[i] = int(r.archetype)
            alignment_scores[i] = float(r.alignment_score)
            pmids[i] = r.pmid
            years[i] = r.publication_year
            is_repeating[i] = r.is_repeating

            structure_raw_list.append(r.structure_raw)
            structure_linear_list.append(r.structure_linear)
            doi_list.append(r.doi)
            glytoucan_id_list.append(r.glytoucan_id)
            notes_list.append(r.notes)

            glycan_objs.append(r.glycan if r.glycan is not None else empty_g)
            locus_objs.append(r.locus if r.locus is not None else empty_loc)

        # 1. Cluster genes via MCL across all paired loci FIRST so every GeneNode carries a homology cluster_id
        loci_batch = Loci.from_list(locus_objs)
        if (
            len(loci_batch.node_types) > 0
            and len(loci_batch.genes) == len(loci_batch.node_types)
            and bool(np.all(loci_batch.cluster_ids == 0))
        ):
            loci_batch = loci_batch.cluster_genes(
                mode="mcl",
                min_identity=28.0,
                min_bitscore=40.0,
                inflation=2.0,
                by_role=True,
            )

        # 2. Fit cross-record PanGraphConsensusPrior & rotate glycans to biological reducing ends
        pan_prior, rotated_glycans = PanGraphConsensusPrior.fit(
            glycan_objs,
            loci_batch,
            archetypes,
            is_repeating,
            assigner=assigner,
        )
        if len(pan_prior.rescued_loci) == len(raw_recs):
            loci_batch = Loci.from_list(pan_prior.rescued_loci)

        # 3. Complete locus modifiers and resolve pan-graph-informed GeneGlycanAssignments
        final_glycans: list[Glycan] = []
        for i, r in enumerate(raw_recs):
            gly = rotated_glycans[i]
            loc = loci_batch[i] if len(loci_batch) > i else empty_loc
            completed_mod_genes: dict[int, int] | None = None

            if len(loc.nodes) > 0 and len(gly.nodes) > 0:
                comp_gly, comp_mods, comp_map = assigner.complete_glycan_modifiers(gly, loc)
                if comp_mods:
                    gly = comp_gly
                    completed_mod_genes = comp_map
                    notes_b = notes_list[i]
                    if b"[locus_completed=" not in notes_b:
                        tag = b" [locus_completed=" + b",".join(b"+" + m for m in comp_mods) + b"]"
                        notes_list[i] = notes_b + tag

            final_glycans.append(gly)
            if not r.assignments and len(gly.nodes) > 0:
                inferred = assigner.assign_record(
                    gly,
                    loc if len(loc.nodes) > 0 else None,
                    archetype=r.archetype,
                    is_repeating=r.is_repeating,
                    completed_modifier_genes=completed_mod_genes,
                    pan_prior=pan_prior,
                )
                assign_lists.append(inferred)
            else:
                assign_lists.append(r.assignments)

        glycans_batch = Glycans.from_list(final_glycans)
        assignments_batch = GeneGlycanAssignments.from_list(assign_lists)

        return cls(
            source_dbs=source_dbs,
            record_ids=record_ids,
            organisms=organisms,
            strains=strains,
            serotypes=serotypes,
            antigen_types=antigen_types,
            assembly_ids=assembly_ids,
            compound_keys=compound_keys,
            archetypes=archetypes,
            alignment_scores=alignment_scores,
            pmids=pmids,
            years=years,
            is_repeating=is_repeating,
            glycans=glycans_batch,
            loci=loci_batch,
            assignments=assignments_batch,
            structure_raws=np.array(structure_raw_list, dtype="S"),
            structure_linears=np.array(structure_linear_list, dtype="S"),
            dois=np.array(doi_list, dtype="S"),
            glytoucan_ids=np.array(glytoucan_id_list, dtype="S"),
            notes=np.array(notes_list, dtype="S"),
        )

    @classmethod
    async def abuild(cls, records: AsyncIterable[GlycanRecord]) -> Self:
        r"""Asynchronously compiles an async stream of `GlycanRecord` instances into a `GlycanRecords` batch."""
        collected: list[GlycanRecord] = []
        async for r in records:
            collected.append(r)
        return cls.build(collected)

    def save(self, path: Path | str | None = None) -> Path:
        r"""Serializes the `GlycanRecords` instance to disk as a compressed NumPy `.npz` archive (zero pickle)."""
        p = Path(path) if path is not None else DEFAULT_DB_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_name(f".{p.name}.{os.getpid()}.tmp.npz")

        np.savez_compressed(
            tmp_p,
            # Record scalar columns
            source_dbs=self.source_dbs,
            record_ids=self.record_ids,
            organisms=self.organisms,
            strains=self.strains,
            serotypes=self.serotypes,
            antigen_types=self.antigen_types,
            assembly_ids=self.assembly_ids,
            compound_keys=self.compound_keys,
            archetypes=self.archetypes,
            alignment_scores=self.alignment_scores,
            pmids=self.pmids,
            years=self.years,
            is_repeating=self.is_repeating,
            structure_raws=self.structure_raws,
            structure_linears=self.structure_linears,
            dois=self.dois,
            glytoucan_ids=self.glytoucan_ids,
            notes=self.notes,
            # Glycans SoA
            glycan_node_types=self.glycans.node_types,
            glycan_node_offsets=self.glycans.node_offsets,
            glycan_edge_sources=self.glycans.edge_sources,
            glycan_edge_targets=self.glycans.edge_targets,
            glycan_edge_types=self.glycans.edge_types,
            glycan_edge_offsets=self.glycans.edge_offsets,
            glycan_canonicals=self.glycans.canonicals,
            # Loci & Genes SoA
            **_loci_to_npz_dict(self.loci, prefix="locus_"),
            # GeneGlycanAssignments SoA
            assignment_gene_indices=self.assignments.gene_indices,
            assignment_roles=self.assignments.roles,
            assignment_bipartite_indices=self.assignments.bipartite_indices,
            assignment_target_types=self.assignments.target_types,
            assignment_components=self.assignments.components,
            assignment_edge_indices=self.assignments.edge_indices,
            assignment_evidences=self.assignments.evidences,
            assignment_scores=self.assignments.scores,
            assignment_offsets=self.assignments.offsets,
            assignment_lengths=self.assignments.lengths,
        )
        os.replace(tmp_p, p)
        return p

    @classmethod
    def load(cls, path: Path | str | None = None) -> Self:
        r"""Deserializes a `GlycanRecords` instance from a compressed NumPy `.npz` archive (zero pickle, zero eager Python loop)."""
        p = Path(path) if path is not None else DEFAULT_DB_PATH
        with np.load(p, allow_pickle=False) as data:
            source_dbs = data["source_dbs"]
            n = len(source_dbs)
            if n == 0:
                return cls.empty()

            record_ids = data["record_ids"]
            organisms = data["organisms"]
            strains = data["strains"]
            serotypes = data["serotypes"]
            antigen_types = data["antigen_types"]
            assembly_ids = data["assembly_ids"]
            compound_keys = data["compound_keys"]
            archetypes = data["archetypes"]
            alignment_scores = data["alignment_scores"]
            pmids = data["pmids"]
            years = data["years"]
            is_repeating = data["is_repeating"]

            structure_raws = data["structure_raws"]
            structure_linears = data["structure_linears"]
            dois = data["dois"]
            glytoucan_ids = data["glytoucan_ids"]
            notes = data["notes"]

            glycans = Glycans(
                node_types=data["glycan_node_types"],
                node_offsets=data["glycan_node_offsets"],
                edge_sources=data["glycan_edge_sources"],
                edge_targets=data["glycan_edge_targets"],
                edge_types=data["glycan_edge_types"],
                edge_offsets=data["glycan_edge_offsets"],
                canonicals=data["glycan_canonicals"],
            )

            if "locus_gene_starts" in data.files:
                loci = _loci_from_npz_dict(data, prefix="locus_")
            else:
                gene_intervals = Intervals(
                    starts=data["gene_starts"],
                    ends=data["gene_ends"],
                    strands=data["gene_strands"],
                )
                gene_seqs = Sequences(
                    ids=data["gene_seq_ids"],
                    seqs=data["gene_seq_bytes"],
                    offsets=data["gene_seq_offsets"],
                    lengths=data["gene_seq_lengths"],
                )
                gene_indels = Indels(
                    positions=data["gene_indel_positions"],
                    kinds=data["gene_indel_kinds"],
                    offsets=data["gene_indel_offsets"],
                    lengths=data["gene_indel_lengths"],
                )
                genes = Genes(
                    intervals=gene_intervals,
                    seqs=gene_seqs if len(gene_seqs) > 0 else None,
                    contig_indices=data["gene_contig_indices"],
                    indels=gene_indels,
                )
                loci = Loci(
                    node_types=data["locus_node_types"],
                    node_offsets=data["locus_node_offsets"],
                    edge_sources=data["locus_edge_sources"],
                    edge_targets=data["locus_edge_targets"],
                    edge_types=data["locus_edge_types"],
                    edge_offsets=data["locus_edge_offsets"],
                    genes=genes,
                    intervals=Intervals(
                        starts=data["locus_starts"],
                        ends=data["locus_ends"],
                        strands=data["locus_strands"],
                    ),
                    contig_indices=data["locus_contig_indices"],
                    archetypes=data["locus_archetypes"],
                    completeness=data["locus_completeness"],
                    scores=data["locus_scores"],
                    is_tandem_split=data["locus_is_tandem_split"],
                )

            has_saved_assignments = "assignment_offsets" in data.files
            if has_saved_assignments:
                assignments = GeneGlycanAssignments(
                    gene_indices=data["assignment_gene_indices"],
                    roles=data["assignment_roles"],
                    bipartite_indices=data["assignment_bipartite_indices"],
                    target_types=data["assignment_target_types"],
                    components=data["assignment_components"],
                    edge_indices=data["assignment_edge_indices"],
                    evidences=data["assignment_evidences"],
                    scores=data["assignment_scores"],
                    offsets=data["assignment_offsets"],
                    lengths=data["assignment_lengths"],
                )
            else:
                assigner = BiosyntheticAssigner()
                fallback_lists = [
                    assigner.assign_record(
                        glycans[i] if len(glycans[i].nodes) > 0 else None,
                        loci[i] if len(loci[i].nodes) > 0 else None,
                        archetype=LocusArchetype(int(archetypes[i])),
                        is_repeating=bool(is_repeating[i]),
                    )
                    for i in range(n)
                ]
                assignments = GeneGlycanAssignments.from_list(fallback_lists)

            return cls(
                source_dbs=source_dbs,
                record_ids=record_ids,
                organisms=organisms,
                strains=strains,
                serotypes=serotypes,
                antigen_types=antigen_types,
                assembly_ids=assembly_ids,
                compound_keys=compound_keys,
                archetypes=archetypes,
                alignment_scores=alignment_scores,
                pmids=pmids,
                years=years,
                is_repeating=is_repeating,
                glycans=glycans,
                loci=loci,
                assignments=assignments,
                structure_raws=structure_raws,
                structure_linears=structure_linears,
                dois=dois,
                glytoucan_ids=glytoucan_ids,
                notes=notes,
            )

    @classmethod
    def pretrained(cls, path: Path | str | None = None) -> Self:
        r"""Loads the bundled curated `GlycanRecords` database (`src/kbank/data/glycans.npz`)."""
        return cls.load(path=path)

    def to_arrow(self) -> object:
        r"""Exports the `GlycanRecords` batch as a `pyarrow.Table`."""
        import pyarrow as pa

        rows: list[dict[str, Any]] = []
        for r in self:
            g = r.glycan
            loc = r.locus
            sugar_names: list[str] = []
            n_nodes = 0
            n_edges = 0
            if g is not None and len(g.nodes) > 0:
                n_nodes = len(g.nodes)
                n_edges = len(g.edges)
                sugar_names = [
                    Sugars.names[node.sugar].decode("ascii", errors="replace")
                    for node in g.nodes
                    if not node.is_modifier
                ]
            assign_structs = [
                {
                    "gene_index": int(a.gene_index),
                    "role": a.role.name,
                    "role_abbrev": a.role.abbreviation,
                    "target_type": a.target_type.value,
                    "target_index": int(a.target_index),
                    "bipartite_index": int(a.bipartite_glycan_index),
                    "component": a.component_label,
                    "evidence": a.evidence.name,
                    "score": float(a.score),
                }
                for a in r.assignments
            ]
            rows.append(
                {
                    "compound_key": r.compound_key.decode("utf-8", errors="replace"),
                    "source_db": r.source_db.decode("utf-8", errors="replace"),
                    "record_id": r.record_id.decode("utf-8", errors="replace"),
                    "organism": r.organism.decode("utf-8", errors="replace"),
                    "strain": r.strain.decode("utf-8", errors="replace"),
                    "serotype": r.serotype.decode("utf-8", errors="replace"),
                    "antigen_type": r.antigen_type.decode("utf-8", errors="replace"),
                    "assembly_id": r.assembly_id.decode("utf-8", errors="replace"),
                    "archetype": r.archetype.name,
                    "archetype_category": r.archetype.category.value,
                    "has_locus": r.has_locus,
                    "locus_num_genes": len(loc.nodes) if loc is not None else 0,
                    "locus_completeness": float(loc.completeness) if loc is not None else 0.0,
                    "locus_score": float(loc.score) if loc is not None else 0.0,
                    "alignment_score": float(r.alignment_score),
                    "num_assignments": r.num_assignments,
                    "num_aligned_assignments": r.num_aligned_assignments,
                    "num_rule_assignments": r.num_rule_assignments,
                    "gene_glycan_assignments": r.format_assignments(),
                    "assignments": assign_structs,
                    "num_residues": len(sugar_names),
                    "num_nodes": n_nodes,
                    "num_linkages": n_edges,
                    "monosaccharides": ", ".join(sugar_names),
                    "is_repeating": bool(r.is_repeating),
                    "structure_linear": r.structure_linear.decode("utf-8", errors="replace"),
                    "structure_raw": r.structure_raw.decode("utf-8", errors="replace"),
                    "pmid": int(r.pmid),
                    "publication_year": int(r.publication_year),
                    "doi": r.doi.decode("utf-8", errors="replace"),
                    "glytoucan_id": r.glytoucan_id.decode("utf-8", errors="replace"),
                    "notes": r.notes.decode("utf-8", errors="replace"),
                }
            )
        return pa.Table.from_pylist(rows)

    def to_parquet(self, path: Path | str) -> Path:
        r"""Writes the `GlycanRecords` batch to a Parquet file via `pyarrow.parquet`."""
        import pyarrow.parquet as pq

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.to_arrow(), p, compression="zstd")
        return p
