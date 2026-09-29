"""Structure-of-Arrays (SoA) container for consolidated bacterial glycan records.

Provides GlycanRecords, a high-performance batched container supporting vectorized
filtering, boolean masking via __getitem__, embedded Glycans batch models, and
pickle-based persistence.
"""

from __future__ import annotations

import pickle
from collections.abc import AsyncIterable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from kaptive.core.collections import BatchedContainer

from kbank.glycan.db.models import GlycanRecord
from kbank.glycan.glycan import Glycan, Glycans


@dataclass(slots=True, frozen=True)
class GlycanRecords(BatchedContainer[GlycanRecord, "GlycanRecords"]):
    r"""Structure-of-Arrays (SoA) batch container for consolidated bacterial glycan records.

    Inherits from [`BatchedContainer`][kaptive.core.collections.BatchedContainer] to provide
    vectorized operations, contiguous flat NumPy arrays, embedded [`Glycans`][kbank.glycan.glycan.Glycans]
    batch representations, and seamless slice / boolean mask indexing.

    Attributes:
        source_dbs: 1D array of source db identifier byte strings (`"S16"`).
        record_ids: 1D array of primary record identifiers (`"S32"`).
        organisms: 1D array of canonical bacterial organism names (`"S64"`).
        strains: 1D array of bacterial strain designations (`"S64"`).
        serotypes: 1D array of serotype designations (`"S32"`).
        antigen_types: 1D array of Kaptive db keywords (`"S32"`).
        kaptive_loci: 1D array of matched Kaptive locus identifiers (`"S32"`).
        compound_keys: 1D array of compound keys (`b"<source_db>:<record_id>"`) (`"S48"`).
        kaptive_keys: 1D array of Kaptive reference keys (`b"<antigen_type>:<kaptive_locus>"`) (`"S64"`).
        pmids: 1D int64 array of PubMed IDs.
        years: 1D int16 array of publication years.
        is_repeating: 1D boolean array indicating whether the glycan represents a polymer repeat unit.
        glycans: Embedded [`Glycans`][kbank.glycan.glycan.Glycans] batch of all parsed glycan graphs.
        records: Tuple of individual [`GlycanRecord`][kbank.glycan.db.models.GlycanRecord] instances.
    """

    source_dbs: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    record_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    organisms: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    strains: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    serotypes: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    antigen_types: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    kaptive_loci: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    compound_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    kaptive_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    pmids: np.ndarray[tuple[int], np.dtype[np.int64]]
    years: np.ndarray[tuple[int], np.dtype[np.int16]]
    is_repeating: np.ndarray[tuple[int], np.dtype[np.bool_]]
    glycans: Glycans
    records: tuple[GlycanRecord, ...]

    def __len__(self) -> int:
        r"""Returns the total number of records in the batch container.

        Returns:
            int: Number of records.
        """
        return len(self.records)

    def __iter__(self) -> Iterator[GlycanRecord]:
        r"""Iterates over the individual GlycanRecord instances."""
        return iter(self.records)

    def __contains__(self, item: object) -> bool:
        r"""Checks membership by GlycanRecord, compound key, or Kaptive key."""
        if isinstance(item, GlycanRecord):
            return item in self.records
        if isinstance(item, bytes):
            return bool(np.any(self.compound_keys == item) or np.any(self.kaptive_keys == item))
        return False

    def __getitem__(
        self,
        item: int | slice | np.ndarray[tuple[int], Any] | Sequence[Any],
    ) -> GlycanRecord | GlycanRecords:
        r"""Access records by scalar index, slice, or boolean / integer mask.

        Args:
            item: An integer index, slice, or mask/indices array.

        Returns:
            GlycanRecord | GlycanRecords: A single [`GlycanRecord`][kbank.glycan.db.models.GlycanRecord]
                if an integer index is supplied, or a new [`GlycanRecords`][kbank.glycan.db.records.GlycanRecords]
                batch if sliced or masked.

        Raises:
            IndexError: If an integer index is out of range.
            TypeError: If an unsupported indexing type is provided.
        """
        if isinstance(item, (int, np.integer)):
            return self.records[int(item)]

        if isinstance(item, slice):
            new_records = self.records[item]
            new_glycans = self.glycans[item]
            return GlycanRecords(
                source_dbs=self.source_dbs[item],
                record_ids=self.record_ids[item],
                organisms=self.organisms[item],
                strains=self.strains[item],
                serotypes=self.serotypes[item],
                antigen_types=self.antigen_types[item],
                kaptive_loci=self.kaptive_loci[item],
                compound_keys=self.compound_keys[item],
                kaptive_keys=self.kaptive_keys[item],
                pmids=self.pmids[item],
                years=self.years[item],
                is_repeating=self.is_repeating[item],
                glycans=new_glycans,
                records=new_records,
            )

        # Handle boolean array mask or integer indices
        arr_item = np.asarray(item)
        if arr_item.dtype == bool:
            if len(arr_item) != len(self):
                raise ValueError(f"Mask length {len(arr_item)} does not match batch length {len(self)}.")
            kept_indices = np.where(arr_item)[0]
            new_records = tuple(self.records[i] for i in kept_indices)
            new_glycans = self.glycans[arr_item]
            return GlycanRecords(
                source_dbs=self.source_dbs[arr_item],
                record_ids=self.record_ids[arr_item],
                organisms=self.organisms[arr_item],
                strains=self.strains[arr_item],
                serotypes=self.serotypes[arr_item],
                antigen_types=self.antigen_types[arr_item],
                kaptive_loci=self.kaptive_loci[arr_item],
                compound_keys=self.compound_keys[arr_item],
                kaptive_keys=self.kaptive_keys[arr_item],
                pmids=self.pmids[arr_item],
                years=self.years[arr_item],
                is_repeating=self.is_repeating[arr_item],
                glycans=new_glycans,
                records=new_records,
            )

        if np.issubdtype(arr_item.dtype, np.integer):
            indices = arr_item.astype(np.int64)
            mask = np.zeros(len(self), dtype=bool)
            mask[indices] = True
            new_records = tuple(self.records[i] for i in indices)
            new_glycans = self.glycans[mask]
            return GlycanRecords(
                source_dbs=self.source_dbs[indices],
                record_ids=self.record_ids[indices],
                organisms=self.organisms[indices],
                strains=self.strains[indices],
                serotypes=self.serotypes[indices],
                antigen_types=self.antigen_types[indices],
                kaptive_loci=self.kaptive_loci[indices],
                compound_keys=self.compound_keys[indices],
                kaptive_keys=self.kaptive_keys[indices],
                pmids=self.pmids[indices],
                years=self.years[indices],
                is_repeating=self.is_repeating[indices],
                glycans=new_glycans,
                records=new_records,
            )

        raise TypeError(f"Unsupported index type: {type(item)}")

    # Vectorized filtering methods relying directly on __getitem__ masking
    def where_organism(self, organism: bytes) -> Self:
        r"""Filters records by exact bacterial organism name using `__getitem__` masking.

        Args:
            organism: Organism name byte string.

        Returns:
            GlycanRecords: Filtered sub-batch.
        """
        return self[self.organisms == organism]

    def where_serotype(self, serotype: bytes) -> Self:
        r"""Filters records by serotype designation using `__getitem__` masking.

        Args:
            serotype: Serotype designation byte string.

        Returns:
            GlycanRecords: Filtered sub-batch.
        """
        return self[self.serotypes == serotype]

    def where_kaptive_key(self, key: bytes) -> Self:
        r"""Filters records by Kaptive key (`b'<antigen_type>:<kaptive_locus>'`) using `__getitem__` masking.

        Args:
            key: Kaptive key byte string.

        Returns:
            GlycanRecords: Filtered sub-batch.
        """
        return self[self.kaptive_keys == key]

    def where_source_db(self, source: bytes) -> Self:
        r"""Filters records by source db name using `__getitem__` masking.

        Args:
            source: Source db identifier byte string (e.g. `b"ECODAB"`).

        Returns:
            GlycanRecords: Filtered sub-batch.
        """
        return self[self.source_dbs == source]

    def where_kaptive_locus(self, locus: bytes) -> Self:
        r"""Filters records by Kaptive locus using `__getitem__` masking.

        Args:
            locus: Locus identifier byte string (e.g. `b"KL1"`).

        Returns:
            GlycanRecords: Filtered sub-batch.
        """
        return self[self.kaptive_loci == locus]

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
    def kaptive_key(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `kaptive_keys`."""
        return self.kaptive_keys

    @property
    def source_db(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `source_dbs`."""
        return self.source_dbs

    @property
    def kaptive_locus(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `kaptive_loci`."""
        return self.kaptive_loci

    @property
    def compound_key(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        r"""Alias for `compound_keys`."""
        return self.compound_keys

    @classmethod
    def empty(cls) -> Self:
        r"""Constructs an empty, zero-length GlycanRecords container.

        Returns:
            GlycanRecords: An empty batch instance.
        """
        return cls(
            source_dbs=np.empty(0, dtype="S16"),
            record_ids=np.empty(0, dtype="S32"),
            organisms=np.empty(0, dtype="S64"),
            strains=np.empty(0, dtype="S64"),
            serotypes=np.empty(0, dtype="S32"),
            antigen_types=np.empty(0, dtype="S32"),
            kaptive_loci=np.empty(0, dtype="S32"),
            compound_keys=np.empty(0, dtype="S48"),
            kaptive_keys=np.empty(0, dtype="S64"),
            pmids=np.empty(0, dtype=np.int64),
            years=np.empty(0, dtype=np.int16),
            is_repeating=np.empty(0, dtype=bool),
            glycans=Glycans.empty(),
            records=(),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenates multiple GlycanRecords batches into a single container.

        Args:
            batches: Iterable of GlycanRecords instances.

        Returns:
            GlycanRecords: Combined batch container.
        """
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
            kaptive_loci=np.concatenate([b.kaptive_loci for b in batch_list]),
            compound_keys=np.concatenate([b.compound_keys for b in batch_list]),
            kaptive_keys=np.concatenate([b.kaptive_keys for b in batch_list]),
            pmids=np.concatenate([b.pmids for b in batch_list]),
            years=np.concatenate([b.years for b in batch_list]),
            is_repeating=np.concatenate([b.is_repeating for b in batch_list]),
            glycans=Glycans.concat([b.glycans for b in batch_list]),
            records=tuple(r for b in batch_list for r in b.records),
        )

    @classmethod
    def build(cls, records: Iterable[GlycanRecord]) -> Self:
        r"""Compiles an iterable of GlycanRecord instances into a Structure-of-Arrays GlycanRecords batch.

        Args:
            records: Iterable of [`GlycanRecord`][kbank.glycan.db.models.GlycanRecord] instances.

        Returns:
            GlycanRecords: Compiled batch container.
        """
        recs = tuple(records)
        n = len(recs)
        if n == 0:
            return cls.empty()

        source_dbs = np.empty(n, dtype="S16")
        record_ids = np.empty(n, dtype="S32")
        organisms = np.empty(n, dtype="S64")
        strains = np.empty(n, dtype="S64")
        serotypes = np.empty(n, dtype="S32")
        antigen_types = np.empty(n, dtype="S32")
        kaptive_loci = np.empty(n, dtype="S32")
        compound_keys = np.empty(n, dtype="S48")
        kaptive_keys = np.empty(n, dtype="S64")
        pmids = np.empty(n, dtype=np.int64)
        years = np.empty(n, dtype=np.int16)
        is_repeating = np.empty(n, dtype=bool)

        empty_g = Glycan(b"", (), ())
        glycan_objs: list[Glycan] = []

        for i, r in enumerate(recs):
            source_dbs[i] = r.source_db
            record_ids[i] = r.record_id
            organisms[i] = r.organism
            strains[i] = r.strain
            serotypes[i] = r.serotype
            antigen_types[i] = r.antigen_type
            kaptive_loci[i] = r.kaptive_locus
            compound_keys[i] = r.compound_key
            kaptive_keys[i] = r.kaptive_key
            pmids[i] = r.pmid
            years[i] = r.publication_year
            is_repeating[i] = r.is_repeating

            glycan_objs.append(r.glycan if r.glycan is not None else empty_g)

        glycans_batch = Glycans.from_list(glycan_objs)

        return cls(
            source_dbs=source_dbs,
            record_ids=record_ids,
            organisms=organisms,
            strains=strains,
            serotypes=serotypes,
            antigen_types=antigen_types,
            kaptive_loci=kaptive_loci,
            compound_keys=compound_keys,
            kaptive_keys=kaptive_keys,
            pmids=pmids,
            years=years,
            is_repeating=is_repeating,
            glycans=glycans_batch,
            records=recs,
        )

    @classmethod
    async def abuild(cls, records: AsyncIterable[GlycanRecord]) -> Self:
        r"""Asynchronously compiles an async stream of GlycanRecord instances into a GlycanRecords batch.

        Args:
            records: Async iterable of [`GlycanRecord`][kbank.glycan.db.models.GlycanRecord] instances.

        Returns:
            GlycanRecords: Compiled batch container.
        """
        collected: list[GlycanRecord] = []
        async for r in records:
            collected.append(r)
        return cls.build(collected)

    def save(self, path: Path | str) -> None:
        r"""Serializes the GlycanRecords instance to disk via pickle.

        Args:
            path: Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        r"""Deserializes a GlycanRecords instance from disk via pickle.

        Args:
            path: Source file path.

        Returns:
            GlycanRecords: Loaded batch container.

        Raises:
            TypeError: If the deserialized object is not a GlycanRecords instance.
        """
        p = Path(path)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
