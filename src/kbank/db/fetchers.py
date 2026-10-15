"""Asynchronous fetchers for external bacterial glycan databases.

Provides a lean abstract base class GlycanRecordFetcher and concrete streaming implementations
for CSDB, ECODAB, EK3D, CERMAV EPS-DB, and KEGG Glycan.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from html import unescape
from pathlib import Path

import polars as pl

from kbank.db.matcher import KaptiveLocusMatcher
from kbank.db.models import GlycanRecord
from kbank.glycan import Glycan


def _clean_html(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    return unescape(cleaned).replace("\xa0", " ").strip() if cleaned else None


async def chain_fetchers(*fetchers: AsyncIterable[GlycanRecord]) -> AsyncIterator[GlycanRecord]:
    r"""Chains multiple asynchronous GlycanRecord streams into a single continuous stream.

    Args:
        *fetchers: One or more async iterables of [`GlycanRecord`][kbank.glycan.db.models.GlycanRecord].

    Yields:
        GlycanRecord: Successive records from each fetcher in order.
    """
    for f in fetchers:
        async for record in f:
            yield record


class GlycanRecordFetcher(ABC):
    r"""Lean abstract base class for streaming GlycanRecord instances."""

    source_name: bytes = b""

    def __init__(self, matcher: KaptiveLocusMatcher | None = None) -> None:
        r"""Initializes the fetcher with a KaptiveLocusMatcher.

        Args:
            matcher: Optional [`KaptiveLocusMatcher`][kbank.glycan.db.matcher.KaptiveLocusMatcher] instance.
        """
        self.matcher = matcher or KaptiveLocusMatcher()

    @abstractmethod
    def fetch(self) -> AsyncIterator[GlycanRecord]:
        r"""Asynchronously streams GlycanRecord instances from the data source."""
        ...

    def __aiter__(self) -> AsyncIterator[GlycanRecord]:
        r"""Enables direct async iteration over the fetcher."""
        return self.fetch()


class CSDBFetcher(GlycanRecordFetcher):
    r"""Asynchronously streams bacterial carbohydrate records from CSDB.

    Supports reading from local parquet dumps or streaming from the web.
    """

    source_name: bytes = b"CSDB"

    def __init__(
        self,
        source: Path | str | None = None,
        matcher: KaptiveLocusMatcher | None = None,
        max_records: int | None = None,
    ) -> None:
        r"""Initializes the CSDB fetcher.

        Args:
            source: Optional local path to CSDB parquet dump.
            matcher: Optional KaptiveLocusMatcher instance.
            max_records: Optional record limit.
        """
        super().__init__(matcher=matcher)
        self.source = Path(source) if source else None
        self.max_records = max_records

    def _find_default_source(self) -> Path | None:
        candidates = [
            Path("data/csdb.parquet"),
            Path("notebooks/csdb.parquet"),
            Path("../data/csdb.parquet"),
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    async def fetch(self) -> AsyncIterator[GlycanRecord]:
        r"""Asynchronously streams GlycanRecord instances from CSDB.

        Yields:
            GlycanRecord: Deserialized and matched glycan record.
        """
        src = self.source or self._find_default_source()
        if src and src.is_file():
            df = await asyncio.to_thread(pl.read_parquet, src)
            count = 0
            for row in df.iter_rows(named=True):
                if self.max_records and count >= self.max_records:
                    break

                rec_id = str(row.get("record_id") or row.get("ID") or "").encode()
                bio_source = str(row.get("biological_source") or row.get("SO") or "")

                # Extract organism, serotype, strain from biological source
                org_str, sero_str, strain_str = self._parse_bio_source(bio_source)
                organism = org_str.encode()
                serotype = sero_str.encode()
                strain = strain_str.encode()

                lin_code = str(row.get("structure_linear") or row.get("ST1") or "").strip()
                structure_linear = lin_code.encode()

                # Literature & identifiers
                py_val = row.get("publication_year") or row.get("PY")
                year = int(py_val) if py_val and str(py_val).isdigit() else 0

                ref_val = str(row.get("references") or row.get("RL") or "")
                pmid_m = re.search(r"\bPMID\s*[:=]?\s*(\d+)", ref_val, re.IGNORECASE)
                pmid = int(pmid_m.group(1)) if pmid_m else 0

                doi_m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", ref_val)
                doi = doi_m.group(0).encode() if doi_m else b""

                db_val = str(row.get("cross_references") or row.get("DB") or "")
                gtc_m = re.search(r"\b(G\d{5}[A-Z]{2})\b", db_val)
                glytoucan = gtc_m.group(1).encode() if gtc_m else b""

                # Kaptive matching
                antigen_type, kaptive_locus = self.matcher.match(organism, serotype)

                # Parse glycan core
                glycan: Glycan | None = None
                if structure_linear:
                    try:
                        glycan = Glycan.parse(structure_linear, strict=False)
                    except Exception:
                        glycan = None

                yield GlycanRecord(
                    source_db=self.source_name,
                    record_id=rec_id,
                    organism=organism,
                    strain=strain,
                    serotype=serotype,
                    antigen_type=antigen_type,
                    kaptive_locus=kaptive_locus,
                    structure_raw=structure_linear,
                    structure_linear=structure_linear,
                    glycan=glycan,
                    is_repeating=True,
                    pmid=pmid,
                    publication_year=year,
                    doi=doi,
                    glytoucan_id=glytoucan,
                    notes=str(row.get("comments") or row.get("NT") or "").encode(),
                )
                count += 1
                await asyncio.sleep(0)  # Yield control to event loop

    @staticmethod
    def _parse_bio_source(bio: str) -> tuple[str, str, str]:
        if not bio:
            return ("", "", "")
        parts = bio.split(";")
        org = parts[0].strip()
        serotype = ""
        strain = ""

        # Extract serotype / serogroup with word boundaries
        sero_m = re.search(
            r"\b(?:serogroup|serotype|antigen|serovar)\s*[:=]?\s*([A-Za-z0-9\-_αβγ.]+)",
            bio,
            re.IGNORECASE,
        )
        if sero_m:
            serotype = sero_m.group(1).strip()
        else:
            code_m = re.search(
                r"\b(KL|OL|OCL|[KO])\s*[-:]?\s*(\d+[A-Za-z0-9\-_αβγ.]*)\b",
                bio,
                re.IGNORECASE,
            )
            if code_m:
                serotype = (code_m.group(1).upper() + code_m.group(2)).strip()

        strain_m = re.search(r"strain\s*[:=]?\s*([^;]+)", bio, re.IGNORECASE)
        if strain_m:
            strain = strain_m.group(1).strip()

        return (org, serotype, strain)


class ECODABFetcher(GlycanRecordFetcher):
    r"""Asynchronously streams Escherichia coli O-antigen records from ECODAB."""

    source_name: bytes = b"ECODAB"
    ECODAB_BASE = "http://nevyn.organ.su.se/ECODAB/"

    def __init__(
        self,
        source: Path | str | None = None,
        matcher: KaptiveLocusMatcher | None = None,
        max_records: int | None = None,
    ) -> None:
        r"""Initializes the ECODAB fetcher.

        Args:
            source: Optional local path to ECODAB parquet file.
            matcher: Optional KaptiveLocusMatcher instance.
            max_records: Optional record limit.
        """
        super().__init__(matcher=matcher)
        self.source = Path(source) if source else None
        self.max_records = max_records

    def _find_default_source(self) -> Path | None:
        candidates = [
            Path("notebooks/ecodab.parquet"),
            Path("data/ecodab.parquet"),
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    async def fetch(self) -> AsyncIterator[GlycanRecord]:
        r"""Asynchronously streams GlycanRecord instances from ECODAB.

        Yields:
            GlycanRecord: Deserialized and matched glycan record.
        """
        src = self.source or self._find_default_source()
        if src and src.is_file():
            df = await asyncio.to_thread(pl.read_parquet, src)
            count = 0
            for row in df.iter_rows(named=True):
                if self.max_records and count >= self.max_records:
                    break

                rec_id = str(row.get("id_serogroup") or "").encode()
                serogroup = str(row.get("serogroup") or "").strip()
                serotype_str = serogroup if serogroup.upper().startswith("O") else f"O{serogroup}"
                serotype = serotype_str.encode()
                organism = b"Escherichia coli"
                strain = str(row.get("strain") or "").encode()

                raw_struct = str(row.get("structure") or "").strip()
                structure_raw = raw_struct.encode()

                py_val = row.get("publication_year")
                year = int(py_val) if py_val and str(py_val).isdigit() else 0

                pmid_val = row.get("pmid")
                pmid = int(pmid_val) if pmid_val and str(pmid_val).isdigit() else 0

                antigen_type, kaptive_locus = self.matcher.match(organism, serotype)

                glycan: Glycan | None = None
                if structure_raw:
                    try:
                        glycan = Glycan.parse(structure_raw, strict=False)
                    except Exception:
                        glycan = None

                yield GlycanRecord(
                    source_db=self.source_name,
                    record_id=rec_id,
                    organism=organism,
                    strain=strain,
                    serotype=serotype,
                    antigen_type=antigen_type,
                    kaptive_locus=kaptive_locus,
                    structure_raw=structure_raw,
                    structure_linear=structure_raw,
                    glycan=glycan,
                    is_repeating=True,
                    pmid=pmid,
                    publication_year=year,
                    notes=str(row.get("comment") or "").encode(),
                )
                count += 1
                await asyncio.sleep(0)


class EK3DFetcher(GlycanRecordFetcher):
    r"""Asynchronously streams Escherichia coli K-antigen records from EK3D."""

    source_name: bytes = b"EK3D"
    EK3D_BASE = "https://www.iith.ac.in/EK3D/"

    def __init__(
        self,
        source: Path | str | None = None,
        matcher: KaptiveLocusMatcher | None = None,
        max_records: int | None = None,
    ) -> None:
        r"""Initializes the EK3D fetcher.

        Args:
            source: Optional local path to EK3D parquet file.
            matcher: Optional KaptiveLocusMatcher instance.
            max_records: Optional record limit.
        """
        super().__init__(matcher=matcher)
        self.source = Path(source) if source else None
        self.max_records = max_records

    def _find_default_source(self) -> Path | None:
        candidates = [
            Path("notebooks/ek3d.parquet"),
            Path("data/ek3d.parquet"),
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    async def fetch(self) -> AsyncIterator[GlycanRecord]:
        r"""Asynchronously streams GlycanRecord instances from EK3D.

        Yields:
            GlycanRecord: Deserialized and matched glycan record.
        """
        src = self.source or self._find_default_source()
        if src and src.is_file():
            df = await asyncio.to_thread(pl.read_parquet, src)
            count = 0
            for row in df.iter_rows(named=True):
                if self.max_records and count >= self.max_records:
                    break

                rec_id = str(row.get("ek_id") or "").encode()
                ag_name = str(row.get("antigen_name") or "").strip()
                serotype_str = ag_name if ag_name.upper().startswith("K") else f"K{ag_name}"
                serotype = serotype_str.encode()
                organism = b"Escherichia coli"

                group = str(row.get("biosynthetic_group") or "")
                notes = f"Group: {group}".encode() if group else b""

                pmid_val = row.get("pmid")
                pmid = int(pmid_val) if pmid_val and str(pmid_val).isdigit() else 0

                antigen_type, kaptive_locus = self.matcher.match(organism, serotype)

                yield GlycanRecord(
                    source_db=self.source_name,
                    record_id=rec_id,
                    organism=organism,
                    serotype=serotype,
                    antigen_type=antigen_type,
                    kaptive_locus=kaptive_locus,
                    structure_raw=b"",
                    structure_linear=b"",
                    glycan=None,
                    is_repeating=True,
                    pmid=pmid,
                    notes=notes,
                )
                count += 1
                await asyncio.sleep(0)


class EPSDBFetcher(GlycanRecordFetcher):
    r"""Asynchronously streams bacterial exopolysaccharide records from CERMAV EPS-DB."""

    source_name: bytes = b"EPSDB"

    def __init__(
        self,
        source: Path | str | None = None,
        matcher: KaptiveLocusMatcher | None = None,
        max_records: int | None = None,
    ) -> None:
        r"""Initializes the CERMAV EPS-DB fetcher.

        Args:
            source: Optional local path to EPSDB parquet file.
            matcher: Optional KaptiveLocusMatcher instance.
            max_records: Optional record limit.
        """
        super().__init__(matcher=matcher)
        self.source = Path(source) if source else None
        self.max_records = max_records

    def _find_default_source(self) -> Path | None:
        candidates = [
            Path("notebooks/epsdb.parquet"),
            Path("data/epsdb.parquet"),
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    async def fetch(self) -> AsyncIterator[GlycanRecord]:
        r"""Asynchronously streams GlycanRecord instances from CERMAV EPS-DB.

        Yields:
            GlycanRecord: Deserialized and matched glycan record.
        """
        src = self.source or self._find_default_source()
        if src and src.is_file():
            df = await asyncio.to_thread(pl.read_parquet, src)
            count = 0
            for row in df.iter_rows(named=True):
                if self.max_records and count >= self.max_records:
                    break

                rec_id = str(row.get("eps_id") or "").encode()
                organism = str(row.get("organism") or "").encode()
                strain = str(row.get("strain") or "").encode()

                backbone = str(row.get("backbone") or "").strip()
                branches = str(row.get("branches") or "").strip()
                raw_struct = f"{backbone} [branches: {branches}]" if branches else backbone
                structure_raw = raw_struct.encode()
                structure_linear = backbone.encode()

                py_val = row.get("publication_year")
                year = int(py_val) if py_val and str(py_val).isdigit() else 0

                # Derive serotype cues from strain or organism
                st_text = f"{row.get('strain') or ''} {row.get('organism') or ''}"
                st_m = re.search(r"\b([KO][0-9]+[A-Za-z0-9\-_.]*)\b", st_text)
                serotype = st_m.group(1).encode() if st_m else b""

                antigen_type, kaptive_locus = self.matcher.match(organism, serotype)

                glycan: Glycan | None = None
                if structure_linear:
                    try:
                        glycan = Glycan.parse(structure_linear, strict=False)
                    except Exception:
                        glycan = None

                yield GlycanRecord(
                    source_db=self.source_name,
                    record_id=rec_id,
                    organism=organism,
                    strain=strain,
                    serotype=serotype,
                    antigen_type=antigen_type,
                    kaptive_locus=kaptive_locus,
                    structure_raw=structure_raw,
                    structure_linear=structure_linear,
                    glycan=glycan,
                    is_repeating=True,
                    publication_year=year,
                    notes=str(row.get("function") or "").encode(),
                )
                count += 1
                await asyncio.sleep(0)


class KEGGFetcher(GlycanRecordFetcher):
    r"""Asynchronously streams bacterial surface glycan records from KEGG Glycan."""

    source_name: bytes = b"KEGG"

    def __init__(
        self,
        source: Path | str | None = None,
        matcher: KaptiveLocusMatcher | None = None,
        max_records: int | None = None,
    ) -> None:
        r"""Initializes the KEGG Glycan fetcher.

        Args:
            source: Optional local path to KEGG Glycan parquet file.
            matcher: Optional KaptiveLocusMatcher instance.
            max_records: Optional record limit.
        """
        super().__init__(matcher=matcher)
        self.source = Path(source) if source else None
        self.max_records = max_records

    def _find_default_source(self) -> Path | None:
        candidates = [
            Path("notebooks/kegg_glycan.parquet"),
            Path("data/kegg_glycan.parquet"),
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    async def fetch(self) -> AsyncIterator[GlycanRecord]:
        r"""Asynchronously streams GlycanRecord instances from KEGG Glycan.

        Yields:
            GlycanRecord: Deserialized and matched glycan record.
        """
        src = self.source or self._find_default_source()
        if src and src.is_file():
            df = await asyncio.to_thread(pl.read_parquet, src)
            count = 0
            for row in df.iter_rows(named=True):
                if self.max_records and count >= self.max_records:
                    break

                rec_id = str(row.get("glycan_id") or "").encode()
                name = str(row.get("name") or "").strip()
                organism = self._extract_organism_from_name(name)
                serotype = self._extract_serotype_from_name(name)

                composition = str(row.get("composition") or "")
                nodes = str(row.get("nodes") or "")
                edges = str(row.get("edges") or "")
                raw_struct = f"composition: {composition}; nodes: {nodes}; edges: {edges}"
                structure_raw = raw_struct.encode()

                gtc_id = str(row.get("glytoucan_id") or "").encode()

                pmid_val = row.get("pmid")
                pmid = int(pmid_val) if pmid_val and str(pmid_val).isdigit() else 0

                antigen_type, kaptive_locus = self.matcher.match(organism, serotype)

                yield GlycanRecord(
                    source_db=self.source_name,
                    record_id=rec_id,
                    organism=organism,
                    serotype=serotype,
                    antigen_type=antigen_type,
                    kaptive_locus=kaptive_locus,
                    structure_raw=structure_raw,
                    structure_linear=b"",
                    glycan=None,
                    is_repeating=True,
                    pmid=pmid,
                    glytoucan_id=gtc_id,
                    notes=name.encode(),
                )
                count += 1
                await asyncio.sleep(0)

    @staticmethod
    def _extract_organism_from_name(name: str) -> bytes:
        name_lower = name.lower()
        if "escherichia coli" in name_lower or "e. coli" in name_lower:
            return b"Escherichia coli"
        if "klebsiella" in name_lower:
            return b"Klebsiella pneumoniae"
        if "acinetobacter" in name_lower:
            return b"Acinetobacter baumannii"
        if "pseudomonas" in name_lower:
            return b"Pseudomonas aeruginosa"
        if "salmonella" in name_lower:
            return b"Salmonella enterica"
        if "streptococcus" in name_lower:
            return b"Streptococcus pneumoniae"
        return b"Bacteria"

    @staticmethod
    def _extract_serotype_from_name(name: str) -> bytes:
        m = re.search(r"\b([KO][0-9]+[A-Za-z0-9\-_.]*)\b", name)
        return m.group(1).encode() if m else b""
