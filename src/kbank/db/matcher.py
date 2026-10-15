"""Programmatic locus and serotype matching against Kaptive databases.

Routes bacterial glycan records by organism taxonomy and antigen characteristics,
resolving specific Kaptive loci and db keywords via metadata prefix properties,
two-tier exact matching, and numeric run comparison fallbacks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from kaptive.db.core import Database


class KaptiveLocusMatcher:
    r"""Programmatic matcher that links external bacterial glycan records to Kaptive reference databases.

    Inspects installed Kaptive reference databases (e.g., `ab_k`, `ab_o`, `kpsc_k`, `kpsc_o`,
    `ecoli_kps`) via [`DatabaseManager`][kaptive.db.manager.DatabaseManager], examines their
    organism, antigen, and prefix metadata, and maps input serotypes or locus hints using
    exact matches, subvariant suffix stripping, and numeric run comparisons.
    """

    # Subvariant suffix pattern: e.g. "-I", "-II", "-A", "-A1", ".1", " A", " a"
    _SUBVARIANT_RE = re.compile(rb"[\s\-_.]+(?:[IVXLCDM]+|[A-Za-z]\d*|\d+)$", re.IGNORECASE)
    # Greek letter normalization: α -> a, β -> b, γ -> g
    _GREEK_MAP = {
        b"\xce\xb1": b"a",
        b"\xce\xb2": b"b",
        b"\xce\xb3": b"g",
    }

    def __init__(self, db_keywords: Iterable[str] | None = None) -> None:
        r"""Initializes the matcher with installed Kaptive reference databases.

        Args:
            db_keywords: Optional iterable of db keywords to load (e.g. `['kpsc_k', 'kpsc_o']`).
                If `None`, automatically queries `DatabaseManager.installed()`.
        """
        self.databases: dict[bytes, dict[str, Any]] = {}
        self._load_databases(db_keywords)

    def _load_databases(self, db_keywords: Iterable[str] | None) -> None:
        try:
            from kaptive.db.manager import DatabaseManager
        except ImportError:
            return

        kwds = list(db_keywords) if db_keywords is not None else DatabaseManager.installed()

        for kwd in kwds:
            try:
                db: Database = DatabaseManager.load(kwd)
            except Exception:
                continue

            meta = db.metadata
            kwd_bytes = meta.keyword
            org_bytes = meta.organism
            antigen_bytes = meta.antigen

            # Extract prefix property from Database metadata or infer from locus IDs
            prefix = getattr(meta, "prefix", b"") or b""
            if not prefix and len(db.loci.ids) > 0:
                m = re.match(rb"^[A-Za-z]+", db.loci.ids[0].strip())
                if m:
                    prefix = m.group(0)

            # Index loci by their first run of numeric characters
            num_to_locus: dict[bytes, bytes] = {}
            for loc in db.loci.ids:
                loc_clean = loc.strip()
                m = re.search(rb"\d+", loc_clean)
                if m:
                    digits = m.group(0)
                    if digits not in num_to_locus:
                        num_to_locus[digits] = loc_clean

            # Build serotype/locus lookup dictionaries
            serotype_to_locus: dict[bytes, bytes] = {}
            loci_set = set(db.loci.ids)

            for loc, st in zip(db.loci.ids, db.serotypes, strict=False):
                loc_clean = loc.strip()
                serotype_to_locus[loc_clean] = loc_clean
                serotype_to_locus[loc_clean.upper()] = loc_clean
                serotype_to_locus[loc_clean.lower()] = loc_clean

                st_clean = st.strip()
                if st_clean and st_clean != b"unknown":
                    serotype_to_locus[st_clean] = loc_clean
                    serotype_to_locus[st_clean.upper()] = loc_clean
                    serotype_to_locus[st_clean.lower()] = loc_clean

            # Map db prefix + numeric run to locus
            if prefix:
                for digits, loc_clean in num_to_locus.items():
                    serotype_to_locus.setdefault(prefix + digits, loc_clean)
                    serotype_to_locus.setdefault((prefix + digits).upper(), loc_clean)
                    serotype_to_locus.setdefault((prefix + digits).lower(), loc_clean)

            self.databases[kwd_bytes] = {
                "keyword": kwd_bytes,
                "prefix": prefix,
                "organism": org_bytes,
                "antigen": antigen_bytes,
                "loci_set": loci_set,
                "serotype_to_locus": serotype_to_locus,
                "num_to_locus": num_to_locus,
            }

    def _normalize_serotype(self, st: bytes) -> bytes:
        norm = st.strip()
        for g_from, g_to in self._GREEK_MAP.items():
            norm = norm.replace(g_from, g_to)
        return norm

    def match(
        self,
        organism: bytes,
        serotype: bytes = b"",
        locus_hint: bytes = b"",
    ) -> tuple[bytes, bytes]:
        r"""Matches organism and serotype/locus queries to an installed Kaptive db and locus.

        Args:
            organism: Canonical or colloquial organism name byte string (e.g. `b"Escherichia coli"`).
            serotype: Serotype or antigen designation byte string (e.g. `b"K1"`, `b"O1 A"`, `b"KL242"`).
            locus_hint: Optional direct locus hint byte string (e.g. `b"KL10"`).

        Returns:
            tuple[bytes, bytes]: `(kaptive_db_keyword, kaptive_locus)` if a match is resolved,
                or `(b"", b"")` if no match is found or the db is uninstalled.
        """
        org_bytes = organism.strip().lower()
        st_bytes = serotype.strip()
        loc_bytes = locus_hint.strip()

        if not org_bytes:
            return b"", b""

        # 1. Candidate db routing by organism
        candidates: list[bytes] = []
        for kwd, db_info in self.databases.items():
            db_org = db_info["organism"].lower()
            if self._organism_matches(org_bytes, db_org):
                candidates.append(kwd)

        if not candidates:
            return b"", b""

        # 2. Prioritize candidate databases by metadata prefix and antigen cues
        query_text = (st_bytes + b" " + loc_bytes).upper()

        def priority_key(kwd: bytes) -> int:
            db_prefix = self.databases[kwd]["prefix"]
            if db_prefix and query_text.startswith(db_prefix.upper()):
                return 0
            antigen = self.databases[kwd]["antigen"].lower()
            if b"capsul" in antigen and any(c in query_text for c in (b"K", b"CPS")):
                return 0
            if (b"outer" in antigen or b"lps" in antigen) and any(c in query_text for c in (b"O", b"LPS")):
                return 0
            return 1

        candidates.sort(key=priority_key)

        # 3. Two-tier locus resolution with numeric run fallback
        queries_to_try: list[bytes] = []
        for q in (loc_bytes, st_bytes):
            if q:
                queries_to_try.append(q)
                norm_q = self._normalize_serotype(q)
                if norm_q != q:
                    queries_to_try.append(norm_q)

        for kwd in candidates:
            db_info = self.databases[kwd]
            s2l = db_info["serotype_to_locus"]
            loci_set = db_info["loci_set"]
            num_to_locus = db_info["num_to_locus"]
            db_prefix = db_info["prefix"]

            for q in queries_to_try:
                # Tier 1: Exact match against locus set or serotype map
                res = self._match_exact(q, loci_set, s2l)
                if res:
                    return kwd, res

                # Tier 2: Subvariant suffix stripping
                res = self._match_subvariant(q, loci_set, s2l)
                if res:
                    return kwd, res

                # Tier 3: Numeric run comparison fallback
                qm = re.match(rb"^([A-Za-z]*)(\d+)", q)
                if qm:
                    q_prefix = qm.group(1).upper()
                    q_num = qm.group(2)
                    # Check prefix compatibility if both specify an antigen prefix
                    if q_prefix and db_prefix:
                        if not (q_prefix.startswith(db_prefix) or db_prefix.startswith(q_prefix)):
                            continue
                    if q_num in num_to_locus:
                        return kwd, num_to_locus[q_num]

        return b"", b""

    @staticmethod
    def _organism_matches(query_org: bytes, db_org: bytes) -> bool:
        """Determines if query organism name aligns with db organism."""
        if query_org == db_org or query_org in db_org or db_org in query_org:
            return True
        # Common taxonomic synonyms and abbreviations
        if b"klebsiella" in query_org and b"klebsiella" in db_org:
            return True
        if b"acinetobacter" in query_org and b"acinetobacter" in db_org:
            return True
        if (b"escherichia" in query_org or query_org.startswith(b"e. coli")) and b"escherichia" in db_org:
            return True
        return False

    @staticmethod
    def _match_exact(
        query: bytes,
        loci_set: set[bytes],
        s2l: dict[bytes, bytes],
    ) -> bytes | None:
        """Tier 1: Exact match against locus set or serotype map."""
        if query in loci_set:
            return query
        res = s2l.get(query)
        if res is not None:
            return res
        qu = query.upper()
        res = s2l.get(qu)
        if res is not None:
            return res
        ql = query.lower()
        if ql != qu:
            return s2l.get(ql)
        return None

    @classmethod
    def _match_subvariant(
        cls,
        query: bytes,
        loci_set: set[bytes],
        s2l: dict[bytes, bytes],
    ) -> bytes | None:
        """Tier 2: Strips subvariant suffix (e.g. '-I', '-A1', '.1') and attempts match."""
        stripped = cls._SUBVARIANT_RE.sub(b"", query).strip()
        if stripped and stripped != query:
            res = cls._match_exact(stripped, loci_set, s2l)
            if res:
                return res

        # Also strip trailing single letters (e.g. O1a -> O1, K2A -> K2)
        m = re.match(rb"^([A-Za-z]+\d+)[A-Za-z]$", query)
        if m:
            base = m.group(1)
            res = cls._match_exact(base, loci_set, s2l)
            if res:
                return res

        return None
