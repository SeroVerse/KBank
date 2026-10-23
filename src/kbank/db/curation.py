"""Taxonomy normalization, topological glycan consolidation, and 4-tier SeroGen locus pairing."""

from __future__ import annotations

import asyncio
import gzip
import os
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from numba import njit, prange
from serogen.core.collections import RaggedArrayContainer, ScalarView, lengths_to_offsets
from serogen.core.genome import GenomeAssembly
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import ArchetypeCategory, LocusArchetype, roles_to_mask
from serogen.loci.models import Loci, Locus
from serogen.loci.predictor import LocusPredictor

from kbank.biosynthesis.align import BiosyntheticAssigner, _cached_locus_phage_probs
from kbank.db.models import EntryRelation, GlycanEntry, GlycanRecord
from kbank.db.records import DEFAULT_DB_PATH, GlycanRecords, _loci_from_npz_dict, _loci_to_npz_dict
from kbank.glycan import Glycan, Glycans
from kbank.io.sources import DEFAULT_CACHE_DIR, SALMONELLA_SEROVAR_TO_OGROUP
from kbank.sugars import Sugars

NCBI_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_LOCI_CACHE_PATH = DEFAULT_CACHE_DIR / "predicted_loci.npz"

OGROUP_TO_SALMONELLA_SEROVARS: dict[str, tuple[str, ...]] = {
    "O2": ("Paratyphi A",),
    "O4": ("Typhimurium", "Paratyphi B", "Heidelberg", "Agona", "Derby", "Saintpaul"),
    "O7": ("Choleraesuis", "Montevideo", "Infantis", "Thompson", "Paratyphi C"),
    "O8": ("Newport", "Muenchen", "Kentucky", "Hadar"),
    "O9": ("Typhi", "Enteritidis", "Dublin", "Gallinarum", "Pullorum"),
    "O3,10": ("Anatum", "Give", "London"),
    "O1,3,19": ("Senftenberg",),
    "O11": ("Aberdeen",),
    "O13": ("Poona",),
    "O18": ("Cerro",),
    "O21": ("Minnesota",),
    "O30": ("Urbana",),
    "O35": ("Adelaide",),
}

_CSDB_XREF_RE = re.compile(rb"\bCSDB:(\d+)\b")
_WORD_SPLIT_RE = re.compile(r"[A-Za-z0-9]+(?::[A-Za-z0-9]+)?")
_NUM_SERO_RE = re.compile(r"(?:serotype|serogroup|type|capsule|b-band|osa|iats)\s+O?([0-9]{1,3}[A-Za-z]?)\b", re.I)
_COLLECTION_RE = re.compile(r"\b(PCM|CCM|IMET|IAM|CIP|JCM|DSM|LMG|CECT|NCIMB|VKM)\b", re.I)
_DB_PRIORITY: dict[bytes, int] = {b"ECODAB": 5, b"EK3D": 5, b"EPSDB": 4, b"CSDB": 3, b"KEGG": 1}
_GENUS_CANONICAL_SPECIES: dict[bytes, bytes] = {
    sp.split(b" ", 1)[0]: sp
    for sp in (
        b"Klebsiella pneumoniae|Salmonella enterica|Escherichia coli|Shigella flexneri|Streptococcus pneumoniae|"
        b"Acinetobacter baumannii|Pseudomonas aeruginosa|Proteus mirabilis|Yersinia pseudotuberculosis|"
        b"Vibrio cholerae|Campylobacter jejuni|Helicobacter pylori|Haemophilus influenzae|Neisseria meningitidis|"
        b"Staphylococcus aureus|Listeria monocytogenes|Enterococcus faecalis|Cronobacter sakazakii|"
        b"Serratia marcescens|Citrobacter freundii|Aeromonas hydrophila|Providencia alcalifaciens"
    ).split(b"|")
}
_GENUS_LEVEL_TAXA = frozenset(
    "Klebsiella Acinetobacter Salmonella Shigella Proteus Yersinia Campylobacter Cronobacter Burkholderia "
    "Providencia Raoultella Serratia Citrobacter Enterobacter Morganella Hafnia Edwardsiella Vibrio Aeromonas "
    "Shewanella Pseudoalteromonas Rhodococcus Rhizobium Sinorhizobium Bradyrhizobium Agrobacterium Azospirillum "
    "Ochrobactrum Xanthomonas Pantoea Erwinia Pectobacterium Dickeya Bacteroides Clostridium Bacillus "
    "Lactobacillus Bifidobacterium".split()
)

_IS_MODIFIER_LUT: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.zeros(256, dtype=np.bool_)
_IS_MODIFIER_LUT[: len(Sugars.classes)] = Sugars.classes == b"Mod"
_OL_SUGAR_IDX: int = int(Sugars.name_to_index.get(b"ol", -1))


def _cassette_ctx(recorder: Any | None, cassette_name: str) -> Any:
    """Return `recorder.use_cassette(cassette_name)` if `recorder` is provided, else `nullcontext()`."""
    has_cassette = recorder is not None and hasattr(recorder, "use_cassette")
    return recorder.use_cassette(cassette_name) if has_cassette else nullcontext()


def antigen_family(antigen_type: bytes) -> bytes:
    """Normalize `antigen_type` labels into canonical family buckets (`CPS`, `O-antigen`, `WTA`, `EPS`)."""
    if antigen_type.startswith(b"CPS"):
        return b"CPS"
    return b"O-antigen" if antigen_type in (b"O-antigen", b"LPS", b"LOS") else antigen_type


def normalize_raw_record_taxonomy(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unify `<Genus> sp.` organism labels with specific species when serotype or strain matches."""
    genus_sero_to_species: dict[tuple[bytes, bytes], bytes] = {}
    genus_strain_to_species: dict[tuple[bytes, bytes], bytes] = {}

    for r in raw_records:
        org: bytes = r["organism"]
        if not org.endswith(b" sp.") and b" " in org:
            genus = org.split(b" ", 1)[0]
            if sero := r.get("serotype", b""):
                genus_sero_to_species.setdefault((genus, sero), org)
            if strain := r.get("strain", b""):
                genus_strain_to_species.setdefault((genus, strain), org)

    for r in raw_records:
        org = r["organism"]
        if org.endswith(b" sp."):
            genus = org.split(b" ", 1)[0]
            sero, strain = r.get("serotype", b""), r.get("strain", b"")
            if sero and (genus, sero) in genus_sero_to_species:
                r["organism"] = genus_sero_to_species[(genus, sero)]
            elif strain and (genus, strain) in genus_strain_to_species:
                r["organism"] = genus_strain_to_species[(genus, strain)]
            elif sero and genus in _GENUS_CANONICAL_SPECIES:
                r["organism"] = _GENUS_CANONICAL_SPECIES[genus]

        if r["organism"] == b"Salmonella enterica" and not r.get("serotype") and r.get("strain"):
            st_low = r["strain"].decode("utf-8", errors="replace").strip().lower()
            if st_low in SALMONELLA_SEROVAR_TO_OGROUP:
                r["serotype"] = SALMONELLA_SEROVAR_TO_OGROUP[st_low].encode("utf-8")

    return raw_records


# ======================================================================================================================
# 1. Vectorized Numba Fingerprinting (`GlycanFingerprint` & `GlycanFingerprints`) & Consolidation
# ======================================================================================================================


@njit(cache=True, nogil=True)
def _sort_unique_u16(buf: np.ndarray[Any, Any], n: int) -> int:
    if n <= 1:
        return n
    for i in range(1, n):
        key = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > key:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = key
    w = 1
    for i in range(1, n):
        if buf[i] != buf[w - 1]:
            buf[w] = buf[i]
            w += 1
    return w


@njit(cache=True, nogil=True)
def _sort_unique_u32(buf: np.ndarray[Any, Any], n: int) -> int:
    if n <= 1:
        return n
    for i in range(1, n):
        key = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > key:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = key
    w = 1
    for i in range(1, n):
        if buf[i] != buf[w - 1]:
            buf[w] = buf[i]
            w += 1
    return w


@njit(cache=True, nogil=True)
def _sorted_subset_u16(sub: np.ndarray[Any, Any], sup: np.ndarray[Any, Any]) -> bool:
    n_sub, n_sup = len(sub), len(sup)
    if n_sub == 0:
        return True
    if n_sub > n_sup:
        return False
    i, j = 0, 0
    while i < n_sub and j < n_sup:
        if sub[i] == sup[j]:
            i += 1
            j += 1
        elif sub[i] > sup[j]:
            j += 1
        else:
            return False
    return i == n_sub


@njit(cache=True, nogil=True)
def _sorted_subset_u32(sub: np.ndarray[Any, Any], sup: np.ndarray[Any, Any]) -> bool:
    n_sub, n_sup = len(sub), len(sup)
    if n_sub == 0:
        return True
    if n_sub > n_sup:
        return False
    i, j = 0, 0
    while i < n_sub and j < n_sup:
        if sub[i] == sup[j]:
            i += 1
            j += 1
        elif sub[i] > sup[j]:
            j += 1
        else:
            return False
    return i == n_sub


@njit(cache=True, nogil=True, parallel=True)
def _compute_fingerprints_pass1(
    node_types: np.ndarray[Any, Any],
    node_offsets: np.ndarray[Any, Any],
    edge_sources: np.ndarray[Any, Any],
    edge_targets: np.ndarray[Any, Any],
    edge_types: np.ndarray[Any, Any],
    edge_offsets: np.ndarray[Any, Any],
    is_mod_lut: np.ndarray[Any, Any],
    ol_idx: int,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int16]],
    np.ndarray[tuple[int], np.dtype[np.int16]],
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    n = len(node_offsets) - 1
    num_core = np.zeros(n, dtype=np.int16)
    num_mod = np.zeros(n, dtype=np.int16)
    c_lo = np.zeros(n, dtype=np.uint64)
    c_hi = np.zeros(n, dtype=np.uint64)
    s_lo = np.zeros(n, dtype=np.uint64)
    s_hi = np.zeros(n, dtype=np.uint64)
    is_ald = np.zeros(n, dtype=np.bool_)
    has_wc = np.zeros(n, dtype=np.bool_)
    pair_lens = np.zeros(n, dtype=np.int32)
    triple_lens = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        n_start, n_end = int(node_offsets[i]), int(node_offsets[i + 1])
        e_start, e_end = int(edge_offsets[i]), int(edge_offsets[i + 1])
        nc, nm = 0, 0
        clo, chi = np.uint64(0), np.uint64(0)
        slo, shi = np.uint64(0), np.uint64(0)

        for u in range(n_start, n_end):
            s = int(node_types[u] & 0xFF)
            if s < 64:
                bit = np.uint64(1) << np.uint64(s)
                slo |= bit
                if not is_mod_lut[s]:
                    clo |= bit
                    nc += 1
                else:
                    nm += 1
            else:
                bit = np.uint64(1) << np.uint64(s - 64)
                shi |= bit
                if not is_mod_lut[s]:
                    chi |= bit
                    nc += 1
                else:
                    nm += 1

        m_edges = e_end - e_start
        p_buf = np.empty(m_edges, dtype=np.uint16)
        t_buf = np.empty(m_edges, dtype=np.uint32)
        c_cnt = 0
        ald = False
        wc = False

        for e in range(e_start, e_end):
            src, tgt = int(edge_sources[e]), int(edge_targets[e])
            s_src, s_tgt = int(node_types[src] & 0xFF), int(node_types[tgt] & 0xFF)
            if tgt == n_start and s_src == ol_idx:
                ald = True
            if not is_mod_lut[s_src] and not is_mod_lut[s_tgt]:
                et = int(edge_types[e])
                anom = et & 0x3
                don = (et >> 2) & 0xF
                acc = (et >> 6) & 0xF
                if anom == 0 or don == 0 or acc == 0:
                    wc = True
                chem = et & 0x3FF
                p_buf[c_cnt] = np.uint16((s_src << 8) | s_tgt)
                t_buf[c_cnt] = np.uint32((s_src << 18) | (chem << 8) | s_tgt)
                c_cnt += 1

        num_core[i] = nc
        num_mod[i] = nm
        c_lo[i], c_hi[i] = clo, chi
        s_lo[i], s_hi[i] = slo, shi
        is_ald[i] = ald
        has_wc[i] = wc
        pair_lens[i] = _sort_unique_u16(p_buf, c_cnt)
        triple_lens[i] = _sort_unique_u32(t_buf, c_cnt)

    return num_core, num_mod, c_lo, c_hi, s_lo, s_hi, is_ald, has_wc, pair_lens, triple_lens


@njit(cache=True, nogil=True, parallel=True)
def _compute_fingerprints_pass2(
    node_types: np.ndarray[Any, Any],
    edge_sources: np.ndarray[Any, Any],
    edge_targets: np.ndarray[Any, Any],
    edge_types: np.ndarray[Any, Any],
    edge_offsets: np.ndarray[Any, Any],
    is_mod_lut: np.ndarray[Any, Any],
    pair_offsets: np.ndarray[Any, Any],
    pair_codes: np.ndarray[Any, Any],
    offsets: np.ndarray[Any, Any],
    triple_codes: np.ndarray[Any, Any],
) -> None:
    n = len(edge_offsets) - 1
    for i in prange(n):
        e_start, e_end = int(edge_offsets[i]), int(edge_offsets[i + 1])
        m_edges = e_end - e_start
        if m_edges == 0:
            continue
        p_buf = np.empty(m_edges, dtype=np.uint16)
        t_buf = np.empty(m_edges, dtype=np.uint32)
        c_cnt = 0
        for e in range(e_start, e_end):
            src, tgt = int(edge_sources[e]), int(edge_targets[e])
            s_src, s_tgt = int(node_types[src] & 0xFF), int(node_types[tgt] & 0xFF)
            if not is_mod_lut[s_src] and not is_mod_lut[s_tgt]:
                chem = int(edge_types[e]) & 0x3FF
                p_buf[c_cnt] = np.uint16((s_src << 8) | s_tgt)
                t_buf[c_cnt] = np.uint32((s_src << 18) | (chem << 8) | s_tgt)
                c_cnt += 1
        np_u = _sort_unique_u16(p_buf, c_cnt)
        nt_u = _sort_unique_u32(t_buf, c_cnt)
        p_off = int(pair_offsets[i])
        for k in range(np_u):
            pair_codes[p_off + k] = p_buf[k]
        t_off = int(offsets[i])
        for k in range(nt_u):
            triple_codes[t_off + k] = t_buf[k]


@njit(cache=True, nogil=True)
def _can_be_iso_kernel(
    n_a: int,
    clo_a: int | np.uint64,
    chi_a: int | np.uint64,
    wc_a: bool,
    pairs_a: np.ndarray[Any, Any],
    triples_a: np.ndarray[Any, Any],
    n_b: int,
    clo_b: int | np.uint64,
    chi_b: int | np.uint64,
    wc_b: bool,
    pairs_b: np.ndarray[Any, Any],
    triples_b: np.ndarray[Any, Any],
) -> bool:
    if n_a != n_b or clo_a != clo_b or chi_a != chi_b:
        return False
    if not (_sorted_subset_u16(pairs_a, pairs_b) or _sorted_subset_u16(pairs_b, pairs_a)):
        return False
    if not (wc_a or wc_b):
        return _sorted_subset_u32(triples_a, triples_b) or _sorted_subset_u32(triples_b, triples_a)
    return True


@njit(cache=True, nogil=True)
def _can_be_sub_kernel(
    n_a: int,
    clo_a: int | np.uint64,
    chi_a: int | np.uint64,
    wc_a: bool,
    pairs_a: np.ndarray[Any, Any],
    triples_a: np.ndarray[Any, Any],
    n_b: int,
    clo_b: int | np.uint64,
    chi_b: int | np.uint64,
    wc_b: bool,
    pairs_b: np.ndarray[Any, Any],
    triples_b: np.ndarray[Any, Any],
) -> bool:
    if n_a == 0 or n_a > n_b:
        return False
    if (clo_a & ~clo_b) != 0 or (chi_a & ~chi_b) != 0:
        return False
    if not _sorted_subset_u16(pairs_a, pairs_b):
        return False
    return _sorted_subset_u32(triples_a, triples_b) if not (wc_a or wc_b) else True


@njit(cache=True, nogil=True)
def _filter_iso_batch_kernel(
    q_idx: int,
    cand_indices: np.ndarray[Any, Any],
    num_core_nodes: np.ndarray[Any, Any],
    core_mask_lo: np.ndarray[Any, Any],
    core_mask_hi: np.ndarray[Any, Any],
    has_wildcard_edges: np.ndarray[Any, Any],
    pair_codes: np.ndarray[Any, Any],
    pair_offsets: np.ndarray[Any, Any],
    pair_lengths: np.ndarray[Any, Any],
    triple_codes: np.ndarray[Any, Any],
    offsets: np.ndarray[Any, Any],
    lengths: np.ndarray[Any, Any],
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    n_cands = len(cand_indices)
    out = np.empty(n_cands, dtype=np.int32)
    cnt = 0
    n_q = int(num_core_nodes[q_idx])
    clo_q, chi_q = core_mask_lo[q_idx], core_mask_hi[q_idx]
    wc_q = bool(has_wildcard_edges[q_idx])
    po_q, pl_q = int(pair_offsets[q_idx]), int(pair_lengths[q_idx])
    to_q, tl_q = int(offsets[q_idx]), int(lengths[q_idx])
    pairs_q = pair_codes[po_q : po_q + pl_q]
    triples_q = triple_codes[to_q : to_q + tl_q]

    for k in range(n_cands):
        c = int(cand_indices[k])
        po_c, pl_c = int(pair_offsets[c]), int(pair_lengths[c])
        to_c, tl_c = int(offsets[c]), int(lengths[c])
        if _can_be_iso_kernel(
            n_q,
            clo_q,
            chi_q,
            wc_q,
            pairs_q,
            triples_q,
            int(num_core_nodes[c]),
            core_mask_lo[c],
            core_mask_hi[c],
            bool(has_wildcard_edges[c]),
            pair_codes[po_c : po_c + pl_c],
            triple_codes[to_c : to_c + tl_c],
        ):
            out[cnt] = k
            cnt += 1
    return out[:cnt]


@njit(cache=True, nogil=True)
def _filter_sub_or_oligo_batch_kernel(
    q_idx: int,
    q_is_rep: bool,
    cand_indices: np.ndarray[Any, Any],
    cand_is_rep: np.ndarray[Any, Any],
    num_core_nodes: np.ndarray[Any, Any],
    core_mask_lo: np.ndarray[Any, Any],
    core_mask_hi: np.ndarray[Any, Any],
    has_wildcard_edges: np.ndarray[Any, Any],
    pair_codes: np.ndarray[Any, Any],
    pair_offsets: np.ndarray[Any, Any],
    pair_lengths: np.ndarray[Any, Any],
    triple_codes: np.ndarray[Any, Any],
    offsets: np.ndarray[Any, Any],
    lengths: np.ndarray[Any, Any],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.int32]], np.ndarray[tuple[int], np.dtype[np.int8]]]:
    n_cands = len(cand_indices)
    out_pos = np.empty(n_cands, dtype=np.int32)
    out_mode = np.empty(n_cands, dtype=np.int8)
    cnt = 0

    n_q = int(num_core_nodes[q_idx])
    clo_q, chi_q = core_mask_lo[q_idx], core_mask_hi[q_idx]
    wc_q = bool(has_wildcard_edges[q_idx])
    po_q, pl_q = int(pair_offsets[q_idx]), int(pair_lengths[q_idx])
    to_q, tl_q = int(offsets[q_idx]), int(lengths[q_idx])
    pairs_q = pair_codes[po_q : po_q + pl_q]
    triples_q = triple_codes[to_q : to_q + tl_q]

    for k in range(n_cands):
        c = int(cand_indices[k])
        n_c = int(num_core_nodes[c])
        clo_c, chi_c = core_mask_lo[c], core_mask_hi[c]
        po_c, pl_c = int(pair_offsets[c]), int(pair_lengths[c])
        to_c, tl_c = int(offsets[c]), int(lengths[c])
        mode = np.int8(0)

        if _can_be_sub_kernel(
            n_q,
            clo_q,
            chi_q,
            wc_q,
            pairs_q,
            triples_q,
            n_c,
            clo_c,
            chi_c,
            bool(has_wildcard_edges[c]),
            pair_codes[po_c : po_c + pl_c],
            triple_codes[to_c : to_c + tl_c],
        ):
            mode |= np.int8(1)
        if not q_is_rep and 0 < n_c < n_q and (clo_q & ~clo_c) == 0 and (chi_q & ~chi_c) == 0:
            mode |= np.int8(2)
        if cand_is_rep[k] and not q_is_rep and 0 < n_q < n_c and (clo_c & ~clo_q) == 0 and (chi_c & ~chi_q) == 0:
            mode |= np.int8(4)

        if mode != 0:
            out_pos[cnt] = k
            out_mode[cnt] = mode
            cnt += 1
    return out_pos[:cnt], out_mode[:cnt]


@dataclass(frozen=True, slots=True)
class GlycanFingerprint(ScalarView):
    """Scalar view for a single glycan's O(1) structural hash and sorted transition codes in `GlycanFingerprints`."""

    num_core_nodes: int
    num_modifiers: int
    core_mask_lo: int
    core_mask_hi: int
    sugar_mask_lo: int
    sugar_mask_hi: int
    is_alditol: bool
    has_wildcard_edges: bool
    pair_codes: np.ndarray[tuple[int], np.dtype[np.uint16]]
    triple_codes: np.ndarray[tuple[int], np.dtype[np.uint32]]
    index: int = -1

    @property
    def core_sugar_mask(self) -> int:
        return int(self.core_mask_lo) | (int(self.core_mask_hi) << 64)

    @property
    def sugar_mask(self) -> int:
        return int(self.sugar_mask_lo) | (int(self.sugar_mask_hi) << 64)

    @property
    def iso_bucket_key(self) -> tuple[int, int, int]:
        return (self.num_core_nodes, self.core_mask_lo, self.core_mask_hi)

    @classmethod
    def from_glycan(cls, gly: Glycan) -> GlycanFingerprint:
        """Compute a scalar `GlycanFingerprint` via `GlycanFingerprints` Numba kernel."""
        return GlycanFingerprints.from_glycans(Glycans.from_list([gly]))[0]

    def can_be_isomorphic(self, other: GlycanFingerprint) -> bool:
        """Return True if `self` and `other` pass O(1) isomorphism/open-repeat pre-filters."""
        return bool(
            _can_be_iso_kernel(
                self.num_core_nodes,
                np.uint64(self.core_mask_lo),
                np.uint64(self.core_mask_hi),
                self.has_wildcard_edges,
                self.pair_codes,
                self.triple_codes,
                other.num_core_nodes,
                np.uint64(other.core_mask_lo),
                np.uint64(other.core_mask_hi),
                other.has_wildcard_edges,
                other.pair_codes,
                other.triple_codes,
            )
        )

    def can_be_subgraph_of(self, other: GlycanFingerprint) -> bool:
        """Return True if `self` can be a topological subgraph fragment of `other` in O(1)."""
        return bool(
            _can_be_sub_kernel(
                self.num_core_nodes,
                np.uint64(self.core_mask_lo),
                np.uint64(self.core_mask_hi),
                self.has_wildcard_edges,
                self.pair_codes,
                self.triple_codes,
                other.num_core_nodes,
                np.uint64(other.core_mask_lo),
                np.uint64(other.core_mask_hi),
                other.has_wildcard_edges,
                other.pair_codes,
                other.triple_codes,
            )
        )

    def can_be_oligomer_of(self, other: GlycanFingerprint) -> bool:
        """Return True if `self` can be a multi-repeat oligomer of `other` in O(1)."""
        return (
            0 < other.num_core_nodes < self.num_core_nodes
            and (self.core_mask_lo & ~other.core_mask_lo) == 0
            and (self.core_mask_hi & ~other.core_mask_hi) == 0
        )


@dataclass(frozen=True, slots=True)
class GlycanFingerprints(RaggedArrayContainer[GlycanFingerprint, "GlycanFingerprints"]):
    """Ragged Structure-of-Arrays (SoA) batch of `GlycanFingerprint` records computed via parallel Numba kernels."""

    num_core_nodes: np.ndarray[tuple[int], np.dtype[np.int16]]
    num_modifiers: np.ndarray[tuple[int], np.dtype[np.int16]]
    core_mask_lo: np.ndarray[tuple[int], np.dtype[np.uint64]]
    core_mask_hi: np.ndarray[tuple[int], np.dtype[np.uint64]]
    sugar_mask_lo: np.ndarray[tuple[int], np.dtype[np.uint64]]
    sugar_mask_hi: np.ndarray[tuple[int], np.dtype[np.uint64]]
    is_alditol: np.ndarray[tuple[int], np.dtype[np.bool_]]
    has_wildcard_edges: np.ndarray[tuple[int], np.dtype[np.bool_]]
    pair_codes: np.ndarray[tuple[int], np.dtype[np.uint16]]
    pair_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    pair_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    triple_codes: np.ndarray[tuple[int], np.dtype[np.uint32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        return len(self.offsets)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> GlycanFingerprint: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> GlycanFingerprints: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> GlycanFingerprint | GlycanFingerprints: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> GlycanFingerprint | GlycanFingerprints:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("GlycanFingerprints index out of range")
            p_off, p_ln = int(self.pair_offsets[idx]), int(self.pair_lengths[idx])
            t_off, t_ln = int(self.offsets[idx]), int(self.lengths[idx])
            return GlycanFingerprint(
                num_core_nodes=int(self.num_core_nodes[idx]),
                num_modifiers=int(self.num_modifiers[idx]),
                core_mask_lo=int(self.core_mask_lo[idx]),
                core_mask_hi=int(self.core_mask_hi[idx]),
                sugar_mask_lo=int(self.sugar_mask_lo[idx]),
                sugar_mask_hi=int(self.sugar_mask_hi[idx]),
                is_alditol=bool(self.is_alditol[idx]),
                has_wildcard_edges=bool(self.has_wildcard_edges[idx]),
                pair_codes=self.pair_codes[p_off : p_off + p_ln],
                triple_codes=self.triple_codes[t_off : t_off + t_ln],
                index=idx,
            )

        indices, new_offsets, new_lengths, (new_triples,) = self._slice_ragged(item, self.triple_codes)
        if len(indices) == 0:
            return self.empty(0)
        new_p_lens = self.pair_lengths[indices].astype(np.int32, copy=False)
        new_p_offs = lengths_to_offsets(new_p_lens)
        tot_p = int(new_p_offs[-1] + new_p_lens[-1])
        if tot_p == 0:
            new_pairs = np.empty(0, dtype=np.uint16)
        else:
            from serogen.core.collections import _gather_ragged_kernel

            new_pairs = _gather_ragged_kernel(
                self.pair_codes, self.pair_offsets, new_p_offs, new_p_lens, indices, tot_p
            )
        return GlycanFingerprints(
            num_core_nodes=self.num_core_nodes[indices],
            num_modifiers=self.num_modifiers[indices],
            core_mask_lo=self.core_mask_lo[indices],
            core_mask_hi=self.core_mask_hi[indices],
            sugar_mask_lo=self.sugar_mask_lo[indices],
            sugar_mask_hi=self.sugar_mask_hi[indices],
            is_alditol=self.is_alditol[indices],
            has_wildcard_edges=self.has_wildcard_edges[indices],
            pair_codes=new_pairs,
            pair_offsets=new_p_offs,
            pair_lengths=new_p_lens,
            triple_codes=new_triples,
            offsets=new_offsets,
            lengths=new_lengths,
        )

    @classmethod
    def empty(cls, length: int = 0) -> Self:
        z_i32 = np.zeros(length, dtype=np.int32) if length > 0 else np.empty(0, dtype=np.int32)
        z_i16 = np.zeros(length, dtype=np.int16) if length > 0 else np.empty(0, dtype=np.int16)
        z_u64 = np.zeros(length, dtype=np.uint64) if length > 0 else np.empty(0, dtype=np.uint64)
        z_b = np.zeros(length, dtype=np.bool_) if length > 0 else np.empty(0, dtype=np.bool_)
        return cls(
            num_core_nodes=z_i16,
            num_modifiers=z_i16.copy(),
            core_mask_lo=z_u64,
            core_mask_hi=z_u64.copy(),
            sugar_mask_lo=z_u64.copy(),
            sugar_mask_hi=z_u64.copy(),
            is_alditol=z_b,
            has_wildcard_edges=z_b.copy(),
            pair_codes=np.empty(0, dtype=np.uint16),
            pair_offsets=z_i32,
            pair_lengths=z_i32.copy(),
            triple_codes=np.empty(0, dtype=np.uint32),
            offsets=z_i32.copy(),
            lengths=z_i32.copy(),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        batch_list = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty(0)
        if len(batch_list) == 1:
            return batch_list[0]
        p_lens = np.concatenate([b.pair_lengths for b in batch_list])
        t_lens = np.concatenate([b.lengths for b in batch_list])
        p_non = [b.pair_codes for b in batch_list if len(b.pair_codes) > 0]
        t_non = [b.triple_codes for b in batch_list if len(b.triple_codes) > 0]
        return cls(
            num_core_nodes=np.concatenate([b.num_core_nodes for b in batch_list]),
            num_modifiers=np.concatenate([b.num_modifiers for b in batch_list]),
            core_mask_lo=np.concatenate([b.core_mask_lo for b in batch_list]),
            core_mask_hi=np.concatenate([b.core_mask_hi for b in batch_list]),
            sugar_mask_lo=np.concatenate([b.sugar_mask_lo for b in batch_list]),
            sugar_mask_hi=np.concatenate([b.sugar_mask_hi for b in batch_list]),
            is_alditol=np.concatenate([b.is_alditol for b in batch_list]),
            has_wildcard_edges=np.concatenate([b.has_wildcard_edges for b in batch_list]),
            pair_codes=np.concatenate(p_non) if p_non else np.empty(0, dtype=np.uint16),
            pair_offsets=lengths_to_offsets(p_lens),
            pair_lengths=p_lens,
            triple_codes=np.concatenate(t_non) if t_non else np.empty(0, dtype=np.uint32),
            offsets=lengths_to_offsets(t_lens),
            lengths=t_lens,
        )

    @classmethod
    def from_glycans(cls, glycans: Glycans) -> Self:
        """Compute `GlycanFingerprints` for an entire `Glycans` SoA batch via parallel Numba kernels."""
        if len(glycans) == 0:
            return cls.empty(0)
        nc, nm, clo, chi, slo, shi, ald, wc, p_lens, t_lens = _compute_fingerprints_pass1(
            glycans.node_types,
            glycans.node_offsets,
            glycans.edge_sources,
            glycans.edge_targets,
            glycans.edge_types,
            glycans.edge_offsets,
            _IS_MODIFIER_LUT,
            _OL_SUGAR_IDX,
        )
        p_offs = lengths_to_offsets(p_lens)
        t_offs = lengths_to_offsets(t_lens)
        tot_p = int(p_offs[-1] + p_lens[-1])
        tot_t = int(t_offs[-1] + t_lens[-1])
        p_codes = np.empty(tot_p, dtype=np.uint16)
        t_codes = np.empty(tot_t, dtype=np.uint32)
        if tot_p > 0 or tot_t > 0:
            _compute_fingerprints_pass2(
                glycans.node_types,
                glycans.edge_sources,
                glycans.edge_targets,
                glycans.edge_types,
                glycans.edge_offsets,
                _IS_MODIFIER_LUT,
                p_offs,
                p_codes,
                t_offs,
                t_codes,
            )
        return cls(nc, nm, clo, chi, slo, shi, ald, wc, p_codes, p_offs, p_lens, t_codes, t_offs, t_lens)

    @classmethod
    def from_list(cls, glycans: Sequence[Glycan]) -> Self:
        """Compile a sequence of `Glycan` objects into a `GlycanFingerprints` SoA container."""
        return cls.from_glycans(Glycans.from_list(glycans))

    def filter_isomorphic_candidates(
        self, q_idx: int, cand_indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    ) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Return positions in `cand_indices` passing O(1) isomorphism pre-filters against `q_idx`."""
        return _filter_iso_batch_kernel(
            q_idx,
            cand_indices,
            self.num_core_nodes,
            self.core_mask_lo,
            self.core_mask_hi,
            self.has_wildcard_edges,
            self.pair_codes,
            self.pair_offsets,
            self.pair_lengths,
            self.triple_codes,
            self.offsets,
            self.lengths,
        )

    def filter_sub_or_oligo_candidates(
        self,
        q_idx: int,
        q_is_rep: bool,
        cand_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
        cand_is_rep: np.ndarray[tuple[int], np.dtype[np.bool_]],
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int32]], np.ndarray[tuple[int], np.dtype[np.int8]]]:
        """Return `(positions, mode_bitmasks)` in `cand_indices` passing O(1) subgraph/oligomer pre-filters."""
        return _filter_sub_or_oligo_batch_kernel(
            q_idx,
            q_is_rep,
            cand_indices,
            cand_is_rep,
            self.num_core_nodes,
            self.core_mask_lo,
            self.core_mask_hi,
            self.has_wildcard_edges,
            self.pair_codes,
            self.pair_offsets,
            self.pair_lengths,
            self.triple_codes,
            self.offsets,
            self.lengths,
        )

    def rank_order(self, records: Sequence[GlycanRecord]) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Compute descending consolidation rank order across `records` using vectorized `np.lexsort`."""
        n = len(records)
        if n == 0:
            return np.empty(0, dtype=np.int32)
        has_sero = np.fromiter((bool(r.serotype) for r in records), dtype=np.int8, count=n)
        is_rep = np.fromiter((r.is_repeating for r in records), dtype=np.int8, count=n)
        has_loc = np.fromiter((r.locus is not None for r in records), dtype=np.int8, count=n)
        align_sc = np.round(np.fromiter((r.alignment_score for r in records), dtype=np.float32, count=n), 3)
        db_prio = np.fromiter((_DB_PRIORITY.get(r.source_db, 0) for r in records), dtype=np.int8, count=n)
        has_pmid = np.fromiter((r.pmid > 0 for r in records), dtype=np.int8, count=n)
        pub_yr = np.fromiter((r.publication_year for r in records), dtype=np.int32, count=n)

        not_ald = (~self.is_alditol).astype(np.int8)
        intact = (is_rep & not_ald).astype(np.int8)
        n_core = self.num_core_nodes.astype(np.int32)
        n_mod = self.num_modifiers.astype(np.int32)
        n_nodes_rank = np.where(intact > 0, n_core, -n_core)

        order = np.lexsort(
            (
                pub_yr,
                has_pmid,
                db_prio,
                n_mod,
                n_nodes_rank,
                align_sc,
                has_loc,
                not_ald,
                is_rep,
                n_mod,
                n_core,
                intact,
                has_sero,
            )
        )
        return order[::-1].astype(np.int32)


def record_rank_key(
    rec: GlycanRecord, is_oligomer_duplicate: bool = False, fp: GlycanFingerprint | None = None
) -> tuple[Any, ...]:
    """Rank candidate records so intact 1x repeating units with loci and modifiers win."""
    assert rec.glycan is not None
    g_fp = fp if fp is not None else GlycanFingerprint.from_glycan(rec.glycan)
    intact = int(rec.is_repeating and not g_fp.is_alditol and not is_oligomer_duplicate)
    n_nodes = g_fp.num_core_nodes if intact else -g_fp.num_core_nodes
    db_prio = _DB_PRIORITY.get(rec.source_db, 0)
    return (
        intact,
        int(not is_oligomer_duplicate),
        int(rec.is_repeating),
        int(not g_fp.is_alditol),
        int(rec.locus is not None),
        round(float(rec.alignment_score), 3),
        n_nodes,
        g_fp.num_modifiers,
        db_prio,
        int(rec.pmid > 0),
        int(rec.publication_year),
    )


def _classify_isomorphic_relation(
    cand_gly: Glycan, cand_fp: GlycanFingerprint, lead_gly: Glycan, lead_fp: GlycanFingerprint
) -> EntryRelation:
    """Distinguish `ISOMORPH`, `CYCLIC_SHIFT`, and `MODIFIER_VARIANT` once isomorphism is confirmed."""
    if (
        cand_fp.num_modifiers != lead_fp.num_modifiers
        or cand_fp.sugar_mask_lo != lead_fp.sugar_mask_lo
        or cand_fp.sugar_mask_hi != lead_fp.sugar_mask_hi
    ):
        return EntryRelation.MODIFIER_VARIANT
    if cand_gly.is_isomorphic(
        lead_gly, ignore_reducing_end=False, ignore_modifiers=False, allow_open_repeat=False, allow_wildcards=True
    ):
        return EntryRelation.ISOMORPH
    if cand_fp.num_modifiers == 0 or cand_gly.is_isomorphic(
        lead_gly, ignore_reducing_end=True, ignore_modifiers=False, allow_open_repeat=True, allow_wildcards=True
    ):
        return EntryRelation.CYCLIC_SHIFT
    return EntryRelation.MODIFIER_VARIANT


@dataclass(frozen=True, slots=True)
class ClusterMember:
    """A constituent `GlycanRecord` tagged with its `GlycanFingerprint` and `EntryRelation` at insertion time."""

    record: GlycanRecord
    fp: GlycanFingerprint
    relation: EntryRelation


@dataclass(slots=True)
class RecordCluster:
    """Mutable cluster of consolidated `ClusterMember` items headed by the highest-ranked canonical representative."""

    lead: ClusterMember
    antigen_fam: bytes = b""
    members: list[ClusterMember] = field(default_factory=list)
    strains: set[bytes] = field(default_factory=set)
    pmids: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.antigen_fam:
            self.antigen_fam = antigen_family(self.lead.record.antigen_type)
        if not self.members:
            self.members.append(self.lead)
        if self.lead.record.strain:
            self.strains.add(self.lead.record.strain)
        if self.lead.record.pmid > 0:
            self.pmids.add(self.lead.record.pmid)

    def shares_context(self, rec: GlycanRecord, rec_fam: bytes) -> bool:
        """Return True if `rec` shares a non-empty `strain` or `pmid` and compatible antigen family with this cluster."""
        unknowns = (b"", b"Unknown")
        if self.antigen_fam != rec_fam and self.antigen_fam not in unknowns and rec_fam not in unknowns:
            return False
        return bool((rec.strain and rec.strain in self.strains) or (rec.pmid > 0 and rec.pmid in self.pmids))

    def classify_relation(self, rec: GlycanRecord, fp: GlycanFingerprint) -> EntryRelation:
        """Determine the `EntryRelation` of `rec` relative to `self.lead` using `fp` O(1) filters first."""
        r_gly, l_gly, l_fp = rec.glycan, self.lead.record.glycan, self.lead.fp
        assert r_gly is not None and l_gly is not None
        if fp.can_be_isomorphic(l_fp) and r_gly.is_isomorphic(
            l_gly, ignore_modifiers=True, allow_open_repeat=True, allow_wildcards=True
        ):
            return _classify_isomorphic_relation(r_gly, fp, l_gly, l_fp)
        if not rec.is_repeating and fp.can_be_oligomer_of(l_fp) and r_gly.is_oligomer_of(l_gly):
            return EntryRelation.OLIGOMER
        return EntryRelation.SUBGRAPH_FRAGMENT

    def add_member(self, rec: GlycanRecord, fp: GlycanFingerprint, relation: EntryRelation | None = None) -> None:
        """Insert `rec` into the cluster, promoting it to `lead` if `self.lead` is an oligomer of `rec`."""
        r_gly, l_gly = rec.glycan, self.lead.record.glycan
        assert r_gly is not None and l_gly is not None
        if rec.strain:
            self.strains.add(rec.strain)
        if rec.pmid > 0:
            self.pmids.add(rec.pmid)

        if (
            rec.is_repeating
            and not fp.is_alditol
            and self.lead.fp.can_be_oligomer_of(fp)
            and l_gly.is_oligomer_of(r_gly)
        ):
            self.members[0] = replace(self.lead, relation=EntryRelation.OLIGOMER)
            self.lead = ClusterMember(rec, fp, EntryRelation.CANONICAL)
            self.members.insert(0, self.lead)
            return

        rel = relation if relation is not None else self.classify_relation(rec, fp)
        cand_rank = record_rank_key(rec, is_oligomer_duplicate=(rel == EntryRelation.OLIGOMER), fp=fp)
        if cand_rank > record_rank_key(self.lead.record, is_oligomer_duplicate=False, fp=self.lead.fp):
            self.members[0] = replace(self.lead, relation=rel)
            self.lead = ClusterMember(rec, fp, EntryRelation.CANONICAL)
            self.members.insert(0, self.lead)
        else:
            self.members.append(ClusterMember(rec, fp, rel))

    def to_canonical_record(self) -> GlycanRecord:
        """Compile the cluster into a single canonical `GlycanRecord` without re-running isomorphism checks."""
        winner = self.lead.record
        if len(self.members) == 1:
            return winner if winner.entries else replace(winner, entries=(winner.to_entry(EntryRelation.CANONICAL),))

        winner_gly = winner.glycan
        assert winner_gly is not None
        others = [m for m in self.members if m.record is not winner]
        recs = [winner, *(m.record for m in others)]

        all_grafted: list[bytes] = []
        for m in others:
            o_gly = m.record.glycan
            if o_gly is not None and m.fp.num_modifiers > 0:
                if (m.fp.sugar_mask & ~winner_gly.sugar_mask) != 0 or m.fp.num_modifiers > winner_gly.num_modifiers:
                    winner_gly, grafted = winner_gly.graft_modifiers_from(o_gly)
                    all_grafted.extend(g for g in grafted if g not in all_grafted)

        ag_type = winner.antigen_type
        if ag_type in (b"CPS", b"O-antigen"):
            ag_type = next((r.antigen_type for r in recs if r.antigen_type.startswith(b"CPS (Group")), ag_type)

        loc_donor = next((r for r in recs if r.locus is not None), winner)
        locus = loc_donor.locus
        asm_id = (loc_donor.assembly_id if locus is not None and loc_donor.assembly_id else b"") or next(
            (r.assembly_id for r in recs if r.assembly_id), b""
        )
        has_loc_arch = locus is not None and loc_donor.archetype != LocusArchetype.UNKNOWN
        arch = (
            loc_donor.archetype
            if has_loc_arch
            else next((r.archetype for r in recs if r.archetype != LocusArchetype.UNKNOWN), LocusArchetype.UNKNOWN)
        )

        seen_keys: set[bytes] = set()
        raw_entries: list[GlycanEntry] = []
        for m in [self.lead, *others]:
            r_item = m.record
            rel = EntryRelation.CANONICAL if r_item.compound_key == winner.compound_key else m.relation
            item_ents = r_item.entries if r_item.entries else (r_item.to_entry(rel),)
            for ent in item_ents:
                if (ck := ent.compound_key) not in seen_keys:
                    seen_keys.add(ck)
                    is_win = ck == winner.compound_key
                    e_rel = EntryRelation.CANONICAL if is_win else (rel if len(item_ents) == 1 else ent.relation)
                    raw_entries.append(replace(ent, relation=e_rel))

        raw_entries.sort(
            key=lambda e: (0 if e.compound_key == winner.compound_key else 1, int(e.relation), e.compound_key)
        )
        notes = winner.notes
        if all_grafted and b"[grafted_modifiers=" not in notes:
            notes = (notes + b" [grafted_modifiers=+" + b",+".join(all_grafted) + b"]")[:256]

        al_score = loc_donor.alignment_score if locus is not None else winner.alignment_score
        return replace(
            winner,
            strain=next((r.strain for r in recs if r.strain), b""),
            serotype=next((r.serotype for r in recs if r.serotype), b""),
            antigen_type=ag_type,
            assembly_id=asm_id,
            archetype=arch,
            locus=locus,
            alignment_score=al_score,
            glycan=winner_gly,
            pmid=next((r.pmid for r in recs if r.pmid > 0), 0),
            publication_year=next((r.publication_year for r in recs if r.publication_year > 0), 0),
            doi=next((r.doi for r in recs if r.doi), b""),
            glytoucan_id=next((r.glytoucan_id for r in recs if r.glytoucan_id), b""),
            notes=notes,
            entries=tuple(raw_entries),
        )


def merge_record_cluster(cluster: list[GlycanRecord]) -> GlycanRecord:
    """Merge a list of duplicate/fragment records into the single highest-ranked canonical `GlycanRecord`."""
    if not cluster:
        raise ValueError("Cannot merge an empty cluster")
    valid = [r for r in cluster if r.glycan is not None]
    fps = GlycanFingerprints.from_list([r.glycan for r in valid if r.glycan is not None])
    order = fps.rank_order(valid)
    rc = RecordCluster(ClusterMember(valid[int(order[0])], fps[int(order[0])], EntryRelation.CANONICAL))
    for idx in order[1:]:
        i = int(idx)
        rc.add_member(valid[i], fps[i])
    return rc.to_canonical_record()


def consolidate_curated_records(records: list[GlycanRecord]) -> list[GlycanRecord]:
    """Consolidate duplicate, cyclic-shift, modifier-variant, oligomer, and fragment `GlycanRecord` entries in one pass."""
    valid_recs = [r for r in records if r.glycan is not None]
    if not valid_recs:
        return []

    fps = GlycanFingerprints.from_list([r.glycan for r in valid_recs if r.glycan is not None])
    order = fps.rank_order(valid_recs)

    all_clusters: list[RecordCluster] = []
    typed_by_key: dict[tuple[bytes, bytes, bytes], RecordCluster] = {}
    csdb_xref_index: dict[tuple[bytes, bytes], RecordCluster] = {}
    gtc_xref_index: dict[tuple[bytes, bytes], RecordCluster] = {}
    iso_index: dict[tuple[bytes, bytes, tuple[int, int, int]], list[RecordCluster]] = defaultdict(list)
    iso_index_org: dict[tuple[bytes, tuple[int, int, int]], list[RecordCluster]] = defaultdict(list)
    org_clusters: dict[bytes, list[RecordCluster]] = defaultdict(list)
    org_fam_typed_clusters: dict[tuple[bytes, bytes], list[RecordCluster]] = defaultdict(list)

    def _index_xrefs(rec: GlycanRecord, cl: RecordCluster) -> None:
        if rec.source_db == b"CSDB":
            csdb_xref_index.setdefault((rec.organism, rec.record_id), cl)
        for m_xref in _CSDB_XREF_RE.finditer(rec.notes):
            csdb_xref_index.setdefault((rec.organism, m_xref.group(1)), cl)
        if rec.glytoucan_id:
            gtc_xref_index.setdefault((rec.organism, rec.glytoucan_id), cl)

    def _new_cluster(rec: GlycanRecord, fp: GlycanFingerprint, fam: bytes) -> RecordCluster:
        cl = RecordCluster(ClusterMember(rec, fp, EntryRelation.CANONICAL), antigen_fam=fam)
        all_clusters.append(cl)
        iso_index[(rec.organism, fam, fp.iso_bucket_key)].append(cl)
        iso_index_org[(rec.organism, fp.iso_bucket_key)].append(cl)
        org_clusters[rec.organism].append(cl)
        if rec.serotype:
            typed_by_key[(rec.organism, rec.serotype, fam)] = cl
            org_fam_typed_clusters[(rec.organism, fam)].append(cl)
            _index_xrefs(rec, cl)
        return cl

    for raw_idx in order:
        q_idx = int(raw_idx)
        rec = valid_recs[q_idx]
        fp = fps[q_idx]
        org, fam, gly = rec.organism, antigen_family(rec.antigen_type), rec.glycan
        assert gly is not None

        if rec.serotype:
            t_key = (org, rec.serotype, fam)
            if t_key in typed_by_key:
                cl = typed_by_key[t_key]
                prev_bucket = cl.lead.fp.iso_bucket_key
                cl.add_member(rec, fp)
                _index_xrefs(rec, cl)
                if cl.lead.fp.iso_bucket_key != prev_bucket:
                    iso_index[(org, fam, cl.lead.fp.iso_bucket_key)].append(cl)
                    iso_index_org[(org, cl.lead.fp.iso_bucket_key)].append(cl)
            else:
                _new_cluster(rec, fp, fam)
            continue

        xref_cl = csdb_xref_index.get((org, rec.record_id)) if rec.source_db == b"CSDB" else None
        if xref_cl is None:
            xref_cl = next(
                (
                    csdb_xref_index[org, m.group(1)]
                    for m in _CSDB_XREF_RE.finditer(rec.notes)
                    if (org, m.group(1)) in csdb_xref_index
                ),
                None,
            )
        if xref_cl is None and rec.glytoucan_id:
            xref_cl = gtc_xref_index.get((org, rec.glytoucan_id))
        if xref_cl is not None:
            xref_cl.add_member(rec, fp)
            continue

        iso_cands = list(iso_index.get((org, fam, fp.iso_bucket_key), ()))
        if rec.strain or rec.pmid > 0:
            extra_iso = (
                c
                for c in iso_index_org.get((org, fp.iso_bucket_key), ())
                if c not in iso_cands and c.shares_context(rec, fam)
            )
            iso_cands.extend(extra_iso)

        if iso_cands:
            cand_fp_idx = np.fromiter((c.lead.fp.index for c in iso_cands), dtype=np.int32, count=len(iso_cands))
            matched_iso = False
            for k_pos in fps.filter_isomorphic_candidates(q_idx, cand_fp_idx):
                cl = iso_cands[int(k_pos)]
                l_gly, l_fp = cl.lead.record.glycan, cl.lead.fp
                assert l_gly is not None
                if gly.is_isomorphic(l_gly, ignore_modifiers=True, allow_open_repeat=True, allow_wildcards=True):
                    cl.add_member(rec, fp, _classify_isomorphic_relation(gly, fp, l_gly, l_fp))
                    matched_iso = True
                    break
            if matched_iso:
                continue

        placed = False
        if rec.strain or rec.pmid > 0:
            ctx_cands = [cl for cl in org_clusters.get(org, ()) if cl.shares_context(rec, fam)]
            if ctx_cands:
                c_idx = np.fromiter((c.lead.fp.index for c in ctx_cands), dtype=np.int32, count=len(ctx_cands))
                c_rep = np.fromiter(
                    (c.lead.record.is_repeating for c in ctx_cands), dtype=np.bool_, count=len(ctx_cands)
                )
                pos_arr, mode_arr = fps.filter_sub_or_oligo_candidates(q_idx, rec.is_repeating, c_idx, c_rep)
                for k_pos, mode in zip(pos_arr, mode_arr, strict=False):
                    cl = ctx_cands[int(k_pos)]
                    l_gly = cl.lead.record.glycan
                    assert l_gly is not None
                    if (mode & 1) and gly.is_subgraph_of(l_gly):
                        cl.add_member(rec, fp, EntryRelation.SUBGRAPH_FRAGMENT)
                        placed = True
                        break
                    if (mode & 2) and gly.is_oligomer_of(l_gly):
                        cl.add_member(rec, fp, EntryRelation.OLIGOMER)
                        placed = True
                        break
                    if (mode & 4) and l_gly.is_oligomer_of(gly):
                        cl.add_member(rec, fp, EntryRelation.SUBGRAPH_FRAGMENT)
                        placed = True
                        break
            if placed:
                continue

        if not rec.is_repeating and fp.num_core_nodes >= 3:
            typed_cands = org_fam_typed_clusters.get((org, fam), ())
            if typed_cands:
                t_idx = np.fromiter((c.lead.fp.index for c in typed_cands), dtype=np.int32, count=len(typed_cands))
                t_rep = np.zeros(len(typed_cands), dtype=np.bool_)
                pos_arr, mode_arr = fps.filter_sub_or_oligo_candidates(q_idx, False, t_idx, t_rep)
                matched_typed: list[tuple[RecordCluster, EntryRelation]] = []
                for k_pos, mode in zip(pos_arr, mode_arr, strict=False):
                    cl = typed_cands[int(k_pos)]
                    l_gly = cl.lead.record.glycan
                    assert l_gly is not None
                    if (mode & 1) and gly.is_subgraph_of(l_gly):
                        matched_typed.append((cl, EntryRelation.SUBGRAPH_FRAGMENT))
                    elif (mode & 2) and gly.is_oligomer_of(l_gly):
                        matched_typed.append((cl, EntryRelation.OLIGOMER))
                    if len(matched_typed) > 1:
                        break
                if len(matched_typed) == 1:
                    matched_typed[0][0].add_member(rec, fp, matched_typed[0][1])
                    continue

        _new_cluster(rec, fp, fam)

    return [cl.to_canonical_record() for cl in all_clusters]


# ======================================================================================================================
# 2. Vectorized `LociCacheBundle` (`RaggedArrayContainer`) & `SerotypeTokenIndex`
# ======================================================================================================================


@dataclass(frozen=True, slots=True)
class NuccoreSummary(ScalarView):
    """Normalized NCBI `nuccore` `esummary` document metadata for BGC candidate filtering."""

    uid: str
    accession_version: str
    title: str
    subname: str
    subtype: str
    is_cps: bool

    @property
    def base_accession(self) -> str:
        return self.accession_version.split(".")[0] if self.accession_version else ""

    @property
    def blob_text(self) -> str:
        return f"{self.title} | {self.subname}"

    @property
    def sub_pairs(self) -> list[tuple[str, str]]:
        return list(zip(self.subtype.split("|"), self.subname.split("|"), strict=False))


@dataclass(frozen=True, slots=True)
class SerotypeTokenIndex:
    """Compiled multi-tier lexical & regex index for matching serotype/strain tokens in NCBI summaries."""

    num_map: dict[str, tuple[str, ...]]
    word_map: dict[str, tuple[str, ...]]
    multi_map: dict[str, tuple[tuple[str, re.Pattern[str]], ...]]

    @classmethod
    def build(cls, targets: Iterable[str]) -> Self:
        """Compile a `SerotypeTokenIndex` from target serotype/strain strings."""
        num_acc: dict[str, list[str]] = defaultdict(list)
        word_acc: dict[str, list[str]] = defaultdict(list)
        multi_acc: dict[str, list[tuple[str, re.Pattern[str]]]] = defaultdict(list)

        for tok_s in targets:
            if not tok_s:
                continue
            if tok_s.isdigit():
                num_acc[tok_s.upper()].append(tok_s)
                continue
            if not (words := _WORD_SPLIT_RE.findall(tok_s)):
                continue
            if len(words) == 1 and words[0] == tok_s:
                u_tok = tok_s.upper()
                word_acc[u_tok].append(tok_s)
                if u_tok.startswith("K") and u_tok[1:].isdigit():
                    word_acc["KL" + u_tok[1:]].append(tok_s)
                elif u_tok.startswith("KL") and u_tok[2:].isdigit():
                    word_acc["K" + u_tok[2:]].append(tok_s)
                elif u_tok.startswith("O:") and u_tok[2:].isdigit():
                    word_acc["O" + u_tok[2:]].append(tok_s)
                    num_acc[u_tok[2:]].append(tok_s)
                elif u_tok.startswith("O") and u_tok[1:].isdigit():
                    word_acc["O:" + u_tok[1:]].append(tok_s)
                    num_acc[u_tok[1:]].append(tok_s)
                for sv_alias in OGROUP_TO_SALMONELLA_SEROVARS.get(u_tok, ()):
                    if len(sv_words := _WORD_SPLIT_RE.findall(sv_alias)) == 1:
                        word_acc[sv_words[0].upper()].append(tok_s)
            else:
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(tok_s)}(?![A-Za-z0-9])", re.I)
                multi_acc[max(words, key=len).upper()].append((tok_s, pat))

        return cls(
            {k: tuple(v) for k, v in num_acc.items()},
            {k: tuple(v) for k, v in word_acc.items()},
            {k: tuple(v) for k, v in multi_acc.items()},
        )

    def match(self, summary: NuccoreSummary) -> set[str]:
        """Return all indexed target tokens matched in `summary`."""
        matched: set[str] = set()
        blob_txt = summary.blob_text
        for w in {w.upper() for w in _WORD_SPLIT_RE.findall(blob_txt)}:
            if w in self.word_map:
                matched.update(self.word_map[w])
            for orig_tok, pat in self.multi_map.get(w, ()):
                if orig_tok not in matched and pat.search(blob_txt):
                    matched.add(orig_tok)
        if self.num_map:
            for m_num in _NUM_SERO_RE.findall(summary.title):
                if (u_num := m_num.upper()) in self.num_map:
                    matched.update(self.num_map[u_num])
            for st_k, sn_v in summary.sub_pairs:
                if st_k.strip().lower() in ("serotype", "serogroup", "serovar"):
                    u_sn = sn_v.strip().upper()
                    if u_sn in self.num_map:
                        matched.update(self.num_map[u_sn])
                    elif u_sn.startswith("O") and u_sn[1:] in self.num_map:
                        matched.update(self.num_map[u_sn[1:]])
        return matched


@dataclass(frozen=True, slots=True)
class LociCacheEntry(ScalarView):
    """Scalar view for a single cached `(cache_key, assembly_id, Loci)` row in `LociCacheBundle`."""

    cache_key: bytes
    assembly_id: bytes
    loci: Loci


@dataclass(slots=True)
class LociCacheBundle(RaggedArrayContainer[LociCacheEntry, "LociCacheBundle"]):
    """Zero-pickle `RaggedArrayContainer` storing flat `Loci` and offsets/lengths for zero-copy `.npz` load/save."""

    cache_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S96"))
    asm_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S32"))
    loci: Loci = field(default_factory=Loci.empty)
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    strain_orgs: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S64"))
    strain_toks: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S64"))
    strain_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S96"))
    doi_pmids: np.ndarray[tuple[int], np.dtype[np.int64]] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    doi_vals: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S96"))

    _key_to_idx: dict[bytes, int] | None = field(default=None, init=False, repr=False)
    _base_to_idx: dict[bytes, int] | None = field(default=None, init=False, repr=False)
    _entry_memo: dict[int, LociCacheEntry] = field(default_factory=dict, init=False, repr=False)
    _assembly_loci_dict: dict[bytes, tuple[bytes, Loci]] | None = field(default=None, init=False, repr=False)
    _strain_dict: dict[tuple[bytes, bytes], bytes] | None = field(default=None, init=False, repr=False)
    _doi_dict: dict[int, bytes] | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.offsets) if self._assembly_loci_dict is None else len(self._assembly_loci_dict)

    @property
    def is_warm(self) -> bool:
        if self._assembly_loci_dict is not None:
            return bool(self.assembly_loci and self.strain_to_assembly and self.pmid_to_doi)
        return len(self.cache_keys) > 0 and len(self.strain_keys) > 0 and len(self.doi_pmids) > 0

    def _ensure_indices(self) -> tuple[dict[bytes, int], dict[bytes, int]]:
        if self._key_to_idx is None or self._base_to_idx is None:
            k_map: dict[bytes, int] = {}
            b_map: dict[bytes, int] = {}
            for i, k_raw in enumerate(self.cache_keys):
                k_b = bytes(k_raw).rstrip(b"\x00")
                k_map[k_b] = i
                if not k_b.startswith((b"nuccore:", b"pmid:", b"uid:")):
                    b_map[k_b.split(b".")[0]] = i
            self._key_to_idx = k_map
            self._base_to_idx = b_map
        return self._key_to_idx, self._base_to_idx

    def _entry_at(self, idx: int) -> LociCacheEntry:
        if (cached := self._entry_memo.get(idx)) is not None:
            return cached
        off, ln = int(self.offsets[idx]), int(self.lengths[idx])
        loci_slice = self.loci[off : off + ln] if ln > 0 else Loci.empty()
        entry = LociCacheEntry(
            cache_key=bytes(self.cache_keys[idx]).rstrip(b"\x00"),
            assembly_id=bytes(self.asm_ids[idx]).rstrip(b"\x00"),
            loci=loci_slice,
        )
        self._entry_memo[idx] = entry
        return entry

    def get_entry(self, key: bytes, *, allow_base_acc: bool = False) -> LociCacheEntry | None:
        """Look up a `LociCacheEntry` in O(1) by `key` (or unversioned base accession), slicing `self.loci` on demand."""
        if self._assembly_loci_dict is not None:
            hit = self._assembly_loci_dict.get(key) or (self.base_acc_lookup.get(key) if allow_base_acc else None)
            return LociCacheEntry(key, hit[0], hit[1]) if hit is not None else None
        k_map, b_map = self._ensure_indices()
        idx = k_map.get(key)
        if idx is None and allow_base_acc:
            idx = b_map.get(key)
        return self._entry_at(idx) if idx is not None else None

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> LociCacheEntry: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> LociCacheBundle: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> LociCacheEntry | LociCacheBundle: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> LociCacheEntry | LociCacheBundle:
        if self._assembly_loci_dict is not None:
            self._sync_arrays_from_dicts()
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self.offsets)
            if idx < 0 or idx >= len(self.offsets):
                raise IndexError("LociCacheBundle index out of range")
            return self._entry_at(idx)

        indices, new_offsets, new_lengths, (loci_idx,) = self._slice_ragged(
            item, np.arange(len(self.loci), dtype=np.int32)
        )
        if len(indices) == 0:
            return self.empty(0)
        return LociCacheBundle(
            cache_keys=self.cache_keys[indices],
            asm_ids=self.asm_ids[indices],
            loci=self.loci[loci_idx] if len(loci_idx) > 0 else Loci.empty(),
            offsets=new_offsets,
            lengths=new_lengths,
            strain_orgs=self.strain_orgs,
            strain_toks=self.strain_toks,
            strain_keys=self.strain_keys,
            doi_pmids=self.doi_pmids,
            doi_vals=self.doi_vals,
        )

    @classmethod
    def empty(cls, length: int = 0) -> Self:
        z_i32 = np.zeros(length, dtype=np.int32) if length > 0 else np.empty(0, dtype=np.int32)
        return cls(
            cache_keys=np.zeros(length, dtype="S96") if length > 0 else np.empty(0, dtype="S96"),
            asm_ids=np.zeros(length, dtype="S32") if length > 0 else np.empty(0, dtype="S32"),
            loci=Loci.empty(),
            offsets=z_i32,
            lengths=z_i32.copy(),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        batch_list = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty(0)
        if len(batch_list) == 1:
            return batch_list[0]
        for b in batch_list:
            if b._assembly_loci_dict is not None:
                b._sync_arrays_from_dicts()
        new_lens = np.concatenate([b.lengths for b in batch_list])
        loci_non = [b.loci for b in batch_list if len(b.loci) > 0]
        return cls(
            cache_keys=np.concatenate([b.cache_keys for b in batch_list]),
            asm_ids=np.concatenate([b.asm_ids for b in batch_list]),
            loci=Loci.concat(loci_non) if loci_non else Loci.empty(),
            offsets=lengths_to_offsets(new_lens),
            lengths=new_lens,
            strain_orgs=np.concatenate([b.strain_orgs for b in batch_list if len(b.strain_orgs) > 0])
            if any(len(b.strain_orgs) > 0 for b in batch_list)
            else np.empty(0, dtype="S64"),
            strain_toks=np.concatenate([b.strain_toks for b in batch_list if len(b.strain_toks) > 0])
            if any(len(b.strain_toks) > 0 for b in batch_list)
            else np.empty(0, dtype="S64"),
            strain_keys=np.concatenate([b.strain_keys for b in batch_list if len(b.strain_keys) > 0])
            if any(len(b.strain_keys) > 0 for b in batch_list)
            else np.empty(0, dtype="S96"),
            doi_pmids=np.concatenate([b.doi_pmids for b in batch_list if len(b.doi_pmids) > 0])
            if any(len(b.doi_pmids) > 0 for b in batch_list)
            else np.empty(0, dtype=np.int64),
            doi_vals=np.concatenate([b.doi_vals for b in batch_list if len(b.doi_vals) > 0])
            if any(len(b.doi_vals) > 0 for b in batch_list)
            else np.empty(0, dtype="S96"),
        )

    @classmethod
    def from_dicts(
        cls,
        assembly_loci: dict[bytes, tuple[bytes, Loci]],
        strain_to_assembly: dict[tuple[bytes, bytes], bytes] | None = None,
        pmid_to_doi: dict[int, bytes] | None = None,
    ) -> Self:
        """Pack dictionaries into a vectorized `LociCacheBundle` using `Loci.concat` (zero per-Locus loops)."""
        items = list(assembly_loci.items())
        n = len(items)
        keys_arr = np.array([k[:96] for k, _ in items], dtype="S96") if n > 0 else np.empty(0, dtype="S96")
        asm_arr = np.array([a[:32] for _, (a, _) in items], dtype="S32") if n > 0 else np.empty(0, dtype="S32")
        lengths = np.fromiter((len(loc) for _, (_, loc) in items), dtype=np.int32, count=n)
        offsets = lengths_to_offsets(lengths)
        non_empty = [loc for _, (_, loc) in items if len(loc) > 0]
        flat_loci = Loci.concat(non_empty) if non_empty else Loci.empty()

        s_items = sorted((strain_to_assembly or {}).items())
        s_orgs = np.array([o[:64] for (o, _), _ in s_items], dtype="S64") if s_items else np.empty(0, dtype="S64")
        s_toks = np.array([t[:64] for (_, t), _ in s_items], dtype="S64") if s_items else np.empty(0, dtype="S64")
        s_keys = np.array([ck[:96] for _, ck in s_items], dtype="S96") if s_items else np.empty(0, dtype="S96")

        d_items = sorted((pmid_to_doi or {}).items())
        d_pmids = np.array([p for p, _ in d_items], dtype=np.int64) if d_items else np.empty(0, dtype=np.int64)
        d_vals = np.array([d[:96] for _, d in d_items], dtype="S96") if d_items else np.empty(0, dtype="S96")

        return cls(keys_arr, asm_arr, flat_loci, offsets, lengths, s_orgs, s_toks, s_keys, d_pmids, d_vals)

    def _sync_arrays_from_dicts(self) -> None:
        if self._assembly_loci_dict is None and self._strain_dict is None and self._doi_dict is None:
            return
        rebuilt = self.from_dicts(self.assembly_loci, self.strain_to_assembly, self.pmid_to_doi)
        self.cache_keys, self.asm_ids, self.loci = rebuilt.cache_keys, rebuilt.asm_ids, rebuilt.loci
        self.offsets, self.lengths = rebuilt.offsets, rebuilt.lengths
        self.strain_orgs, self.strain_toks, self.strain_keys = (
            rebuilt.strain_orgs,
            rebuilt.strain_toks,
            rebuilt.strain_keys,
        )
        self.doi_pmids, self.doi_vals = rebuilt.doi_pmids, rebuilt.doi_vals
        self._key_to_idx = None
        self._base_to_idx = None
        self._entry_memo.clear()
        self._assembly_loci_dict = None

    @property
    def assembly_loci(self) -> dict[bytes, tuple[bytes, Loci]]:
        if self._assembly_loci_dict is None:
            self._assembly_loci_dict = {
                (e := self._entry_at(i)).cache_key: (e.assembly_id, e.loci) for i in range(len(self.offsets))
            }
        return self._assembly_loci_dict

    @property
    def strain_to_assembly(self) -> dict[tuple[bytes, bytes], bytes]:
        if self._strain_dict is None:
            s_map: dict[tuple[bytes, bytes], bytes] = {
                (p[1], p[2]): k_b
                for k_raw in self.cache_keys
                if (k_b := bytes(k_raw).rstrip(b"\x00")).startswith(b"nuccore:")
                and len(p := k_b.split(b":", 2)) == 3
                and p[1]
                and p[2]
            }
            for o_r, t_r, ck_r in zip(self.strain_orgs, self.strain_toks, self.strain_keys, strict=False):
                o_b, t_b, ck_b = bytes(o_r).rstrip(b"\x00"), bytes(t_r).rstrip(b"\x00"), bytes(ck_r).rstrip(b"\x00")
                if o_b and t_b and ck_b:
                    s_map[(o_b, t_b)] = ck_b
            self._strain_dict = s_map
        return self._strain_dict

    @property
    def pmid_to_doi(self) -> dict[int, bytes]:
        if self._doi_dict is None:
            self._doi_dict = {
                int(p): bytes(d).rstrip(b"\x00")
                for p, d in zip(self.doi_pmids, self.doi_vals, strict=False)
                if int(p) > 0 and bytes(d).rstrip(b"\x00")
            }
        return self._doi_dict

    @property
    def base_acc_lookup(self) -> dict[bytes, tuple[bytes, Loci]]:
        return {
            k_acc.split(b".")[0]: val
            for k_acc, val in self.assembly_loci.items()
            if not k_acc.startswith((b"nuccore:", b"pmid:", b"uid:"))
        }

    def invalidate_index(self) -> None:
        self._key_to_idx = None
        self._base_to_idx = None
        self._entry_memo.clear()

    def as_tuple(self) -> tuple[dict[bytes, tuple[bytes, Loci]], dict[tuple[bytes, bytes], bytes], dict[int, bytes]]:
        return self.assembly_loci, self.strain_to_assembly, self.pmid_to_doi

    @classmethod
    def load(cls, path: Path = DEFAULT_LOCI_CACHE_PATH) -> Self:
        """Load `LociCacheBundle` from `.npz` in O(1) without per-key `Loci` slicing loops."""
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    keys, asm_ids, indptr = data["cache_keys"], data["asm_ids"], data["key_offsets"]
                    flat_loci = _loci_from_npz_dict(data, prefix="loci_")
                    s_orgs = data["strain_orgs"] if "strain_orgs" in data else np.empty(0, dtype="S64")
                    s_toks = data["strain_toks"] if "strain_toks" in data else np.empty(0, dtype="S64")
                    s_keys = data["strain_keys"] if "strain_keys" in data else np.empty(0, dtype="S96")
                    d_pmids = data["doi_pmids"] if "doi_pmids" in data else np.empty(0, dtype=np.int64)
                    d_vals = data["doi_vals"] if "doi_vals" in data else np.empty(0, dtype="S96")
                offsets = indptr[:-1].astype(np.int32, copy=False) if len(indptr) > 1 else np.empty(0, dtype=np.int32)
                lengths = (
                    np.diff(indptr).astype(np.int32, copy=False) if len(indptr) > 1 else np.empty(0, dtype=np.int32)
                )
                return cls(keys, asm_ids, flat_loci, offsets, lengths, s_orgs, s_toks, s_keys, d_pmids, d_vals)
            except Exception:
                pass

        if path == DEFAULT_LOCI_CACHE_PATH and DEFAULT_DB_PATH.is_file():
            try:
                db = GlycanRecords.load(DEFAULT_DB_PATH)
                c_lists: dict[bytes, list[Locus]] = defaultdict(list)
                c_asm: dict[bytes, bytes] = {}
                strain_map, doi_map = {}, {}
                for i in range(len(db)):
                    rec = db[i]
                    for p_v, d_v in ((rec.pmid, rec.doi), *((e.pmid, e.doi) for e in rec.entries)):
                        if p_v > 0 and d_v:
                            doi_map[p_v] = d_v
                    if rec.locus is None:
                        continue
                    asm_id = rec.assembly_id or (rec.locus.name.encode("utf-8")[:32] if rec.locus.name else b"locus")
                    hint_keys = (e.accession_hint for e in rec.entries if e.accession_hint)
                    pmid_keys = (b"pmid:%d" % e.pmid for e in rec.entries if e.pmid > 0)
                    for k in (asm_id, *hint_keys, *pmid_keys):
                        c_asm[k] = asm_id
                        if not any(
                            loc.name == rec.locus.name and len(loc.nodes) == len(rec.locus.nodes) for loc in c_lists[k]
                        ):
                            c_lists[k].append(rec.locus)
                    for tok in (
                        rec.strain,
                        rec.serotype,
                        *(e.strain for e in rec.entries),
                        *(e.serotype for e in rec.entries),
                    ):
                        if tok:
                            strain_map[(rec.organism, tok)] = asm_id
                bundle = cls.from_dicts(
                    {k: (c_asm[k], Loci.from_list(v)) for k, v in c_lists.items()}, strain_map, doi_map
                )
                bundle.save(path)
                return bundle
            except Exception:
                pass
        return cls.empty(0)

    def save(self, path: Path = DEFAULT_LOCI_CACHE_PATH) -> None:
        """Persist `LociCacheBundle` SoA arrays directly to `.npz` without pickle or per-Locus unpacking loops."""
        self._sync_arrays_from_dicts()
        if len(self.cache_keys) == 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            extra: dict[str, Any] = {}
            if len(self.strain_keys) > 0:
                extra["strain_orgs"] = self.strain_orgs
                extra["strain_toks"] = self.strain_toks
                extra["strain_keys"] = self.strain_keys
            if len(self.doi_pmids) > 0:
                extra["doi_pmids"] = self.doi_pmids
                extra["doi_vals"] = self.doi_vals
            tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
            loci_arrays = _loci_to_npz_dict(self.loci, prefix="loci_")
            np.savez_compressed(
                tmp_path,
                cache_keys=self.cache_keys,
                asm_ids=self.asm_ids,
                key_offsets=self.indptr,
                **extra,
                **loci_arrays,
            )
            os.replace(tmp_path, path)
        except Exception:
            pass


def load_predicted_loci_cache(path: Path = DEFAULT_LOCI_CACHE_PATH) -> dict[bytes, tuple[bytes, Loci]]:
    """Load zero-pickle cached `(asm_id, Loci)` entries from `.cache/predicted_loci.npz`."""
    return LociCacheBundle.load(path).assembly_loci


def _load_predicted_loci_cache_bundle(
    path: Path = DEFAULT_LOCI_CACHE_PATH,
) -> tuple[dict[bytes, tuple[bytes, Loci]], dict[tuple[bytes, bytes], bytes], dict[int, bytes]]:
    """Load `(assembly_loci_cache, strain_to_assembly, pmid_to_doi)` from `.cache/predicted_loci.npz`."""
    return LociCacheBundle.load(path).as_tuple()


def save_predicted_loci_cache(
    path: Path,
    cache: dict[bytes, tuple[bytes, Loci]],
    *,
    strain_to_assembly: dict[tuple[bytes, bytes], bytes] | None = None,
    pmid_to_doi: dict[int, bytes] | None = None,
) -> None:
    """Persist `(asm_id, Loci)` entries (plus optional strain/DOI maps) to `.cache/predicted_loci.npz` without pickle."""
    LociCacheBundle.from_dicts(cache, strain_to_assembly, pmid_to_doi).save(path)


# ======================================================================================================================
# 3. Shared NCBI E-Utilities Pipeline & 4-Tier Locus Resolution Hierarchy
# ======================================================================================================================


def _clean_fasta_acc(raw_header: bytes) -> str:
    """Extract clean NCBI accession token from a FASTA header line."""
    first_tok = raw_header.split()[0].decode("utf-8", errors="replace").lstrip(">")
    if "|" in first_tok:
        parts = [
            p for p in first_tok.split("|") if p and not p.isdigit() and p not in ("gb", "emb", "dbj", "ref", "gi")
        ]
        if parts:
            first_tok = parts[-1]
    return first_tok[:32]


def _catalog_taxon(org_b: bytes) -> str:
    """Normalize organism bytes into the taxon query string used for Tier 3 NCBI BGC cataloging."""
    if org_b in (b"Unknown bacterium", b"Bacteria (KEGG pathway)"):
        return ""
    parts = org_b.decode("utf-8", errors="replace").strip().removesuffix(" sp.").strip().split()
    if not parts:
        return ""
    return parts[0] if parts[0] in _GENUS_LEVEL_TAXA else " ".join(parts[:2])


def _is_true_strain_isolate(org_b: bytes, str_b: bytes, sero_b: bytes) -> bool:
    """Return True if `str_b` is a genuine strain identifier rather than a bare serotype label."""
    bad_orgs = (b"Unknown bacterium", b"Bacteria (KEGG pathway)")
    if not str_b or len(str_b) < 2 or str_b == sero_b or b"sp." in org_b or org_b in bad_orgs:
        return False
    pat = r"(?:type\s+|serotype\s+|serogroup\s+)?(?:K|KL|O|OL|OC|OCL|VI|\d{1,3}[A-Za-z]?|[IVX]{1,4}[abAB]?)"
    return not bool(re.fullmatch(pat, str_b.decode("utf-8", errors="replace").strip(), re.I))


@dataclass(frozen=True, slots=True)
class LocusQueryContext:
    """Harvested provenance context for a consolidated `GlycanRecord` during 4-tier cache key lookup."""

    record: GlycanRecord
    bundle: LociCacheBundle
    acc_candidates: tuple[bytes, ...]
    pmid_candidates: tuple[int, ...]
    strain_candidates: tuple[bytes, ...]

    @classmethod
    def from_record(cls, rec: GlycanRecord, bundle: LociCacheBundle) -> Self:
        accs: list[bytes] = [rec.assembly_id] if rec.assembly_id else []
        pmids: list[int] = [rec.pmid] if rec.pmid > 0 else []
        strains: list[bytes] = [rec.strain] if rec.strain else []
        for ent in rec.entries:
            if ent.accession_hint and ent.accession_hint not in accs:
                accs.append(ent.accession_hint)
            if ent.pmid > 0 and ent.pmid not in pmids:
                pmids.append(ent.pmid)
            if ent.strain and ent.strain not in strains:
                strains.append(ent.strain)
        return cls(rec, bundle, tuple(accs), tuple(pmids), tuple(strains))


@dataclass(slots=True)
class _DiscoveryState:
    """Shared mutable state across Tiers 1–4 during `resolve_and_predict_loci`."""

    bundle: LociCacheBundle
    base_acc_in_cache: dict[str, bytes]
    tier1_uids: set[str] = field(default_factory=set)
    pmid_to_nuccore_uids: dict[int, list[str]] = field(default_factory=dict)
    uid_to_summary: dict[str, NuccoreSummary] = field(default_factory=dict)
    uid_to_accver: dict[str, str] = field(default_factory=dict)
    selected_key_uids: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    uid_priority: Counter[str] = field(default_factory=Counter)
    uid_to_predicted: dict[str, tuple[bytes, list[Locus]]] = field(default_factory=dict)

    def pool_predicted_uids(self, uids: Iterable[str]) -> tuple[bytes, Loci] | None:
        """Aggregate predicted `Locus` items across `uids` into `(first_acc_b, Loci)`."""
        pool: list[Locus] = []
        first_acc = b""
        for u in uids:
            if u in self.uid_to_predicted:
                acc_b, l_list = self.uid_to_predicted[u]
                if not first_acc:
                    first_acc = acc_b
                pool.extend(l_list)
        return (first_acc, Loci.from_list(pool)) if pool and first_acc else None


@dataclass(slots=True)
class NCBIEUtilsClient:
    """Shared rate-limited NCBI E-Utilities client & multi-FASTA `LocusPredictor` runner."""

    client: Any
    sem: asyncio.Semaphore
    base_params: dict[str, str]
    api_key: str
    recorder: Any | None
    batch_predictor: LocusPredictor = field(
        default_factory=lambda: LocusPredictor(min_completeness=0.65, stitch_contig_splits=False)
    )
    whole_genome_predictor: LocusPredictor = field(default_factory=lambda: LocusPredictor(min_completeness=0.75))

    async def polite_sleep(self, t_req: float, delay: float = 0.22) -> None:
        elapsed = time.perf_counter() - t_req
        if not (self.recorder is not None and elapsed < 0.25) and elapsed > 0.05:
            await asyncio.sleep(delay)

    async def esearch(self, db: str, term: str, retmax: int = 100) -> list[str]:
        """Run rate-limited `esearch.fcgi` and return matching UID strings."""
        async with self.sem:
            t_req = time.perf_counter()
            try:
                params = {**self.base_params, "db": db, "term": term, "retmax": str(retmax)}
                resp = await self.client.get(f"{NCBI_EUTILS_URL}/esearch.fcgi", params=params)
                if resp.status_code == 200:
                    return [str(u) for u in resp.json().get("esearchresult", {}).get("idlist", [])]
            except Exception:
                pass
            finally:
                await self.polite_sleep(t_req)
        return []

    async def esummary(self, db: str, uids: Sequence[str], chunk_size: int = 200) -> dict[str, dict[str, Any]]:
        """Run batched `esummary.fcgi` POST requests and return `{uid: info_dict}`."""
        out: dict[str, dict[str, Any]] = {}

        async def _fetch_chunk(chunk: Sequence[str]) -> None:
            async with self.sem:
                t_req = time.perf_counter()
                try:
                    data = {**self.base_params, "db": db, "id": ",".join(chunk)}
                    resp = await self.client.post(f"{NCBI_EUTILS_URL}/esummary.fcgi", data=data)
                    if resp.status_code == 200:
                        res = resp.json().get("result", {})
                        for u in chunk:
                            if isinstance(info := res.get(u), dict):
                                out[u] = info
                except Exception:
                    pass
                finally:
                    await self.polite_sleep(t_req)

        chunks = [uids[i : i + chunk_size] for i in range(0, len(uids), chunk_size)]
        if chunks:
            await asyncio.gather(*[_fetch_chunk(c) for c in chunks])
        return out

    async def fetch_nuccore_summaries(
        self, uids: Sequence[str], *, slen_range: tuple[int, int] | None = (2500, 65000), chunk_size: int = 200
    ) -> dict[str, NuccoreSummary]:
        """Fetch `NuccoreSummary` records for `uids`, filtering out rRNA/housekeeping/phage non-BGC entries."""
        raw_map = await self.esummary("nuccore", uids, chunk_size=chunk_size)
        summaries: dict[str, NuccoreSummary] = {}
        bad_tokens = ("16s", "23s", "rrna", "gyrb", "rpob", "reca", "phage", "integron")
        cps_tokens = ("capsul", "cps", "kps", "k-antigen", "k antigen")
        for u, info in raw_map.items():
            if slen_range is not None and not (slen_range[0] <= int(info.get("slen") or 0) <= slen_range[1]):
                continue
            title = info.get("title", "")
            title_low = title.lower()
            if slen_range is not None and any(b in title_low for b in bad_tokens):
                continue
            if acc_ver := (info.get("accessionversion") or info.get("caption") or "").strip():
                is_cps = any(w in title_low for w in cps_tokens)
                summaries[u] = NuccoreSummary(
                    u, acc_ver, title, info.get("subname", ""), info.get("subtype", ""), is_cps
                )
        return summaries

    def predict_multi_fasta_batch(self, fasta_blob: bytes) -> dict[str, tuple[bytes, list[Locus]]]:
        """Segment a multi-FASTA nuccore batch into canonicalized `Locus` predictions keyed by base accession."""
        grouped: dict[str, tuple[bytes, list[Locus]]] = {}
        stripped = fasta_blob.lstrip()
        if not stripped.startswith(b">"):
            return grouped
        try:
            bgc_entries: list[bytes] = []
            for idx_e, chunk in enumerate(stripped.split(b"\n>")):
                entry = chunk if idx_e == 0 else b">" + chunk
                nl_pos = entry.find(b"\n")
                if nl_pos <= 1 or not (acc_tok := _clean_fasta_acc(entry[1:nl_pos])):
                    continue
                grouped.setdefault(acc_tok.split(".")[0], (acc_tok.encode("utf-8")[:32], []))
                if 1_500 <= (len(entry) - nl_pos) <= 95_000:
                    bgc_entries.append(entry)
            if not bgc_entries:
                return grouped
            asm = GenomeAssembly.from_fasta(b"\n".join(bgc_entries), id_=b"batch", truncate_id=True)
            if len(asm) == 0 or len(pred_loci := self.batch_predictor.predict(asm)) == 0:
                return grouped
            by_cidx: dict[int, list[Locus]] = defaultdict(list)
            for j in range(len(pred_loci)):
                loc = pred_loci[j].canonicalize_strand()
                by_cidx[int(loc.contig_index)].append(loc)
            for c_idx, loc_list in by_cidx.items():
                if 0 <= c_idx < len(asm.ids):
                    acc_tok = _clean_fasta_acc(bytes(asm.ids[c_idx]).strip(b"\x00").strip())
                    tagged = [replace(loc.with_name(acc_tok), contig_index=0) for loc in loc_list]
                    grouped[acc_tok.split(".")[0]] = (acc_tok.encode("utf-8")[:32], tagged)
        except Exception:
            pass
        return grouped

    async def fetch_and_predict_nuccore_uids(
        self, uids: Sequence[str], state: _DiscoveryState, *, batch_size: int = 40
    ) -> None:
        """Shared E-Utilities `efetch` -> `LocusPredictor` pipeline for all nuccore-based tiers."""
        empty_loci = Loci.empty()
        accbase_to_uids: dict[str, list[str]] = defaultdict(list)
        uids_to_fetch: list[str] = []

        for u in uids:
            base_av = state.uid_to_accver.get(u, "").split(".")[0]
            if base_av:
                accbase_to_uids[base_av].append(u)
            uid_key = b"uid:" + u.encode("utf-8")
            cache_k = state.base_acc_in_cache[base_av] if base_av and base_av in state.base_acc_in_cache else uid_key
            if (hit := state.bundle.assembly_loci.get(cache_k)) is not None:
                if len(hit[1]) > 0:
                    state.uid_to_predicted[u] = (hit[0], [hit[1][j] for j in range(len(hit[1]))])
            else:
                uids_to_fetch.append(u)

        for i in range(0, len(uids_to_fetch), batch_size):
            batch_uids = uids_to_fetch[i : i + batch_size]
            async with self.sem:
                t_req = time.perf_counter()
                try:
                    params = {"db": "nuccore", "id": ",".join(batch_uids), "rettype": "fasta", "retmode": "text"}
                    if self.api_key:
                        params["api_key"] = self.api_key
                    resp = await self.client.get(f"{NCBI_EUTILS_URL}/efetch.fcgi", params=params, timeout=40.0)
                    if resp.status_code == 200 and resp.content.startswith(b">"):
                        for base_tok, (acc_b, tagged_list) in self.predict_multi_fasta_batch(resp.content).items():
                            loci_obj = Loci.from_list(tagged_list) if tagged_list else empty_loci
                            state.bundle.assembly_loci[acc_b] = (acc_b, loci_obj)
                            state.base_acc_in_cache[base_tok] = acc_b
                            for matched_u in accbase_to_uids.get(base_tok, ()):
                                state.bundle.assembly_loci[b"uid:" + matched_u.encode("utf-8")] = (acc_b, loci_obj)
                                if tagged_list:
                                    state.uid_to_predicted[matched_u] = (acc_b, tagged_list)
                        for u_b in batch_uids:
                            u_key = b"uid:" + u_b.encode("utf-8")
                            if u_key not in state.bundle.assembly_loci:
                                base_u = state.uid_to_accver.get(u_b, "").split(".")[0]
                                acc_u_b = base_u.encode("utf-8")[:32] if base_u else b"empty"
                                state.bundle.assembly_loci[u_key] = (acc_u_b, empty_loci)
                                if base_u and base_u not in state.base_acc_in_cache:
                                    state.bundle.assembly_loci[acc_u_b] = (acc_u_b, empty_loci)
                                    state.base_acc_in_cache[base_u] = acc_u_b
                except Exception:
                    pass
                finally:
                    await self.polite_sleep(t_req)
        state.bundle.invalidate_index()


class LocusResolutionTier(ABC):
    """Abstract base class for a KBank genomic locus discovery and cache-key lookup tier."""

    @abstractmethod
    async def discover(
        self, raw_records: list[dict[str, Any]], eutils: NCBIEUtilsClient, state: _DiscoveryState
    ) -> None:
        """Discover candidate NCBI UIDs or assemblies and populate `state`."""

    @abstractmethod
    def iter_cache_keys(self, ctx: LocusQueryContext) -> Iterator[tuple[bytes, bool]]:
        """Yield `(cache_key, is_untagged_pool)` candidates in priority order for `ctx`."""


class ExplicitAccessionTier(LocusResolutionTier):
    """Tier 1: Explicit GenBank/RefSeq nucleotide or assembly accession hints."""

    async def discover(
        self, raw_records: list[dict[str, Any]], eutils: NCBIEUtilsClient, state: _DiscoveryState
    ) -> None:
        explicit_nuccore = {
            acc_s
            for r in raw_records
            if (acc_b := r.get("accession_hint", b""))
            and len(acc_s := acc_b.decode("utf-8", errors="replace").strip()) >= 6
            and not acc_s.startswith(("GCF_", "GCA_", "NC_", "NZ_"))
        }
        uncached = [acc for acc in sorted(explicit_nuccore)[:300] if acc.split(".")[0] not in state.base_acc_in_cache]
        for i in range(0, len(uncached), 35):
            clause = " OR ".join(f"{acc}[ACCN]" for acc in uncached[i : i + 35])
            state.tier1_uids.update(await eutils.esearch("nuccore", f"({clause}) AND 2000:85000[SLEN]", retmax=100))

    def iter_cache_keys(self, ctx: LocusQueryContext) -> Iterator[tuple[bytes, bool]]:
        for acc_hint in ctx.acc_candidates:
            yield acc_hint, False
            if (base_acc := acc_hint.split(b".")[0]) != acc_hint:
                yield base_acc, False


class PubMedElinkTier(LocusResolutionTier):
    """Tier 1.5 (token-matched PubMed->nuccore) and Tier 4 (untagged PubMed paper BGC pool)."""

    def __init__(self, *, untagged_fallback: bool = False) -> None:
        self.untagged_fallback = untagged_fallback

    async def discover(
        self, raw_records: list[dict[str, Any]], eutils: NCBIEUtilsClient, state: _DiscoveryState
    ) -> None:
        if self.untagged_fallback:
            return
        pmid_weights: Counter[int] = Counter(
            p_val
            for r in raw_records
            if (p_val := int(r.get("pmid", 0))) > 0
            and not (
                r.get("accession_hint", b"")
                and r["accession_hint"].decode("utf-8", errors="replace").split(".")[0] in state.base_acc_in_cache
            )
        )
        top_pmids = [p for p, _ in pmid_weights.most_common(600)]

        async def _elink_batch(batch_pmids: list[int]) -> None:
            async with eutils.sem:
                t_req = time.perf_counter()
                try:
                    data = {
                        **eutils.base_params,
                        "dbfrom": "pubmed",
                        "db": "nuccore",
                        "linkname": "pubmed_nuccore",
                        "id": [str(p) for p in batch_pmids],
                    }
                    r_el = await eutils.client.post(f"{NCBI_EUTILS_URL}/elink.fcgi", data=data)
                    if r_el.status_code == 200:
                        for ls in r_el.json().get("linksets", []):
                            if ids_l := ls.get("ids", []):
                                for db_l in ls.get("linksetdbs", []):
                                    if (
                                        db_l.get("linkname") == "pubmed_nuccore"
                                        and 1 <= len(links := [str(u) for u in db_l.get("links", [])]) <= 45
                                    ):
                                        state.pmid_to_nuccore_uids[int(ids_l[0])] = links[:20]
                except Exception:
                    pass
                finally:
                    await eutils.polite_sleep(t_req)

        await asyncio.gather(*[_elink_batch(top_pmids[i : i + 60]) for i in range(0, len(top_pmids), 60)])
        elink_uids = sorted(state.tier1_uids | {u for u_list in state.pmid_to_nuccore_uids.values() for u in u_list})
        for u, s_obj in (
            await eutils.fetch_nuccore_summaries(elink_uids, slen_range=(2500, 65000), chunk_size=200)
        ).items():
            state.uid_to_summary[u] = s_obj
            state.uid_to_accver[u] = s_obj.accession_version

        for u in state.tier1_uids:
            if u in state.uid_to_summary:
                state.uid_priority[u] += 50
        for _, u_list in sorted(state.pmid_to_nuccore_uids.items()):
            for idx_u, u in enumerate([u for u in u_list if u in state.uid_to_summary][:8]):
                state.uid_priority[u] += 25 - idx_u

    def populate_cache_from_predictions(self, raw_records: list[dict[str, Any]], state: _DiscoveryState) -> None:
        """Populate `pmid:{p}` and token-matched `pmid:{p}:{tok}` cache keys after nuccore FASTA prediction."""
        for p_val, u_list in state.pmid_to_nuccore_uids.items():
            if pooled := state.pool_predicted_uids([u for u in u_list if u in state.uid_to_predicted][:8]):
                state.bundle.assembly_loci[b"pmid:%d" % p_val] = pooled

        for r in raw_records:
            p_val = int(r.get("pmid", 0))
            pred_u = [
                u
                for u in state.pmid_to_nuccore_uids.get(p_val, ())
                if u in state.uid_to_predicted and u in state.uid_to_summary
            ]
            if p_val <= 0 or not pred_u:
                continue
            if not (
                r_toks := {t.decode("utf-8", errors="replace").strip(): t for t in (r["serotype"], r["strain"]) if t}
            ):
                continue
            tok_index = SerotypeTokenIndex.build(r_toks.keys())
            tok_to_uids: dict[bytes, list[str]] = defaultdict(list)
            for u in pred_u:
                for m_s in tok_index.match(state.uid_to_summary[u]):
                    tok_to_uids[r_toks[m_s]].append(u)
            for tok_b, matched_uids in tok_to_uids.items():
                if pooled := state.pool_predicted_uids(matched_uids):
                    state.bundle.assembly_loci.setdefault(b"pmid:%d:%s" % (p_val, tok_b), pooled)

    def iter_cache_keys(self, ctx: LocusQueryContext) -> Iterator[tuple[bytes, bool]]:
        if not self.untagged_fallback:
            for pmid_cand in ctx.pmid_candidates:
                for tok_b in (ctx.record.serotype, *ctx.strain_candidates):
                    if tok_b:
                        yield b"pmid:%d:%s" % (pmid_cand, tok_b), False
        else:
            for pmid_cand in ctx.pmid_candidates:
                yield b"pmid:%d" % pmid_cand, True


class StrainAssemblyTier(LocusResolutionTier):
    """Tier 2: Strain whole-genome RefSeq/GenBank assemblies (with BioSample culture-collection synonym resolution)."""

    def __init__(self, max_assemblies: int = 100) -> None:
        self.max_assemblies = max_assemblies

    async def discover(
        self, raw_records: list[dict[str, Any]], eutils: NCBIEUtilsClient, state: _DiscoveryState
    ) -> None:
        strain_counts: Counter[tuple[bytes, bytes]] = Counter(
            (r["organism"], r["strain"])
            for r in raw_records
            if _is_true_strain_isolate(r["organism"], r["strain"], r["serotype"])
        )
        top_queries = [
            pair for pair, _ in sorted(strain_counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_assemblies]
        ]
        uid_to_strain: dict[str, tuple[bytes, bytes]] = {}
        unresolved_collections: list[tuple[bytes, bytes]] = []

        async def _search_strain(org_b: bytes, str_b: bytes) -> None:
            org_s, str_s = org_b.decode("utf-8", errors="replace"), str_b.decode("utf-8", errors="replace")
            q_asm = f'"{org_s}"[Organism] AND "{str_s}"[Infraspecies] AND "latest refseq"[Filter]'
            if ids := await eutils.esearch("assembly", q_asm, retmax=1):
                uid_to_strain[ids[0]] = (org_b, str_b)
            elif _COLLECTION_RE.search(str_s):
                unresolved_collections.append((org_b, str_b))

        await asyncio.gather(*[_search_strain(o_b, s_b) for o_b, s_b in top_queries])
        biosample_uid_to_strain: dict[str, tuple[bytes, bytes]] = {}

        async def _search_biosample(org_b: bytes, str_b: bytes) -> None:
            org_s, str_s = org_b.decode("utf-8", errors="replace"), str_b.decode("utf-8", errors="replace")
            if bs_ids := await eutils.esearch("biosample", f'"{org_s}"[Organism] AND "{str_s}"', retmax=1):
                biosample_uid_to_strain[bs_ids[0]] = (org_b, str_b)

        await asyncio.gather(*[_search_biosample(o_b, s_b) for o_b, s_b in unresolved_collections[:30]])
        if biosample_uid_to_strain:
            bs_summaries = await eutils.esummary("biosample", sorted(biosample_uid_to_strain.keys()), chunk_size=200)
            for b_uid, b_info in bs_summaries.items():
                b_blob = f"{b_info.get('title', '')} {b_info.get('infraspecies', '')} {b_info.get('sampledata', '')}"
                if m_syn := re.search(r"\b((?:ATCC|NCTC|DSM|CCUG)\s*\d+)\b", b_blob, re.I):
                    syn_s = re.sub(r"\s+", " ", m_syn.group(1).strip())
                    org_b, orig_str_b = biosample_uid_to_strain[b_uid]
                    org_s = org_b.decode("utf-8", errors="replace")
                    q_syn = f'"{org_s}"[Organism] AND "{syn_s}"[Infraspecies] AND "latest refseq"[Filter]'
                    if a_ids := await eutils.esearch("assembly", q_syn, retmax=1):
                        uid_to_strain[a_ids[0]] = (org_b, orig_str_b)

        if not uid_to_strain:
            return
        empty_loci = Loci.empty()
        for uid, info in (await eutils.esummary("assembly", sorted(uid_to_strain.keys()), chunk_size=200)).items():
            acc_str = info.get("assemblyaccession", "")
            ftp_url = info.get("ftppath_refseq") or info.get("ftppath_genbank") or ""
            if not (acc_str and ftp_url):
                continue
            acc_b = acc_str.encode("utf-8")[:32]
            if acc_b not in state.bundle.assembly_loci:
                https_base = ftp_url.replace("ftp://", "https://").rstrip("/")
                fna_url = f"{https_base}/{https_base.split('/')[-1]}_genomic.fna.gz"
                async with eutils.sem:
                    try:
                        f_resp = await eutils.client.get(fna_url, timeout=45.0)
                        if f_resp.status_code == 200 and len(f_resp.content) > 100:
                            asm_obj = GenomeAssembly.from_fasta(gzip.decompress(f_resp.content), id_=acc_b)
                            pred = eutils.whole_genome_predictor.predict(asm_obj)
                            tagged = (
                                Loci.from_list([loc.canonicalize_strand().with_name(acc_str[:32]) for loc in pred])
                                if len(pred) > 0
                                else empty_loci
                            )
                            state.bundle.assembly_loci[acc_b] = (acc_b, tagged)
                    except Exception:
                        pass
            if acc_b in state.bundle.assembly_loci and len(state.bundle.assembly_loci[acc_b][1]) > 0:
                state.bundle.strain_to_assembly[uid_to_strain[uid]] = acc_b
        state.bundle.invalidate_index()

    def iter_cache_keys(self, ctx: LocusQueryContext) -> Iterator[tuple[bytes, bool]]:
        for st_cand in ctx.strain_candidates:
            if (cache_key := ctx.bundle.strain_to_assembly.get((ctx.record.organism, st_cand))) is not None:
                yield cache_key, False


class TaxonCatalogTier(LocusResolutionTier):
    """Tier 3: Taxon-wide `nuccore` BGC title search and landmark sweeps indexed by `SerotypeTokenIndex`."""

    _BGC_TITLE_FILTER = (
        "AND 3000:60000[SLEN] AND ("
        '"capsular"[Title] OR "capsule"[Title] OR "cps"[Title] OR "kps"[Title] '
        'OR "K-antigen"[Title] OR "O-antigen"[Title] OR "lipopolysaccharide"[Title] '
        'OR "lipooligosaccharide"[Title] OR "polysaccharide"[Title] OR "exopolysaccharide"[Title] '
        'OR "teichoic"[Title] OR "outer core"[Title] OR "B-band"[Title] '
        'OR "O antigen"[Title] OR "K antigen"[Title] OR "wb"[Title] OR "waa"[Title] OR "wba"[Title]'
        ') NOT "whole genome shotgun"[Title] NOT "patent"[Filter] NOT "targeted locus study"[Title] '
        'NOT "pyocin"[Title] NOT "16S"[Title] NOT "plasmid"[Title] NOT "phage"[Title]'
    )
    _LANDMARK_SWEEPS: dict[str, str] = {
        "Streptococcus pneumoniae": "CR931632:CR931722[ACCN]",
        "Pseudomonas aeruginosa": "AF498400:AF498420[ACCN]",
        "Klebsiella": "AB924547:AB924624[ACCN]",
    }

    async def discover(
        self, raw_records: list[dict[str, Any]], eutils: NCBIEUtilsClient, state: _DiscoveryState
    ) -> None:
        taxon_counts: Counter[str] = Counter()
        taxon_targets: dict[str, set[str]] = defaultdict(set)
        key_weights: Counter[tuple[str, str]] = Counter()

        for r in raw_records:
            if (
                r.get("accession_hint", b"")
                and r["accession_hint"].decode("utf-8", errors="replace").split(".")[0] in state.base_acc_in_cache
            ):
                continue
            org_b, str_b, sero_b = r["organism"], r["strain"], r["serotype"]
            if (org_b, str_b) in state.bundle.strain_to_assembly or not (taxon_q := _catalog_taxon(org_b)):
                continue
            taxon_counts[taxon_q] += 1
            if sero_b:
                sero_s = sero_b.decode("utf-8", errors="replace").strip()
                taxon_targets[taxon_q].add(sero_s)
                key_weights[(taxon_q, sero_s)] += 1
                if m_base := re.match(r"^([KO]\d+)[a-z].*$", sero_s):
                    taxon_targets[taxon_q].add(m_base.group(1))
                    key_weights[(taxon_q, m_base.group(1))] += 1
            if (
                str_b
                and len(str_b) >= 3
                and str_b != sero_b
                and not (str_s := str_b.decode("utf-8", errors="replace").strip()).isdigit()
            ):
                taxon_targets[taxon_q].add(str_s)
                key_weights[(taxon_q, str_s)] += 1

        key_to_candidates: dict[tuple[str, str], list[tuple[str, str, bool]]] = defaultdict(list)

        async def _catalog_taxon_uids(taxon_q: str) -> None:
            idlist = await eutils.esearch("nuccore", f'"{taxon_q}"[Organism] {self._BGC_TITLE_FILTER}', retmax=1000)
            if taxon_q in self._LANDMARK_SWEEPS:
                seen = set(idlist)
                for lid in await eutils.esearch("nuccore", self._LANDMARK_SWEEPS[taxon_q], retmax=200):
                    if lid not in seen:
                        idlist.append(lid)
                        seen.add(lid)
            if not idlist:
                return
            summaries = await eutils.fetch_nuccore_summaries(idlist, slen_range=None, chunk_size=350)
            tok_index = SerotypeTokenIndex.build(taxon_targets[taxon_q])
            for uid in idlist:
                if (s_obj := summaries.get(uid)) is not None:
                    state.uid_to_accver[uid] = s_obj.accession_version
                    for m_tok in tok_index.match(s_obj):
                        cand_list = key_to_candidates[(taxon_q, m_tok)]
                        if not any(u == uid for u, _, _ in cand_list):
                            cand_list.append((uid, s_obj.accession_version, s_obj.is_cps))

        await asyncio.gather(*[_catalog_taxon_uids(tq) for tq, _ in taxon_counts.most_common()])
        for key_pair, cands in sorted(key_to_candidates.items()):
            merged: list[str] = []
            for u in [u for u, _, is_c in cands if is_c][:2] + [u for u, _, is_c in cands if not is_c][:2]:
                if u not in merged and len(merged) < 3:
                    merged.append(u)
            if merged:
                state.selected_key_uids[key_pair] = merged
                w = key_weights.get(key_pair, 1)
                for idx_u, u in enumerate(merged):
                    state.uid_priority[u] += w * (3 - idx_u)

    def populate_cache_from_predictions(self, raw_records: list[dict[str, Any]], state: _DiscoveryState) -> None:
        """Populate `nuccore:{taxon}:{tok}` cache keys and `strain_to_assembly` entries from predicted BGCs."""
        for r in raw_records:
            org_b = r["organism"]
            if not (taxon_q := _catalog_taxon(org_b)):
                continue
            for tok_b in (r["strain"], r["serotype"]):
                if not tok_b or (org_b, tok_b) in state.bundle.strain_to_assembly:
                    continue
                tok_s = tok_b.decode("utf-8", errors="replace").strip()
                uids = state.selected_key_uids.get((taxon_q, tok_s), [])
                if not uids and (m_base := re.match(r"^([KO]\d+)[a-z].*$", tok_s)):
                    uids = state.selected_key_uids.get((taxon_q, m_base.group(1)), [])
                if uids and (pooled := state.pool_predicted_uids(uids)):
                    cache_key = b"nuccore:%s:%s" % (taxon_q.encode("utf-8"), tok_b)
                    state.bundle.assembly_loci.setdefault(cache_key, pooled)
                    state.bundle.strain_to_assembly[(org_b, tok_b)] = cache_key

    def iter_cache_keys(self, ctx: LocusQueryContext) -> Iterator[tuple[bytes, bool]]:
        if (
            ctx.record.serotype
            and (cache_key := ctx.bundle.strain_to_assembly.get((ctx.record.organism, ctx.record.serotype))) is not None
        ):
            yield cache_key, False


async def resolve_and_predict_loci(
    raw_records: list[dict[str, Any]],
    max_assemblies: int = 100,
    *,
    recorder: Any | None = None,
    loci_cache_path: Path = DEFAULT_LOCI_CACHE_PATH,
) -> tuple[dict[bytes, tuple[bytes, Loci]], dict[tuple[bytes, bytes], bytes], dict[int, bytes]]:
    """Fetch in-memory `GenomeAssembly` FASTAs from NCBI and predict candidate `Loci` via `LocusPredictor`."""
    import httpx2

    t_stage0 = time.perf_counter()
    bundle = LociCacheBundle.load(loci_cache_path)
    if bundle.is_warm and os.environ.get("KBANK_FORCE_LOCI_REFRESH", "0") != "1":
        n_l, n_s, n_d = len(bundle.cache_keys), len(bundle.strain_to_assembly), len(bundle.doi_pmids)
        print(
            f"[KBank] Loaded warm predicted_loci bundle ({n_l} keys, {n_s} strains, {n_d} DOIs) in {time.perf_counter() - t_stage0:.2f}s",
            flush=True,
        )
        return bundle.as_tuple()

    base_in_cache = {
        k.decode("utf-8", errors="replace").split(".")[0]: k
        for k in bundle.assembly_loci
        if not k.startswith((b"nuccore:", b"pmid:", b"uid:"))
    }
    state = _DiscoveryState(bundle=bundle, base_acc_in_cache=base_in_cache)
    api_key = os.environ.get("NCBI_API_KEY", "")
    base_params = {"retmode": "json", **({"api_key": api_key} if api_key else {})}
    tier1, tier1_5, tier2, tier3 = (
        ExplicitAccessionTier(),
        PubMedElinkTier(untagged_fallback=False),
        StrainAssemblyTier(max_assemblies),
        TaxonCatalogTier(),
    )

    with _cassette_ctx(recorder, "ncbi_eutils.toml"):
        async with httpx2.AsyncClient(timeout=35.0, follow_redirects=True) as http_client:
            eutils = NCBIEUtilsClient(
                http_client, asyncio.Semaphore(5 if api_key else 3), base_params, api_key, recorder
            )
            await tier1.discover(raw_records, eutils, state)
            await tier1_5.discover(raw_records, eutils, state)
            await tier2.discover(raw_records, eutils, state)
            bundle.save(loci_cache_path)
            await tier3.discover(raw_records, eutils, state)

            top_uids = [u for u, _ in sorted(state.uid_priority.items(), key=lambda kv: (-kv[1], kv[0]))[:1200]]
            await eutils.fetch_and_predict_nuccore_uids(top_uids, state)
            tier1_5.populate_cache_from_predictions(raw_records, state)
            tier3.populate_cache_from_predictions(raw_records, state)

            unique_pmids = [
                str(p)
                for p in sorted(
                    {int(r["pmid"]) for r in raw_records if int(r.get("pmid", 0)) > 0 and not r.get("doi")}
                )[:800]
            ]
            for pid_s, info in (await eutils.esummary("pubmed", unique_pmids, chunk_size=150)).items():
                for aid in info.get("articleids", []):
                    if aid.get("idtype") == "doi" and aid.get("value"):
                        bundle.pmid_to_doi[int(pid_s)] = aid["value"].strip().encode("utf-8")
                        break

    bundle.save(loci_cache_path)
    print(f"[KBank] resolve_and_predict_loci finished in {time.perf_counter() - t_stage0:.1f}s", flush=True)
    return bundle.as_tuple()


# ======================================================================================================================
# 4. Top-Down Locus Archetype Verification & Database Compilation
# ======================================================================================================================


def resolve_top_down_locus_archetype(
    locus: Locus,
    antigen_type: bytes,
    expected_cat: ArchetypeCategory | None,
    expected_arch: LocusArchetype | None,
    default_arch: LocusArchetype,
) -> tuple[LocusArchetype, bool]:
    """Verify candidate `locus` compatibility top-down against `expected_cat`/`expected_arch` and resolve its archetype."""
    roles = [GeneRole(int(r)) for r in locus.roles]
    rmask = roles_to_mask(roles)
    has_wzy = bool(
        rmask & roles_to_mask((GeneRole.FLIPPASE, GeneRole.POLYMERASE, GeneRole.COPOLYMERASE, GeneRole.LIGASE))
    )
    has_abc = bool(rmask & roles_to_mask((GeneRole.ABC_PERMEASE, GeneRole.ABC_ATPASE)))
    has_syn = bool(rmask & GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask)

    if (
        len(locus.genes) == len(locus.nodes)
        and 0 < len(locus.nodes) <= 8
        and not (rmask & roles_to_mask((GeneRole.POLYMERASE, GeneRole.TRANSLOCON, GeneRole.ABC_ATPASE)))
    ):
        phage_probs = _cached_locus_phage_probs(locus)
        if phage_probs is not None and len(phage_probs) > 0 and float(np.mean(phage_probs)) >= 0.60:
            return default_arch, False

    if expected_arch == LocusArchetype.TEICHOIC_ACID or antigen_type == b"WTA":
        return LocusArchetype.TEICHOIC_ACID, (rmask & LocusArchetype.TEICHOIC_ACID.spec.forbidden_mask) == 0
    if expected_arch == LocusArchetype.CPS_ABC:
        return LocusArchetype.CPS_ABC, has_abc
    if expected_arch == LocusArchetype.CPS_WZY:
        return LocusArchetype.CPS_WZY, has_wzy

    if expected_cat == ArchetypeCategory.O_ANTIGEN:
        if expected_arch == LocusArchetype.LPS_ABC:
            return LocusArchetype.LPS_ABC, has_abc
        if locus.archetype in (LocusArchetype.LPS_WZY, LocusArchetype.LPS_ABC):
            return locus.archetype, True
        return (
            (LocusArchetype.LPS_WZY, True)
            if has_wzy
            else ((LocusArchetype.LPS_ABC, True) if has_abc else (default_arch, False))
        )

    if expected_cat == ArchetypeCategory.CAPSULE:
        if locus.archetype in (LocusArchetype.CPS_WZY, LocusArchetype.CPS_ABC, LocusArchetype.CPS_SYNTHASE):
            return locus.archetype, True
        if has_syn and not has_wzy and not has_abc:
            return LocusArchetype.CPS_SYNTHASE, True
        if has_abc and not has_wzy:
            return LocusArchetype.CPS_ABC, True
        return (LocusArchetype.CPS_WZY, True) if has_wzy else (default_arch, False)

    if expected_cat == ArchetypeCategory.EXOPOLYSACCHARIDE:
        if locus.archetype.category == ArchetypeCategory.EXOPOLYSACCHARIDE:
            return locus.archetype, True
        if has_wzy:
            return LocusArchetype.CPS_WZY, True
        return (expected_arch or LocusArchetype.EPS_PGA, True) if has_syn else (default_arch, False)

    return (locus.archetype if locus.archetype != LocusArchetype.UNKNOWN else default_arch), True


_RESOLUTION_TIERS: tuple[LocusResolutionTier, ...] = (
    ExplicitAccessionTier(),
    PubMedElinkTier(untagged_fallback=False),
    StrainAssemblyTier(),
    TaxonCatalogTier(),
    PubMedElinkTier(untagged_fallback=True),
)


def _iter_candidate_pools(rec: GlycanRecord, bundle: LociCacheBundle) -> tuple[list[tuple[bytes, Loci, bool]], bytes]:
    """Harvest provenance from `rec.entries` and yield deduplicated `(asm_id, Loci, is_untagged)` pools across Tiers 1–4."""
    ctx = LocusQueryContext.from_record(rec, bundle)
    pools: list[tuple[bytes, Loci, bool]] = []
    seen_keys: set[bytes] = set()
    fallback_asm_id = ctx.acc_candidates[0] if ctx.acc_candidates else b""

    for tier in _RESOLUTION_TIERS:
        for cache_key, is_untagged in tier.iter_cache_keys(ctx):
            entry = bundle.get_entry(cache_key, allow_base_acc=not is_untagged)
            if entry is not None:
                if entry.assembly_id and not fallback_asm_id and not is_untagged:
                    fallback_asm_id = entry.assembly_id
                if len(entry.loci) > 0 and entry.cache_key not in seen_keys:
                    seen_keys.add(entry.cache_key)
                    pools.append((entry.assembly_id, entry.loci, is_untagged))

    return pools, fallback_asm_id


def build_curated_glycan_records(
    raw_records: list[dict[str, Any]],
    assembly_loci_cache: dict[bytes, tuple[bytes, Loci]] | LociCacheBundle,
    strain_to_assembly: dict[tuple[bytes, bytes], bytes] | None = None,
    pmid_to_doi: dict[int, bytes] | None = None,
    canonical_to_gtc: dict[bytes, bytes] | None = None,
) -> GlycanRecords:
    """Consolidate duplicate/fragment records, graft modifiers, pair with `Loci`, and compile `GlycanRecords`."""
    normalize_raw_record_taxonomy(raw_records)
    assigner = BiosyntheticAssigner()
    gtc_lookup = canonical_to_gtc or {}
    bundle = (
        assembly_loci_cache
        if isinstance(assembly_loci_cache, LociCacheBundle)
        else LociCacheBundle.from_dicts(assembly_loci_cache, strain_to_assembly, pmid_to_doi)
    )
    doi_lookup = pmid_to_doi if pmid_to_doi is not None else bundle.pmid_to_doi
    seen_keys: set[bytes] = set()
    initial_records: list[GlycanRecord] = []
    cat_defaults = {
        ArchetypeCategory.CAPSULE: LocusArchetype.CPS_WZY,
        ArchetypeCategory.O_ANTIGEN: LocusArchetype.LPS_WZY,
        ArchetypeCategory.EXOPOLYSACCHARIDE: LocusArchetype.EPS_PGA,
    }

    for r in raw_records:
        if (comp_key := b"%s:%s" % (r["source_db"], r["record_id"])) in seen_keys:
            continue
        seen_keys.add(comp_key)
        gly, ag_type, notes_b = r["glycan"], r["antigen_type"], r["notes"]
        expected_cat, expected_arch = assigner.infer_expected_archetype(b"%s %s" % (ag_type, notes_b))
        default_arch = expected_arch or cat_defaults.get(expected_cat, LocusArchetype.UNKNOWN)  # type: ignore[arg-type]
        pmid, acc_hint_b = int(r.get("pmid", 0)), r.get("accession_hint", b"")
        rec_obj = GlycanRecord(
            source_db=r["source_db"],
            record_id=r["record_id"],
            organism=r["organism"],
            strain=r["strain"],
            serotype=r["serotype"],
            antigen_type=ag_type,
            assembly_id=acc_hint_b,
            archetype=default_arch,
            structure_raw=r["structure_raw"],
            structure_linear=r["structure_linear"],
            glycan=gly,
            is_repeating=bool(r["is_repeating"]),
            pmid=pmid,
            publication_year=int(r.get("publication_year", 0)),
            doi=r.get("doi", b"") or doi_lookup.get(pmid, b""),
            glytoucan_id=r.get("glytoucan_id", b"") or gtc_lookup.get(gly.canonical, b""),
            notes=notes_b,
        )
        initial_records.append(
            replace(rec_obj, entries=(rec_obj.to_entry(EntryRelation.CANONICAL, accession_hint=acc_hint_b),))
        )

    paired_records: list[GlycanRecord] = []
    for rec in consolidate_curated_records(initial_records):
        assert rec.glycan is not None
        hint_text = b"%s %s" % (rec.antigen_type, rec.notes)
        expected_cat, expected_arch = assigner.infer_expected_archetype(hint_text)
        default_arch = expected_arch or (
            rec.archetype if rec.archetype != LocusArchetype.UNKNOWN else LocusArchetype.LPS_WZY
        )

        candidate_pools, fallback_asm_id = _iter_candidate_pools(rec, bundle)
        matched_locus: Locus | None = None
        matched_arch: LocusArchetype = default_arch
        align_score, resolved_asm_id = 0.0, fallback_asm_id

        for pool_asm_id, pool_loci, is_untagged_pool in candidate_pools:
            compatible_loci: list[Locus] = []
            for j in range(len(pool_loci)):
                cand_loc = pool_loci[j]
                res_arch, is_compat = resolve_top_down_locus_archetype(
                    cand_loc, rec.antigen_type, expected_cat, expected_arch, default_arch
                )
                if is_compat:
                    compatible_loci.append(
                        replace(cand_loc, archetype=res_arch) if cand_loc.archetype != res_arch else cand_loc
                    )
            if not compatible_loci:
                continue

            match_res = assigner.match_locus(
                rec.glycan,
                compatible_loci,
                antigen_hint=hint_text,
                expected_category=expected_cat,
                expected_archetype=expected_arch or default_arch,
            )
            if match_res is not None and (not is_untagged_pool or match_res.bio_score >= 0.30):
                matched_locus = match_res.locus
                loc_name = matched_locus.name
                resolved_asm_id = (
                    (loc_name.encode("utf-8")[:32] if isinstance(loc_name, str) else bytes(loc_name)[:32])
                    if loc_name
                    else pool_asm_id
                )
                matched_arch = match_res.archetype if match_res.archetype != LocusArchetype.UNKNOWN else default_arch
                if matched_locus.archetype != matched_arch:
                    matched_locus = replace(matched_locus, archetype=matched_arch)
                align_score = float(match_res.combined_score)
                break

        asm_out = resolved_asm_id if matched_locus is not None or fallback_asm_id else b""
        paired_records.append(
            replace(rec, assembly_id=asm_out, archetype=matched_arch, locus=matched_locus, alignment_score=align_score)
        )

    return GlycanRecords.build(paired_records)
