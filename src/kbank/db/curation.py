"""Taxonomy normalization, topological glycan consolidation, and 4-tier SeroGen locus pairing."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import replace
import gzip
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from serogen.core.genome import GenomeAssembly
from serogen.loci.archetypes import ArchetypeCategory, LocusArchetype
from serogen.loci.models import Loci, Locus
from serogen.loci.predictor import LocusPredictor

from kbank.biosynthesis.align import BiosyntheticAssigner
from kbank.db.models import GlycanRecord
from kbank.db.records import GlycanRecords, _loci_from_npz_dict, _loci_to_npz_dict
from kbank.glycan import Glycan
from kbank.io.sources import DEFAULT_CACHE_DIR, SALMONELLA_SEROVAR_TO_OGROUP

NCBI_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_LOCI_CACHE_PATH = DEFAULT_CACHE_DIR / "predicted_loci.npz"

# Reverse lookup from canonical Salmonella O-serogroup (e.g. 'O4') to common Kauffmann-White serovars
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


def _cassette_ctx(recorder: Any | None, cassette_name: str) -> Any:
    """Return `recorder.use_cassette(cassette_name)` if `recorder` is provided, else `nullcontext()`."""
    if recorder is not None and hasattr(recorder, "use_cassette"):
        return recorder.use_cassette(cassette_name)
    return nullcontext()


def antigen_family(antigen_type: bytes) -> bytes:
    """Normalize `antigen_type` labels into canonical family buckets (`CPS`, `O-antigen`, `WTA`, `EPS`)."""
    if antigen_type.startswith(b"CPS"):
        return b"CPS"
    if antigen_type in (b"O-antigen", b"LPS", b"LOS"):
        return b"O-antigen"
    if antigen_type == b"WTA":
        return b"WTA"
    if antigen_type == b"EPS":
        return b"EPS"
    return antigen_type


def normalize_raw_record_taxonomy(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unify `<Genus> sp.` organism labels with specific species when serotype or strain matches."""
    genus_canonical_species: dict[bytes, bytes] = {
        b"Klebsiella": b"Klebsiella pneumoniae",
        b"Salmonella": b"Salmonella enterica",
        b"Escherichia": b"Escherichia coli",
        b"Shigella": b"Shigella flexneri",
        b"Streptococcus": b"Streptococcus pneumoniae",
        b"Acinetobacter": b"Acinetobacter baumannii",
        b"Pseudomonas": b"Pseudomonas aeruginosa",
        b"Proteus": b"Proteus mirabilis",
        b"Yersinia": b"Yersinia pseudotuberculosis",
        b"Vibrio": b"Vibrio cholerae",
        b"Campylobacter": b"Campylobacter jejuni",
        b"Helicobacter": b"Helicobacter pylori",
        b"Haemophilus": b"Haemophilus influenzae",
        b"Neisseria": b"Neisseria meningitidis",
        b"Staphylococcus": b"Staphylococcus aureus",
        b"Listeria": b"Listeria monocytogenes",
        b"Enterococcus": b"Enterococcus faecalis",
        b"Cronobacter": b"Cronobacter sakazakii",
        b"Serratia": b"Serratia marcescens",
        b"Citrobacter": b"Citrobacter freundii",
        b"Aeromonas": b"Aeromonas hydrophila",
        b"Providencia": b"Providencia alcalifaciens",
    }
    genus_sero_to_species: dict[tuple[bytes, bytes], bytes] = {}
    genus_strain_to_species: dict[tuple[bytes, bytes], bytes] = {}

    for r in raw_records:
        org: bytes = r["organism"]
        if not org.endswith(b" sp.") and b" " in org:
            genus = org.split(b" ", 1)[0]
            sero: bytes = r.get("serotype", b"")
            strain: bytes = r.get("strain", b"")
            if sero:
                genus_sero_to_species.setdefault((genus, sero), org)
            if strain:
                genus_strain_to_species.setdefault((genus, strain), org)

    for r in raw_records:
        org = r["organism"]
        if org.endswith(b" sp."):
            genus = org.split(b" ", 1)[0]
            sero = r.get("serotype", b"")
            strain = r.get("strain", b"")
            if sero and (genus, sero) in genus_sero_to_species:
                r["organism"] = genus_sero_to_species[(genus, sero)]
            elif strain and (genus, strain) in genus_strain_to_species:
                r["organism"] = genus_strain_to_species[(genus, strain)]
            elif sero and genus in genus_canonical_species:
                r["organism"] = genus_canonical_species[genus]

        # Ensure Salmonella records with serovar strain names also populate canonical O-group serotype
        if r["organism"] == b"Salmonella enterica" and not r.get("serotype") and r.get("strain"):
            st_low = r["strain"].decode("utf-8", errors="replace").strip().lower()
            if st_low in SALMONELLA_SEROVAR_TO_OGROUP:
                r["serotype"] = SALMONELLA_SEROVAR_TO_OGROUP[st_low].encode("utf-8")

    return raw_records


def record_rank_key(rec: GlycanRecord, is_oligomer_duplicate: bool = False) -> tuple[Any, ...]:
    """Rank candidate records so intact 1x repeating units with loci and modifiers win."""
    db_priority: dict[bytes, int] = {
        b"ECODAB": 5,
        b"EK3D": 5,
        b"EPSDB": 4,
        b"CSDB": 3,
        b"KEGG": 1,
    }
    gly = rec.glycan
    intact_rep = int(rec.is_repeating and not gly.is_alditol and not is_oligomer_duplicate)
    backbone_len = len(gly.without_modifiers().nodes)
    return (
        intact_rep,
        int(not is_oligomer_duplicate),
        int(rec.is_repeating),
        int(not gly.is_alditol),
        int(rec.locus is not None),
        round(float(rec.alignment_score), 3),
        backbone_len if intact_rep else -backbone_len,
        gly.num_modifiers,
        db_priority.get(rec.source_db, 0),
        int(rec.pmid > 0),
        int(rec.publication_year),
    )


def merge_record_cluster(cluster: list[GlycanRecord]) -> GlycanRecord:
    """Merge a cluster of duplicate/fragment records into the single highest-ranked canonical `GlycanRecord`."""
    if len(cluster) == 1:
        return cluster[0]

    intact_backbones = [r for r in cluster if r.is_repeating and not r.glycan.is_alditol]
    oligomer_flags: dict[int, bool] = {}
    for idx_a, rec_a in enumerate(cluster):
        is_olig = False
        for rec_b in intact_backbones:
            if rec_a is not rec_b and len(rec_a.glycan.without_modifiers().nodes) > len(
                rec_b.glycan.without_modifiers().nodes
            ):
                if rec_a.glycan.is_oligomer_of(rec_b.glycan):
                    is_olig = True
                    break
        oligomer_flags[idx_a] = is_olig

    ranked_indices = sorted(
        range(len(cluster)),
        key=lambda i: record_rank_key(cluster[i], is_oligomer_duplicate=oligomer_flags[i]),
        reverse=True,
    )
    winner = cluster[ranked_indices[0]]
    others = [cluster[i] for i in ranked_indices[1:]]

    # Graft modifiers (OAc, Pyr, OMe, Fo, P, PEtn, GroP, DAla, Lac) from fragments/variants onto winner
    winner_gly = winner.glycan
    all_grafted: list[bytes] = []
    for other in others:
        if other.glycan.num_modifiers > 0:
            winner_gly, grafted = winner_gly.graft_modifiers_from(other.glycan)
            for g_name in grafted:
                if g_name not in all_grafted:
                    all_grafted.append(g_name)

    strain = winner.strain
    serotype = winner.serotype
    antigen_type = winner.antigen_type
    assembly_id = winner.assembly_id
    archetype = winner.archetype
    locus = winner.locus
    alignment_score = winner.alignment_score
    pmid = winner.pmid
    publication_year = winner.publication_year
    doi = winner.doi
    glytoucan_id = winner.glytoucan_id

    for other in others:
        if not strain and other.strain:
            strain = other.strain
        if not serotype and other.serotype:
            serotype = other.serotype
        if antigen_type in (b"CPS", b"O-antigen") and other.antigen_type.startswith(b"CPS (Group"):
            antigen_type = other.antigen_type
        if locus is None and other.locus is not None:
            locus = other.locus
            alignment_score = other.alignment_score
            if other.assembly_id:
                assembly_id = other.assembly_id
            if other.archetype != LocusArchetype.UNKNOWN:
                archetype = other.archetype
        elif not assembly_id and other.assembly_id:
            assembly_id = other.assembly_id
        if archetype == LocusArchetype.UNKNOWN and other.archetype != LocusArchetype.UNKNOWN:
            archetype = other.archetype
        if pmid <= 0 and other.pmid > 0:
            pmid = other.pmid
        if publication_year <= 0 and other.publication_year > 0:
            publication_year = other.publication_year
        if not doi and other.doi:
            doi = other.doi
        if not glytoucan_id and other.glytoucan_id:
            glytoucan_id = other.glytoucan_id

    notes = winner.notes
    merged_keys = [
        b"%s:%s" % (o.source_db, o.record_id)
        for o in others[:6]
        if (o.source_db, o.record_id) != (winner.source_db, winner.record_id)
    ]
    if merged_keys:
        merged_tag = b" [merged_ids=" + b",".join(merged_keys) + b"]"
        if b"[merged_ids=" not in notes:
            notes = (notes + merged_tag)[:256]
    if all_grafted:
        graft_tag = b" [grafted_modifiers=+" + b",+".join(all_grafted) + b"]"
        if b"[grafted_modifiers=" not in notes:
            notes = (notes + graft_tag)[:256]

    return GlycanRecord(
        source_db=winner.source_db,
        record_id=winner.record_id,
        organism=winner.organism,
        strain=strain,
        serotype=serotype,
        antigen_type=antigen_type,
        assembly_id=assembly_id,
        archetype=archetype,
        locus=locus,
        alignment_score=alignment_score,
        structure_raw=winner.structure_raw,
        structure_linear=winner.structure_linear,
        glycan=winner_gly,
        is_repeating=winner.is_repeating,
        pmid=pmid,
        publication_year=publication_year,
        doi=doi,
        glytoucan_id=glytoucan_id,
        notes=notes,
    )


def consolidate_curated_records(records: list[GlycanRecord]) -> list[GlycanRecord]:
    """Consolidate duplicate, cyclic-shift, oligomer, and fragment `GlycanRecord` entries."""
    typed_buckets: dict[tuple[bytes, bytes, bytes], list[GlycanRecord]] = defaultdict(list)
    untyped_records: list[GlycanRecord] = []

    for rec in records:
        if rec.serotype:
            fam = antigen_family(rec.antigen_type)
            typed_buckets[(rec.organism, rec.serotype, fam)].append(rec)
        else:
            untyped_records.append(rec)

    typed_clusters: dict[tuple[bytes, bytes, bytes], list[GlycanRecord]] = {
        k: list(v) for k, v in typed_buckets.items()
    }
    typed_reps: dict[tuple[bytes, bytes, bytes], GlycanRecord] = {
        k: merge_record_cluster(v) for k, v in typed_clusters.items()
    }

    typed_by_org: dict[bytes, list[tuple[bytes, bytes, bytes]]] = defaultdict(list)
    for k in typed_reps:
        typed_by_org[k[0]].append(k)

    remaining_untyped: list[GlycanRecord] = []
    for u_rec in untyped_records:
        absorbed = False
        u_fam = antigen_family(u_rec.antigen_type)
        for k in typed_by_org.get(u_rec.organism, []):
            rep = typed_reps[k]
            same_strain = bool(u_rec.strain and rep.strain and u_rec.strain == rep.strain)
            same_pmid = bool(u_rec.pmid > 0 and rep.pmid > 0 and u_rec.pmid == rep.pmid)
            same_fam = k[2] == u_fam

            if same_fam and u_rec.glycan.is_isomorphic(rep.glycan, allow_wildcards=True):
                typed_clusters[k].append(u_rec)
                typed_reps[k] = merge_record_cluster(typed_clusters[k])
                absorbed = True
                break
            if (same_strain or same_pmid) and (
                u_rec.glycan.is_isomorphic(rep.glycan, allow_wildcards=True)
                or u_rec.glycan.is_subgraph_of(rep.glycan)
                or (not u_rec.is_repeating and u_rec.glycan.is_oligomer_of(rep.glycan))
            ):
                typed_clusters[k].append(u_rec)
                typed_reps[k] = merge_record_cluster(typed_clusters[k])
                absorbed = True
                break
        if not absorbed:
            remaining_untyped.append(u_rec)

    untyped_by_org_fam: dict[tuple[bytes, bytes], list[GlycanRecord]] = defaultdict(list)
    for u_rec in remaining_untyped:
        untyped_by_org_fam[(u_rec.organism, antigen_family(u_rec.antigen_type))].append(u_rec)

    consolidated_untyped: list[GlycanRecord] = []
    for _, grp in untyped_by_org_fam.items():
        grp_sorted = sorted(
            grp,
            key=lambda r: (
                int(r.is_repeating and not r.glycan.is_alditol),
                len(r.glycan.without_modifiers().nodes),
                r.glycan.num_modifiers,
                record_rank_key(r),
            ),
            reverse=True,
        )
        clusters: list[list[GlycanRecord]] = []
        for cand in grp_sorted:
            placed = False
            for cl in clusters:
                lead = cl[0]
                same_context = bool(
                    (cand.strain and lead.strain and cand.strain == lead.strain)
                    or (cand.pmid > 0 and lead.pmid > 0 and cand.pmid == lead.pmid)
                )
                if cand.glycan.is_isomorphic(lead.glycan, allow_wildcards=True):
                    cl.append(cand)
                    placed = True
                    break
                if same_context and (
                    cand.glycan.is_subgraph_of(lead.glycan)
                    or (not cand.is_repeating and cand.glycan.is_oligomer_of(lead.glycan))
                    or (lead.is_repeating and not cand.is_repeating and lead.glycan.is_oligomer_of(cand.glycan))
                ):
                    cl.append(cand)
                    placed = True
                    break
            if not placed:
                clusters.append([cand])

        for cl in clusters:
            consolidated_untyped.append(merge_record_cluster(cl))

    return list(typed_reps.values()) + consolidated_untyped


def load_predicted_loci_cache(path: Path = DEFAULT_LOCI_CACHE_PATH) -> dict[bytes, tuple[bytes, Loci]]:
    """Load zero-pickle cached `(asm_id, Loci)` entries from `.cache/predicted_loci.npz`."""
    out, _, _ = _load_predicted_loci_cache_bundle(path)
    return out


def _load_predicted_loci_cache_bundle(
    path: Path = DEFAULT_LOCI_CACHE_PATH,
) -> tuple[dict[bytes, tuple[bytes, Loci]], dict[tuple[bytes, bytes], bytes], dict[int, bytes]]:
    """Load `(assembly_loci_cache, strain_to_assembly, pmid_to_doi)` from `.cache/predicted_loci.npz`."""
    if not path.is_file():
        return {}, {}, {}
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = data["cache_keys"]
            asm_ids = data["asm_ids"]
            offsets = data["key_offsets"]
            flat_loci = _loci_from_npz_dict(data, prefix="loci_")
            strain_orgs = data["strain_orgs"] if "strain_orgs" in data else None
            strain_toks = data["strain_toks"] if "strain_toks" in data else None
            strain_keys = data["strain_keys"] if "strain_keys" in data else None
            doi_pmids = data["doi_pmids"] if "doi_pmids" in data else None
            doi_vals = data["doi_vals"] if "doi_vals" in data else None
        out: dict[bytes, tuple[bytes, Loci]] = {}
        empty_loci = Loci.empty()
        for i in range(len(keys)):
            k_b = bytes(keys[i]).rstrip(b"\x00")
            a_b = bytes(asm_ids[i]).rstrip(b"\x00")
            s_idx = int(offsets[i])
            e_idx = int(offsets[i + 1])
            if e_idx > s_idx:
                out[k_b] = (a_b, flat_loci[s_idx:e_idx])
            else:
                out[k_b] = (a_b, empty_loci)
        strain_map: dict[tuple[bytes, bytes], bytes] = {}
        for k_b in out:
            if k_b.startswith(b"nuccore:"):
                parts = k_b.split(b":", 2)
                if len(parts) == 3 and parts[1] and parts[2]:
                    strain_map[(parts[1], parts[2])] = k_b
        if strain_orgs is not None and strain_toks is not None and strain_keys is not None:
            for i in range(len(strain_orgs)):
                o_b = bytes(strain_orgs[i]).rstrip(b"\x00")
                t_b = bytes(strain_toks[i]).rstrip(b"\x00")
                ck_b = bytes(strain_keys[i]).rstrip(b"\x00")
                if o_b and t_b and ck_b:
                    strain_map[(o_b, t_b)] = ck_b
        doi_map: dict[int, bytes] = {}
        if doi_pmids is not None and doi_vals is not None:
            for i in range(len(doi_pmids)):
                p_val = int(doi_pmids[i])
                d_b = bytes(doi_vals[i]).rstrip(b"\x00")
                if p_val > 0 and d_b:
                    doi_map[p_val] = d_b
        return out, strain_map, doi_map
    except Exception:
        return {}, {}, {}


def save_predicted_loci_cache(
    path: Path,
    cache: dict[bytes, tuple[bytes, Loci]],
    *,
    strain_to_assembly: dict[tuple[bytes, bytes], bytes] | None = None,
    pmid_to_doi: dict[int, bytes] | None = None,
) -> None:
    """Persist `(asm_id, Loci)` entries (plus optional strain/DOI maps) to `.cache/predicted_loci.npz` without pickle."""
    persist_items = list(cache.items())
    if not persist_items:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(persist_items)
        keys_arr = np.zeros(n, dtype="S96")
        asm_arr = np.zeros(n, dtype="S32")
        offsets = np.zeros(n + 1, dtype=np.int32)
        all_loci_list: list[Locus] = []
        curr = 0
        for i, (k_b, (asm_b, loci_obj)) in enumerate(persist_items):
            keys_arr[i] = k_b[:96]
            asm_arr[i] = asm_b[:32]
            for j in range(len(loci_obj)):
                all_loci_list.append(loci_obj[j])
            curr += len(loci_obj)
            offsets[i + 1] = curr
        flat_loci = Loci.from_list(all_loci_list) if all_loci_list else Loci.empty()
        extra_arrays: dict[str, np.ndarray] = {}
        if strain_to_assembly:
            s_items = sorted(strain_to_assembly.items())
            extra_arrays["strain_orgs"] = np.array([o[:64] for (o, _), _ in s_items], dtype="S64")
            extra_arrays["strain_toks"] = np.array([t[:64] for (_, t), _ in s_items], dtype="S64")
            extra_arrays["strain_keys"] = np.array([ck[:96] for _, ck in s_items], dtype="S96")
        if pmid_to_doi:
            d_items = sorted(pmid_to_doi.items())
            extra_arrays["doi_pmids"] = np.array([p for p, _ in d_items], dtype=np.int64)
            extra_arrays["doi_vals"] = np.array([d[:96] for _, d in d_items], dtype="S96")
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp_path,
            cache_keys=keys_arr,
            asm_ids=asm_arr,
            key_offsets=offsets,
            **extra_arrays,
            **_loci_to_npz_dict(flat_loci, prefix="loci_"),
        )
        os.replace(tmp_path, path)
    except Exception:
        pass


async def resolve_and_predict_loci(
    raw_records: list[dict[str, Any]],
    max_assemblies: int = 100,
    *,
    recorder: Any | None = None,
    loci_cache_path: Path = DEFAULT_LOCI_CACHE_PATH,
) -> tuple[dict[bytes, tuple[bytes, Loci]], dict[tuple[bytes, bytes], bytes], dict[int, bytes]]:
    """Fetch in-memory `GenomeAssembly` FASTAs from NCBI and predict candidate `Loci` via `LocusPredictor`."""
    import httpx

    t_stage0 = time.perf_counter()
    batch_predictor = LocusPredictor(min_completeness=0.65, stitch_contig_splits=False)
    whole_genome_predictor = LocusPredictor(min_completeness=0.75)
    word_split_re = re.compile(r"[A-Za-z0-9]+(?::[A-Za-z0-9]+)?")
    num_sero_re = re.compile(
        r"(?:serotype|serogroup|type|capsule|b-band|osa|iats)\s+O?([0-9]{1,3}[A-Za-z]?)\b",
        re.IGNORECASE,
    )

    def _clean_fasta_acc(raw_header: bytes) -> str:
        first_tok = raw_header.split()[0].decode("utf-8", errors="replace").lstrip(">")
        if "|" in first_tok:
            parts_pipe = [
                p for p in first_tok.split("|") if p and not p.isdigit() and p not in ("gb", "emb", "dbj", "ref", "gi")
            ]
            if parts_pipe:
                first_tok = parts_pipe[-1]
        return first_tok[:32]

    def _predict_multi_fasta_batch(fasta_blob: bytes) -> dict[str, tuple[bytes, list[Locus]]]:
        grouped: dict[str, tuple[bytes, list[Locus]]] = {}
        stripped = fasta_blob.lstrip()
        if not stripped.startswith(b">"):
            return grouped
        try:
            raw_entries = stripped.split(b"\n>")
            bgc_entries: list[bytes] = []
            for idx_e, chunk in enumerate(raw_entries):
                entry = chunk if idx_e == 0 else b">" + chunk
                nl_pos = entry.find(b"\n")
                if nl_pos <= 1:
                    continue
                acc_tok = _clean_fasta_acc(entry[1:nl_pos])
                if not acc_tok:
                    continue
                acc_b = acc_tok.encode("utf-8")[:32]
                base_tok = acc_tok.split(".")[0]
                grouped.setdefault(base_tok, (acc_b, []))
                seq_bytes_len = len(entry) - nl_pos
                if 1_500 <= seq_bytes_len <= 95_000:
                    bgc_entries.append(entry)

            if not bgc_entries:
                return grouped

            filtered_blob = b"\n".join(bgc_entries)
            asm = GenomeAssembly.from_fasta(filtered_blob, id_=b"batch", truncate_id=True)
            if len(asm) == 0:
                return grouped
            pred_loci = batch_predictor.predict(asm)
            if len(pred_loci) == 0:
                return grouped
            by_cidx: dict[int, list[Locus]] = defaultdict(list)
            for j in range(len(pred_loci)):
                loc = pred_loci[j].canonicalize_strand()
                by_cidx[int(loc.contig_index)].append(loc)
            for c_idx, loc_list in by_cidx.items():
                if 0 <= c_idx < len(asm.ids):
                    raw_id = bytes(asm.ids[c_idx]).strip(b"\x00").strip()
                    acc_tok = _clean_fasta_acc(raw_id)
                    acc_b = acc_tok.encode("utf-8")[:32]
                    base_tok = acc_tok.split(".")[0]
                    tagged = [replace(loc.with_name(acc_tok), contig_index=0) for loc in loc_list]
                    grouped[base_tok] = (acc_b, tagged)
        except Exception:
            pass
        return grouped

    def _is_true_strain_isolate(org_b: bytes, str_b: bytes, sero_b: bytes) -> bool:
        if (
            not str_b
            or len(str_b) < 2
            or b"sp." in org_b
            or org_b in (b"Unknown bacterium", b"Bacteria (KEGG pathway)")
        ):
            return False
        if str_b == sero_b:
            return False
        str_s = str_b.decode("utf-8", errors="replace").strip()
        if re.fullmatch(
            r"(?:type\s+|serotype\s+|serogroup\s+)?(?:K|KL|O|OL|OC|OCL|VI|\d{1,3}[A-Za-z]?|[IVX]{1,4}[abAB]?)",
            str_s,
            re.IGNORECASE,
        ):
            return False
        return True

    def _catalog_taxon(org_b: bytes) -> str:
        if org_b in (b"Unknown bacterium", b"Bacteria (KEGG pathway)"):
            return ""
        org_s = org_b.decode("utf-8", errors="replace").strip()
        if org_s.endswith(" sp."):
            return org_s[:-4].strip()
        parts = org_s.split()
        if not parts:
            return ""
        genus = parts[0]
        if genus in (
            "Klebsiella",
            "Acinetobacter",
            "Salmonella",
            "Shigella",
            "Proteus",
            "Yersinia",
            "Campylobacter",
            "Cronobacter",
            "Burkholderia",
            "Providencia",
            "Raoultella",
            "Serratia",
            "Citrobacter",
            "Enterobacter",
            "Morganella",
            "Hafnia",
            "Edwardsiella",
            "Vibrio",
            "Aeromonas",
            "Shewanella",
            "Pseudoalteromonas",
            "Rhodococcus",
            "Rhizobium",
            "Sinorhizobium",
            "Bradyrhizobium",
            "Agrobacterium",
            "Azospirillum",
            "Ochrobactrum",
            "Xanthomonas",
            "Pantoea",
            "Erwinia",
            "Pectobacterium",
            "Dickeya",
            "Bacteroides",
            "Clostridium",
            "Bacillus",
            "Lactobacillus",
            "Bifidobacterium",
        ):
            return genus
        return " ".join(parts[:2])

    def _build_token_index(
        targets: set[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[tuple[str, re.Pattern[str]]]]]:
        num_map: dict[str, list[str]] = defaultdict(list)
        word_map: dict[str, list[str]] = defaultdict(list)
        multi_map: dict[str, list[tuple[str, re.Pattern[str]]]] = defaultdict(list)

        for tok_s in targets:
            if not tok_s:
                continue
            if tok_s.isdigit():
                num_map[tok_s.upper()].append(tok_s)
                continue
            words = word_split_re.findall(tok_s)
            if not words:
                continue
            if len(words) == 1 and words[0] == tok_s:
                u_tok = tok_s.upper()
                word_map[u_tok].append(tok_s)
                if u_tok.startswith("K") and u_tok[1:].isdigit():
                    word_map["KL" + u_tok[1:]].append(tok_s)
                elif u_tok.startswith("KL") and u_tok[2:].isdigit():
                    word_map["K" + u_tok[2:]].append(tok_s)
                elif u_tok.startswith("O:") and u_tok[2:].isdigit():
                    word_map["O" + u_tok[2:]].append(tok_s)
                    num_map[u_tok[2:]].append(tok_s)
                elif u_tok.startswith("O") and u_tok[1:].isdigit():
                    word_map["O:" + u_tok[1:]].append(tok_s)
                    # Also index bare number for 'serotype 6' / 'B-band O6' / 'serogroup 11' (e.g. P. aeruginosa OSA)
                    num_map[u_tok[1:]].append(tok_s)
                # Map Salmonella O-serogroups (e.g. O4, O9) to Kauffmann-White serovar words in NCBI titles
                if u_tok in OGROUP_TO_SALMONELLA_SEROVARS:
                    for sv_alias in OGROUP_TO_SALMONELLA_SEROVARS[u_tok]:
                        sv_words = word_split_re.findall(sv_alias)
                        if len(sv_words) == 1:
                            word_map[sv_words[0].upper()].append(tok_s)
            else:
                anchor = max(words, key=len).upper()
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(tok_s)}(?![A-Za-z0-9])", re.IGNORECASE)
                multi_map[anchor].append((tok_s, pat))
        return num_map, word_map, multi_map

    def _find_matching_tokens(
        title: str,
        blob_txt: str,
        sub_pairs: list[tuple[str, str]],
        num_map: dict[str, list[str]],
        word_map: dict[str, list[str]],
        multi_map: dict[str, list[tuple[str, re.Pattern[str]]]],
    ) -> set[str]:
        matched: set[str] = set()
        words_upper = {w.upper() for w in word_split_re.findall(blob_txt)}
        for w in words_upper:
            if w in word_map:
                matched.update(word_map[w])
            if w in multi_map:
                for orig_tok, pat in multi_map[w]:
                    if orig_tok not in matched and pat.search(blob_txt):
                        matched.add(orig_tok)
        if num_map:
            for m_num in num_sero_re.findall(title):
                u_num = m_num.upper()
                if u_num in num_map:
                    matched.update(num_map[u_num])
            for st_k, sn_v in sub_pairs:
                if st_k.strip().lower() in ("serotype", "serogroup", "serovar"):
                    u_sn = sn_v.strip().upper()
                    if u_sn in num_map:
                        matched.update(num_map[u_sn])
                    elif u_sn.startswith("O") and u_sn[1:] in num_map:
                        matched.update(num_map[u_sn[1:]])
        return matched

    assembly_loci_cache, strain_to_assembly, pmid_to_doi = _load_predicted_loci_cache_bundle(loci_cache_path)
    if (
        assembly_loci_cache
        and strain_to_assembly
        and pmid_to_doi
        and os.environ.get("KBANK_FORCE_LOCI_REFRESH", "0") != "1"
    ):
        print(
            f"[KBank] Loaded warm predicted_loci bundle ({len(assembly_loci_cache)} keys, {len(strain_to_assembly)} strain mappings, {len(pmid_to_doi)} DOIs) in {time.perf_counter() - t_stage0:.2f}s",
            flush=True,
        )
        return assembly_loci_cache, strain_to_assembly, pmid_to_doi

    base_acc_in_cache: dict[str, bytes] = {
        k.decode("utf-8", errors="replace").split(".")[0]: k
        for k in assembly_loci_cache
        if not k.startswith((b"nuccore:", b"pmid:", b"uid:"))
    }
    empty_loci_singleton = Loci.empty()

    explicit_nuccore: set[str] = set()
    for r in raw_records:
        acc_b = r.get("accession_hint", b"")
        if acc_b:
            acc_s = acc_b.decode("utf-8", errors="replace").strip()
            if not acc_s.startswith(("GCF_", "GCA_", "NC_", "NZ_")) and len(acc_s) >= 6:
                explicit_nuccore.add(acc_s)

    strain_counts: Counter[tuple[bytes, bytes]] = Counter()
    for r in raw_records:
        org_b = r["organism"]
        str_b = r["strain"]
        sero_b = r["serotype"]
        if _is_true_strain_isolate(org_b, str_b, sero_b):
            strain_counts[(org_b, str_b)] += 1

    top_strain_queries = [
        pair for pair, _ in sorted(strain_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_assemblies]
    ]

    api_key = os.environ.get("NCBI_API_KEY", "")
    base_params: dict[str, str] = {"retmode": "json"}
    if api_key:
        base_params["api_key"] = api_key

    sem = asyncio.Semaphore(5 if api_key else 3)

    async def _polite_sleep(t_req: float, delay: float = 0.22) -> None:
        elapsed = time.perf_counter() - t_req
        if recorder is not None and elapsed < 0.25:
            return
        if elapsed > 0.05:
            await asyncio.sleep(delay)

    with _cassette_ctx(recorder, "ncbi_eutils.toml"):
        async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
            uncached_tier1 = [
                acc for acc in sorted(explicit_nuccore)[:300] if acc.split(".")[0] not in base_acc_in_cache
            ]
            tier1_uids: set[str] = set()
            for i in range(0, len(uncached_tier1), 35):
                batch_accs = uncached_tier1[i : i + 35]
                acc_clause = " OR ".join(f"{acc}[ACCN]" for acc in batch_accs)
                term_t1 = f"({acc_clause}) AND 2000:85000[SLEN]"
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r_t1 = await client.get(
                            f"{NCBI_EUTILS_URL}/esearch.fcgi",
                            params={**base_params, "db": "nuccore", "term": term_t1, "retmax": "100"},
                        )
                        if r_t1.status_code == 200:
                            for u in r_t1.json().get("esearchresult", {}).get("idlist", []):
                                tier1_uids.add(str(u))
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

            print(
                f"[KBank] Tier 1 explicit accessions resolved ({len(tier1_uids)} nuccore UIDs, {len(base_acc_in_cache)} pre-cached, {time.perf_counter() - t_stage0:.1f}s)",
                flush=True,
            )

            pmid_weights: Counter[int] = Counter()
            for r in raw_records:
                p_val = int(r.get("pmid", 0))
                if p_val <= 0:
                    continue
                acc_b = r.get("accession_hint", b"")
                if acc_b and acc_b.decode("utf-8", errors="replace").split(".")[0] in base_acc_in_cache:
                    continue
                pmid_weights[p_val] += 1

            top_pmids = [p for p, _ in pmid_weights.most_common(600)]
            pmid_to_nuccore_uids: dict[int, list[str]] = {}

            async def _fetch_elink_batch(batch_pmids: list[int]) -> None:
                form_data: dict[str, Any] = {
                    **base_params,
                    "dbfrom": "pubmed",
                    "db": "nuccore",
                    "linkname": "pubmed_nuccore",
                    "id": [str(p) for p in batch_pmids],
                }
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r_el = await client.post(f"{NCBI_EUTILS_URL}/elink.fcgi", data=form_data)
                        if r_el.status_code == 200:
                            for ls in r_el.json().get("linksets", []):
                                ids_l = ls.get("ids", [])
                                if not ids_l:
                                    continue
                                p_val = int(ids_l[0])
                                for db_l in ls.get("linksetdbs", []):
                                    if db_l.get("linkname") == "pubmed_nuccore":
                                        links = [str(u) for u in db_l.get("links", [])]
                                        if 1 <= len(links) <= 45:
                                            pmid_to_nuccore_uids[p_val] = links[:20]
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

            elink_batches = [top_pmids[i : i + 60] for i in range(0, len(top_pmids), 60)]
            await asyncio.gather(*[_fetch_elink_batch(b) for b in elink_batches])

            elink_uids = sorted(tier1_uids | {u for u_list in pmid_to_nuccore_uids.values() for u in u_list})
            uid_to_meta: dict[str, tuple[str, str, str, str, bool]] = {}
            uid_to_accver: dict[str, str] = {}

            async def _fetch_elink_summary(batch_u: list[str]) -> None:
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r_sum = await client.post(
                            f"{NCBI_EUTILS_URL}/esummary.fcgi",
                            data={**base_params, "db": "nuccore", "id": ",".join(batch_u)},
                        )
                        if r_sum.status_code == 200:
                            sum_res = r_sum.json().get("result", {})
                            for u in batch_u:
                                info = sum_res.get(u, {})
                                if not isinstance(info, dict):
                                    continue
                                slen = int(info.get("slen") or 0)
                                if not (2500 <= slen <= 65000):
                                    continue
                                title = info.get("title", "")
                                title_low = title.lower()
                                if any(
                                    bad in title_low
                                    for bad in ("16s", "23s", "rrna", "gyrb", "rpob", "reca", "phage", "integron")
                                ):
                                    continue
                                acc_ver = (info.get("accessionversion") or info.get("caption") or "").strip()
                                if not acc_ver:
                                    continue
                                subtype = info.get("subtype", "")
                                subname = info.get("subname", "")
                                is_cps_t = any(w in title_low for w in ("capsul", "cps", "kps", "k-antigen", "k antigen"))
                                uid_to_meta[u] = (acc_ver, title, subname, subtype, is_cps_t)
                                uid_to_accver[u] = acc_ver
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

            esum_batches = [elink_uids[i : i + 200] for i in range(0, len(elink_uids), 200)]
            await asyncio.gather(*[_fetch_elink_summary(b) for b in esum_batches])
            print(
                f"[KBank] Tier 1.5 PubMed elink complete ({len(pmid_to_nuccore_uids)} PMIDs -> {len(uid_to_meta)} BGC UIDs, {time.perf_counter() - t_stage0:.1f}s)",
                flush=True,
            )

            uid_to_strain: dict[str, tuple[bytes, bytes]] = {}
            unresolved_collection_strains: list[tuple[bytes, bytes]] = []
            collection_re = re.compile(r"\b(PCM|CCM|IMET|IAM|CIP|JCM|DSM|LMG|CECT|NCIMB|VKM)\b", re.IGNORECASE)

            async def _search_strain_asm(org_b: bytes, str_b: bytes) -> None:
                org_s = org_b.decode("utf-8", errors="replace")
                str_s = str_b.decode("utf-8", errors="replace")
                term = f'"{org_s}"[Organism] AND "{str_s}"[Infraspecies] AND "latest refseq"[Filter]'
                found_asm = False
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r = await client.get(
                            f"{NCBI_EUTILS_URL}/esearch.fcgi",
                            params={**base_params, "db": "assembly", "term": term, "retmax": "1"},
                        )
                        if r.status_code == 200:
                            idlist = r.json().get("esearchresult", {}).get("idlist", [])
                            if idlist:
                                uid_to_strain[str(idlist[0])] = (org_b, str_b)
                                found_asm = True
                    except Exception:
                        pass
                    await _polite_sleep(t_req)
                if not found_asm and collection_re.search(str_s):
                    unresolved_collection_strains.append((org_b, str_b))

            await asyncio.gather(*[_search_strain_asm(o_b, s_b) for o_b, s_b in top_strain_queries])

            biosample_uid_to_strain: dict[str, tuple[bytes, bytes]] = {}

            async def _search_biosample(org_b: bytes, str_b: bytes) -> None:
                org_s = org_b.decode("utf-8", errors="replace")
                str_s = str_b.decode("utf-8", errors="replace")
                bs_term = f'"{org_s}"[Organism] AND "{str_s}"'
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r_bs = await client.get(
                            f"{NCBI_EUTILS_URL}/esearch.fcgi",
                            params={**base_params, "db": "biosample", "term": bs_term, "retmax": "1"},
                        )
                        if r_bs.status_code == 200:
                            bs_ids = r_bs.json().get("esearchresult", {}).get("idlist", [])
                            if bs_ids:
                                biosample_uid_to_strain[str(bs_ids[0])] = (org_b, str_b)
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

            await asyncio.gather(*[_search_biosample(o_b, s_b) for o_b, s_b in unresolved_collection_strains[:30]])

            if biosample_uid_to_strain:
                bs_uids = sorted(biosample_uid_to_strain.keys())
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r_bss = await client.post(
                            f"{NCBI_EUTILS_URL}/esummary.fcgi",
                            data={**base_params, "db": "biosample", "id": ",".join(bs_uids)},
                        )
                        if r_bss.status_code == 200:
                            bs_res = r_bss.json().get("result", {})
                            for b_uid in bs_uids:
                                b_info = bs_res.get(b_uid, {})
                                if not isinstance(b_info, dict):
                                    continue
                                b_blob = f"{b_info.get('title', '')} {b_info.get('infraspecies', '')} {b_info.get('sampledata', '')}"
                                m_syn = re.search(r"\b((?:ATCC|NCTC|DSM|CCUG)\s*\d+)\b", b_blob, re.IGNORECASE)
                                if m_syn:
                                    syn_s = re.sub(r"\s+", " ", m_syn.group(1).strip())
                                    org_b, orig_str_b = biosample_uid_to_strain[b_uid]
                                    org_s = org_b.decode("utf-8", errors="replace")
                                    term_syn = f'"{org_s}"[Organism] AND "{syn_s}"[Infraspecies] AND "latest refseq"[Filter]'
                                    r_as = await client.get(
                                        f"{NCBI_EUTILS_URL}/esearch.fcgi",
                                        params={**base_params, "db": "assembly", "term": term_syn, "retmax": "1"},
                                    )
                                    if r_as.status_code == 200:
                                        a_ids = r_as.json().get("esearchresult", {}).get("idlist", [])
                                        if a_ids:
                                            uid_to_strain[str(a_ids[0])] = (org_b, orig_str_b)
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

            if uid_to_strain:
                uids = sorted(uid_to_strain.keys())
                try:
                    t_req = time.perf_counter()
                    r_sum = await client.post(
                        f"{NCBI_EUTILS_URL}/esummary.fcgi",
                        data={**base_params, "db": "assembly", "id": ",".join(uids)},
                    )
                    await _polite_sleep(t_req)
                    if r_sum.status_code == 200:
                        res_dict = r_sum.json().get("result", {})
                        for uid in uids:
                            info = res_dict.get(uid, {})
                            if not isinstance(info, dict):
                                continue
                            acc_str = info.get("assemblyaccession", "")
                            ftp_url = info.get("ftppath_refseq") or info.get("ftppath_genbank") or ""
                            if acc_str and ftp_url:
                                acc_b = acc_str.encode("utf-8")[:32]
                                strain_pair = uid_to_strain[uid]
                                if acc_b not in assembly_loci_cache:
                                    https_base = ftp_url.replace("ftp://", "https://")
                                    asm_name = https_base.rstrip("/").split("/")[-1]
                                    fna_url = f"{https_base}/{asm_name}_genomic.fna.gz"
                                    async with sem:
                                        try:
                                            f_resp = await client.get(fna_url, timeout=45.0)
                                            if f_resp.status_code == 200 and len(f_resp.content) > 100:
                                                raw_fna = gzip.decompress(f_resp.content)
                                                asm = GenomeAssembly.from_fasta(raw_fna, id_=acc_b)
                                                pred_loci = whole_genome_predictor.predict(asm)
                                                if len(pred_loci) > 0:
                                                    tagged = Loci.from_list(
                                                        [
                                                            loc.canonicalize_strand().with_name(acc_str[:32])
                                                            for loc in pred_loci
                                                        ]
                                                    )
                                                    assembly_loci_cache[acc_b] = (acc_b, tagged)
                                                else:
                                                    assembly_loci_cache[acc_b] = (acc_b, empty_loci_singleton)
                                        except Exception:
                                            pass
                                if acc_b in assembly_loci_cache and len(assembly_loci_cache[acc_b][1]) > 0:
                                    strain_to_assembly[strain_pair] = acc_b
                except Exception:
                    pass

            save_predicted_loci_cache(
        loci_cache_path,
        assembly_loci_cache,
        strain_to_assembly=strain_to_assembly,
        pmid_to_doi=pmid_to_doi,
    )
            print(
                f"[KBank] Tier 2 whole-genome assemblies complete ({len(strain_to_assembly)} strain assemblies, {time.perf_counter() - t_stage0:.1f}s)",
                flush=True,
            )

            taxon_record_counts: Counter[str] = Counter()
            taxon_target_seros: dict[str, set[str]] = defaultdict(set)
            taxon_target_strains: dict[str, set[str]] = defaultdict(set)
            key_record_weight: Counter[tuple[str, str]] = Counter()

            for r in raw_records:
                acc_b = r.get("accession_hint", b"")
                if acc_b and acc_b.decode("utf-8", errors="replace").split(".")[0] in base_acc_in_cache:
                    continue
                org_b = r["organism"]
                str_b = r["strain"]
                sero_b = r["serotype"]
                if (org_b, str_b) in strain_to_assembly:
                    continue
                taxon_q = _catalog_taxon(org_b)
                if not taxon_q:
                    continue
                taxon_record_counts[taxon_q] += 1
                if sero_b:
                    sero_s = sero_b.decode("utf-8", errors="replace").strip()
                    taxon_target_seros[taxon_q].add(sero_s)
                    key_record_weight[(taxon_q, sero_s)] += 1
                    m_base = re.match(r"^([KO]\d+)[a-z].*$", sero_s)
                    if m_base:
                        base_s = m_base.group(1)
                        taxon_target_seros[taxon_q].add(base_s)
                        key_record_weight[(taxon_q, base_s)] += 1
                if str_b and len(str_b) >= 3 and str_b != sero_b:
                    str_s = str_b.decode("utf-8", errors="replace").strip()
                    if not str_s.isdigit():
                        taxon_target_strains[taxon_q].add(str_s)
                        key_record_weight[(taxon_q, str_s)] += 1

            bgc_title_filter = (
                "AND 3000:60000[SLEN] AND ("
                '"capsular"[Title] OR "capsule"[Title] OR "cps"[Title] OR "kps"[Title] '
                'OR "K-antigen"[Title] OR "O-antigen"[Title] OR "lipopolysaccharide"[Title] '
                'OR "lipooligosaccharide"[Title] OR "polysaccharide"[Title] OR "exopolysaccharide"[Title] '
                'OR "teichoic"[Title] OR "outer core"[Title] OR "B-band"[Title] '
                'OR "O antigen"[Title] OR "K antigen"[Title] OR "wb"[Title] OR "waa"[Title] OR "wba"[Title]'
                ') NOT "whole genome shotgun"[Title] NOT "patent"[Filter] NOT "targeted locus study"[Title] '
                'NOT "pyocin"[Title] NOT "16S"[Title] NOT "plasmid"[Title] NOT "phage"[Title]'
            )

            landmark_sweeps: dict[str, str] = {
                "Streptococcus pneumoniae": "CR931632:CR931722[ACCN]",
                "Pseudomonas aeruginosa": "AF498400:AF498420[ACCN]",
                "Klebsiella": "AB924547:AB924624[ACCN]",
            }

            key_to_candidates: dict[tuple[str, str], list[tuple[str, str, bool]]] = defaultdict(list)
            top_taxa = [tq for tq, _ in taxon_record_counts.most_common()]

            async def _catalog_single_taxon(taxon_q: str) -> None:
                term = f'"{taxon_q}"[Organism] {bgc_title_filter}'
                idlist: list[str] = []
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r_es = await client.get(
                            f"{NCBI_EUTILS_URL}/esearch.fcgi",
                            params={**base_params, "db": "nuccore", "term": term, "retmax": "1000"},
                        )
                        if r_es.status_code == 200:
                            idlist = [str(x) for x in r_es.json().get("esearchresult", {}).get("idlist", [])]
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

                if taxon_q in landmark_sweeps:
                    async with sem:
                        t_req = time.perf_counter()
                        try:
                            r_lm = await client.get(
                                f"{NCBI_EUTILS_URL}/esearch.fcgi",
                                params={
                                    **base_params,
                                    "db": "nuccore",
                                    "term": landmark_sweeps[taxon_q],
                                    "retmax": "200",
                                },
                            )
                            if r_lm.status_code == 200:
                                lm_ids = [str(x) for x in r_lm.json().get("esearchresult", {}).get("idlist", [])]
                                seen_id_set = set(idlist)
                                for lid in lm_ids:
                                    if lid not in seen_id_set:
                                        idlist.append(lid)
                                        seen_id_set.add(lid)
                        except Exception:
                            pass
                        await _polite_sleep(t_req)

                if not idlist:
                    return

                sum_dict: dict[str, Any] = {}
                for s_i in range(0, len(idlist), 350):
                    chunk_ids = idlist[s_i : s_i + 350]
                    async with sem:
                        t_req = time.perf_counter()
                        try:
                            r_sum = await client.post(
                                f"{NCBI_EUTILS_URL}/esummary.fcgi",
                                data={**base_params, "db": "nuccore", "id": ",".join(chunk_ids)},
                            )
                            if r_sum.status_code == 200:
                                sum_dict.update(r_sum.json().get("result", {}))
                        except Exception:
                            pass
                        await _polite_sleep(t_req)

                t_all_targets = taxon_target_seros[taxon_q] | taxon_target_strains[taxon_q]
                num_map, word_map, multi_map = _build_token_index(t_all_targets)

                for uid in idlist:
                    info = sum_dict.get(uid, {})
                    if not isinstance(info, dict):
                        continue
                    acc_ver = (info.get("accessionversion") or info.get("caption") or "").strip()
                    if not acc_ver:
                        continue
                    uid_to_accver[uid] = acc_ver
                    title = info.get("title", "")
                    subtype = info.get("subtype", "")
                    subname = info.get("subname", "")
                    blob_txt = f"{title} | {subname}"
                    title_low = title.lower()
                    is_cps_t = any(w in title_low for w in ("capsul", "cps", "kps", "k-antigen", "k antigen"))
                    sub_pairs = list(zip(subtype.split("|"), subname.split("|")))

                    matched_toks = _find_matching_tokens(title, blob_txt, sub_pairs, num_map, word_map, multi_map)
                    for m_tok in matched_toks:
                        cand_list = key_to_candidates[(taxon_q, m_tok)]
                        if not any(u == uid for u, _, _ in cand_list):
                            cand_list.append((uid, acc_ver, is_cps_t))

            await asyncio.gather(*[_catalog_single_taxon(tq) for tq in top_taxa])
            print(
                f"[KBank] Tier 3 nuccore catalog indexed ({len(key_to_candidates)} taxon/token candidate pools, {time.perf_counter() - t_stage0:.1f}s)",
                flush=True,
            )

            selected_key_uids: dict[tuple[str, str], list[str]] = {}
            uid_priority: Counter[str] = Counter()

            for u in tier1_uids:
                if u in uid_to_meta:
                    uid_priority[u] += 50

            for p_val, u_list in sorted(pmid_to_nuccore_uids.items()):
                valid_u = [u for u in u_list if u in uid_to_meta]
                for idx_u, u in enumerate(valid_u[:8]):
                    uid_priority[u] += 25 - idx_u

            for key_pair, cands in sorted(key_to_candidates.items()):
                cps_u = [u for u, _, is_c in cands if is_c][:2]
                ops_u = [u for u, _, is_c in cands if not is_c][:2]
                merged: list[str] = []
                for u in cps_u + ops_u:
                    if u not in merged and len(merged) < 3:
                        merged.append(u)
                if merged:
                    selected_key_uids[key_pair] = merged
                    w = key_record_weight.get(key_pair, 1)
                    for idx_u, u in enumerate(merged):
                        uid_priority[u] += w * (3 - idx_u)

            top_nuccore_uids = [
                u for u, _ in sorted(uid_priority.items(), key=lambda kv: (-kv[1], kv[0]))[:1200]
            ]
            uid_to_predicted: dict[str, tuple[bytes, list[Locus]]] = {}
            accbase_to_uids: dict[str, list[str]] = defaultdict(list)
            uids_to_fetch: list[str] = []

            for u in top_nuccore_uids:
                av = uid_to_accver.get(u, "")
                base_av = av.split(".")[0] if av else ""
                if base_av:
                    accbase_to_uids[base_av].append(u)
                uid_cache_key = b"uid:" + u.encode("utf-8")
                if base_av and base_av in base_acc_in_cache:
                    cached_key = base_acc_in_cache[base_av]
                    c_id, c_loci = assembly_loci_cache[cached_key]
                    if len(c_loci) > 0:
                        uid_to_predicted[u] = (c_id, [c_loci[j] for j in range(len(c_loci))])
                elif uid_cache_key in assembly_loci_cache:
                    c_id, c_loci = assembly_loci_cache[uid_cache_key]
                    if len(c_loci) > 0:
                        uid_to_predicted[u] = (c_id, [c_loci[j] for j in range(len(c_loci))])
                else:
                    uids_to_fetch.append(u)

            for i in range(0, len(uids_to_fetch), 40):
                batch_uids = uids_to_fetch[i : i + 40]
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        params = {
                            "db": "nuccore",
                            "id": ",".join(batch_uids),
                            "rettype": "fasta",
                            "retmode": "text",
                        }
                        if api_key:
                            params["api_key"] = api_key
                        resp = await client.get(f"{NCBI_EUTILS_URL}/efetch.fcgi", params=params, timeout=40.0)
                        if resp.status_code == 200 and resp.content.startswith(b">"):
                            grouped = _predict_multi_fasta_batch(resp.content)
                            for base_tok, (acc_b, tagged_list) in grouped.items():
                                loci_obj = Loci.from_list(tagged_list) if tagged_list else empty_loci_singleton
                                assembly_loci_cache[acc_b] = (acc_b, loci_obj)
                                base_acc_in_cache[base_tok] = acc_b
                                for matched_u in accbase_to_uids.get(base_tok, []):
                                    assembly_loci_cache[b"uid:" + matched_u.encode("utf-8")] = (acc_b, loci_obj)
                                    if tagged_list:
                                        uid_to_predicted[matched_u] = (acc_b, tagged_list)
                            for u_b in batch_uids:
                                u_key = b"uid:" + u_b.encode("utf-8")
                                if u_key not in assembly_loci_cache:
                                    av_u = uid_to_accver.get(u_b, "")
                                    base_u = av_u.split(".")[0] if av_u else ""
                                    acc_u_b = base_u.encode("utf-8")[:32] if base_u else b"empty"
                                    assembly_loci_cache[u_key] = (acc_u_b, empty_loci_singleton)
                                    if base_u and base_u not in base_acc_in_cache:
                                        assembly_loci_cache[acc_u_b] = (acc_u_b, empty_loci_singleton)
                                        base_acc_in_cache[base_u] = acc_u_b
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

            save_predicted_loci_cache(
        loci_cache_path,
        assembly_loci_cache,
        strain_to_assembly=strain_to_assembly,
        pmid_to_doi=pmid_to_doi,
    )
            print(
                f"[KBank] Tier 1.5 + Tier 3 BGC FASTA segmentation complete ({len(uid_to_predicted)} BGCs predicted, {time.perf_counter() - t_stage0:.1f}s)",
                flush=True,
            )

            for p_val, u_list in pmid_to_nuccore_uids.items():
                pred_u = [u for u in u_list if u in uid_to_predicted]
                if not pred_u:
                    continue
                paper_pool: list[Locus] = []
                first_p_acc = b""
                for u in pred_u[:8]:
                    acc_b, l_list = uid_to_predicted[u]
                    if not first_p_acc:
                        first_p_acc = acc_b
                    paper_pool.extend(l_list)
                if paper_pool and first_p_acc:
                    pmid_key = b"pmid:%d" % p_val
                    assembly_loci_cache[pmid_key] = (first_p_acc, Loci.from_list(paper_pool))

            for r in raw_records:
                p_val = int(r.get("pmid", 0))
                if p_val <= 0 or p_val not in pmid_to_nuccore_uids:
                    continue
                pred_u = [u for u in pmid_to_nuccore_uids[p_val] if u in uid_to_predicted and u in uid_to_meta]
                if not pred_u:
                    continue
                r_toks = {
                    tok_b.decode("utf-8", errors="replace").strip(): tok_b
                    for tok_b in (r["serotype"], r["strain"])
                    if tok_b
                }
                if not r_toks:
                    continue
                p_num_map, p_word_map, p_multi_map = _build_token_index(set(r_toks.keys()))
                tok_to_loci: dict[bytes, tuple[bytes, list[Locus]]] = {}
                for u in pred_u:
                    acc_ver, title, subname, subtype, _ = uid_to_meta[u]
                    blob_txt = f"{title} | {subname}"
                    sub_pairs = list(zip(subtype.split("|"), subname.split("|")))
                    m_toks = _find_matching_tokens(title, blob_txt, sub_pairs, p_num_map, p_word_map, p_multi_map)
                    for m_s in m_toks:
                        tok_b = r_toks[m_s]
                        acc_b, l_list = uid_to_predicted[u]
                        if tok_b not in tok_to_loci:
                            tok_to_loci[tok_b] = (acc_b, list(l_list))
                        else:
                            tok_to_loci[tok_b][1].extend(l_list)
                for tok_b, (first_m_acc, matched_loci) in tok_to_loci.items():
                    if matched_loci and first_m_acc:
                        spec_key = b"pmid:%d:%s" % (p_val, tok_b)
                        if spec_key not in assembly_loci_cache:
                            assembly_loci_cache[spec_key] = (first_m_acc, Loci.from_list(matched_loci))

            for r in raw_records:
                org_b = r["organism"]
                taxon_q = _catalog_taxon(org_b)
                if not taxon_q:
                    continue
                for tok_b in (r["strain"], r["serotype"]):
                    if not tok_b or (org_b, tok_b) in strain_to_assembly:
                        continue
                    tok_s = tok_b.decode("utf-8", errors="replace").strip()
                    uids_for_tok = selected_key_uids.get((taxon_q, tok_s), [])
                    if not uids_for_tok:
                        m_base = re.match(r"^([KO]\d+)[a-z].*$", tok_s)
                        if m_base:
                            uids_for_tok = selected_key_uids.get((taxon_q, m_base.group(1)), [])
                    if not uids_for_tok:
                        continue
                    pool_loci: list[Locus] = []
                    first_acc_b = b""
                    for u in uids_for_tok:
                        if u in uid_to_predicted:
                            acc_b, l_list = uid_to_predicted[u]
                            if not first_acc_b:
                                first_acc_b = acc_b
                            pool_loci.extend(l_list)
                    if pool_loci and first_acc_b:
                        cache_key = b"nuccore:%s:%s" % (taxon_q.encode("utf-8"), tok_b)
                        if cache_key not in assembly_loci_cache:
                            assembly_loci_cache[cache_key] = (first_acc_b, Loci.from_list(pool_loci))
                        strain_to_assembly[(org_b, tok_b)] = cache_key

            unique_pmids = sorted(
                {int(r["pmid"]) for r in raw_records if int(r.get("pmid", 0)) > 0 and not r.get("doi")}
            )

            async def _fetch_doi_batch(batch_pmids: list[str]) -> None:
                async with sem:
                    t_req = time.perf_counter()
                    try:
                        r_pm = await client.post(
                            f"{NCBI_EUTILS_URL}/esummary.fcgi",
                            data={**base_params, "db": "pubmed", "id": ",".join(batch_pmids)},
                        )
                        if r_pm.status_code == 200:
                            pm_res = r_pm.json().get("result", {})
                            for pid_s in batch_pmids:
                                art_ids = pm_res.get(pid_s, {}).get("articleids", [])
                                for aid in art_ids:
                                    if aid.get("idtype") == "doi" and aid.get("value"):
                                        pmid_to_doi[int(pid_s)] = aid["value"].strip().encode("utf-8")
                                        break
                    except Exception:
                        pass
                    await _polite_sleep(t_req)

            doi_batches = [
                [str(p) for p in unique_pmids[i : i + 150]] for i in range(0, min(len(unique_pmids), 800), 150)
            ]
            await asyncio.gather(*[_fetch_doi_batch(b) for b in doi_batches])

    save_predicted_loci_cache(
        loci_cache_path,
        assembly_loci_cache,
        strain_to_assembly=strain_to_assembly,
        pmid_to_doi=pmid_to_doi,
    )
    print(
        f"[KBank] resolve_and_predict_loci finished in {time.perf_counter() - t_stage0:.1f}s",
        flush=True,
    )
    return assembly_loci_cache, strain_to_assembly, pmid_to_doi


def build_curated_glycan_records(
    raw_records: list[dict[str, Any]],
    assembly_loci_cache: dict[bytes, tuple[bytes, Loci]],
    strain_to_assembly: dict[tuple[bytes, bytes], bytes],
    pmid_to_doi: dict[int, bytes],
    canonical_to_gtc: dict[bytes, bytes] | None = None,
) -> GlycanRecords:
    """Consolidate duplicate/fragment records, graft modifiers, pair with `Loci`, and compile `GlycanRecords`."""
    normalize_raw_record_taxonomy(raw_records)
    assigner = BiosyntheticAssigner()
    gtc_lookup = canonical_to_gtc or {}
    seen_keys: set[bytes] = set()
    initial_records: list[GlycanRecord] = []

    base_acc_lookup: dict[bytes, tuple[bytes, Loci]] = {
        k_acc.split(b".")[0]: val
        for k_acc, val in assembly_loci_cache.items()
        if not k_acc.startswith((b"nuccore:", b"pmid:"))
    }
    match_memo: dict[tuple[bytes, int, ArchetypeCategory | None, LocusArchetype | None], Any] = {}

    sero_acc_hints: dict[tuple[bytes, bytes], list[bytes]] = defaultdict(list)
    sero_pmids: dict[tuple[bytes, bytes], list[int]] = defaultdict(list)
    sero_strains: dict[tuple[bytes, bytes], list[bytes]] = defaultdict(list)
    comp_key_hints: dict[bytes, tuple[bytes, int, bytes]] = {}

    for r in raw_records:
        s_b = r.get("serotype", b"")
        acc_h = r.get("accession_hint", b"")
        pm_i = int(r.get("pmid", 0))
        st_b = r.get("strain", b"")
        ck = b"%s:%s" % (r["source_db"], r["record_id"])
        comp_key_hints[ck] = (acc_h, pm_i, st_b)

        if s_b:
            sk = (r["organism"], s_b)
            if acc_h and acc_h not in sero_acc_hints[sk]:
                sero_acc_hints[sk].append(acc_h)
            if pm_i > 0 and pm_i not in sero_pmids[sk]:
                sero_pmids[sk].append(pm_i)
            if st_b and st_b not in sero_strains[sk]:
                sero_strains[sk].append(st_b)

    for r in raw_records:
        comp_key = b"%s:%s" % (r["source_db"], r["record_id"])
        if comp_key in seen_keys:
            continue
        seen_keys.add(comp_key)

        gly: Glycan = r["glycan"]
        ag_type: bytes = r["antigen_type"]
        notes_b: bytes = r["notes"]
        hint_text = b"%s %s" % (ag_type, notes_b)

        expected_cat, expected_arch = assigner.infer_expected_archetype(hint_text)
        default_arch = (
            expected_arch
            if expected_arch is not None
            else (
                LocusArchetype.CPS_WZY
                if expected_cat == ArchetypeCategory.CAPSULE
                else (
                    LocusArchetype.LPS_WZY
                    if expected_cat == ArchetypeCategory.O_ANTIGEN
                    else (
                        LocusArchetype.EPS_PGA
                        if expected_cat == ArchetypeCategory.EXOPOLYSACCHARIDE
                        else LocusArchetype.UNKNOWN
                    )
                )
            )
        )

        pmid = int(r.get("pmid", 0))
        doi_b = r.get("doi", b"") or pmid_to_doi.get(pmid, b"")
        gtc_b = r.get("glytoucan_id", b"") or gtc_lookup.get(gly.canonical, b"")

        initial_records.append(
            GlycanRecord(
                source_db=r["source_db"],
                record_id=r["record_id"],
                organism=r["organism"],
                strain=r["strain"],
                serotype=r["serotype"],
                antigen_type=ag_type,
                assembly_id=r.get("accession_hint", b""),
                archetype=default_arch,
                locus=None,
                alignment_score=0.0,
                structure_raw=r["structure_raw"],
                structure_linear=r["structure_linear"],
                glycan=gly,
                is_repeating=bool(r["is_repeating"]),
                pmid=pmid,
                publication_year=int(r.get("publication_year", 0)),
                doi=doi_b,
                glytoucan_id=gtc_b,
                notes=notes_b,
            )
        )

    # 1. Consolidate duplicates/fragments & graft fragment modifiers BEFORE locus matching
    pre_consolidated = consolidate_curated_records(initial_records)

    # 2. Match consolidated canonical records against 4-tier candidate Loci pools
    paired_records: list[GlycanRecord] = []
    merged_ids_re = re.compile(rb"\[merged_ids=([^\]]+)\]")

    for rec in pre_consolidated:
        gly = rec.glycan
        hint_text = b"%s %s" % (rec.antigen_type, rec.notes)
        expected_cat, expected_arch = assigner.infer_expected_archetype(hint_text)
        default_arch = rec.archetype

        sero_key = (rec.organism, rec.serotype)
        acc_candidates: list[bytes] = ([rec.assembly_id] if rec.assembly_id else []) + (
            list(sero_acc_hints.get(sero_key, [])) if rec.serotype else []
        )
        pmid_candidates: list[int] = ([rec.pmid] if rec.pmid > 0 else []) + (
            list(sero_pmids.get(sero_key, [])) if rec.serotype else []
        )
        strain_candidates: list[bytes] = ([rec.strain] if rec.strain else []) + (
            list(sero_strains.get(sero_key, [])) if rec.serotype else []
        )

        # Also harvest any accession/PMID/strain hints from absorbed cluster members
        m_merged = merged_ids_re.search(rec.notes)
        if m_merged:
            for mk in m_merged.group(1).split(b","):
                if mk in comp_key_hints:
                    m_acc, m_pmid, m_str = comp_key_hints[mk]
                    if m_acc and m_acc not in acc_candidates:
                        acc_candidates.append(m_acc)
                    if m_pmid > 0 and m_pmid not in pmid_candidates:
                        pmid_candidates.append(m_pmid)
                    if m_str and m_str not in strain_candidates:
                        strain_candidates.append(m_str)

        resolved_asm_id = b""
        cand_loci: Loci | None = None
        from_untagged_pmid_pool = False

        for acc_hint in acc_candidates:
            if acc_hint in assembly_loci_cache:
                resolved_asm_id, cand_loci = assembly_loci_cache[acc_hint]
                break
            base_acc = acc_hint.split(b".")[0]
            if base_acc in base_acc_lookup:
                resolved_asm_id, cand_loci = base_acc_lookup[base_acc]
                break
            if not resolved_asm_id:
                resolved_asm_id = acc_hint

        if cand_loci is None:
            for pmid_cand in pmid_candidates:
                for tok_b in [rec.serotype, *strain_candidates]:
                    if tok_b:
                        spec_key = b"pmid:%d:%s" % (pmid_cand, tok_b)
                        if spec_key in assembly_loci_cache:
                            resolved_asm_id, cand_loci = assembly_loci_cache[spec_key]
                            break
                if cand_loci is not None:
                    break

        if cand_loci is None:
            for st_cand in strain_candidates:
                strain_key = (rec.organism, st_cand)
                if strain_key in strain_to_assembly:
                    cache_key = strain_to_assembly[strain_key]
                    if cache_key in assembly_loci_cache:
                        resolved_asm_id, cand_loci = assembly_loci_cache[cache_key]
                        break

        if cand_loci is None and rec.serotype and sero_key in strain_to_assembly:
            cache_key = strain_to_assembly[sero_key]
            if cache_key in assembly_loci_cache:
                resolved_asm_id, cand_loci = assembly_loci_cache[cache_key]

        if cand_loci is None:
            for pmid_cand in pmid_candidates:
                pmid_key = b"pmid:%d" % pmid_cand
                if pmid_key in assembly_loci_cache:
                    resolved_asm_id, cand_loci = assembly_loci_cache[pmid_key]
                    from_untagged_pmid_pool = True
                    break

        matched_locus: Locus | None = None
        matched_arch: LocusArchetype = default_arch
        align_score: float = 0.0

        if cand_loci is not None and len(cand_loci) > 0:
            memo_key = (gly.canonical, len(gly.nodes), id(cand_loci), expected_cat, expected_arch)
            if memo_key in match_memo:
                match_res = match_memo[memo_key]
            else:
                match_res = assigner.match_locus(
                    gly,
                    cand_loci,
                    antigen_hint=hint_text,
                    expected_category=expected_cat,
                    expected_archetype=expected_arch,
                )
                match_memo[memo_key] = match_res

            if match_res is not None and (not from_untagged_pmid_pool or match_res.bio_score >= 0.30):
                matched_locus = match_res.locus
                if matched_locus.name:
                    resolved_asm_id = (
                        matched_locus.name.encode("utf-8")[:32]
                        if isinstance(matched_locus.name, str)
                        else bytes(matched_locus.name)[:32]
                    )
                matched_arch = match_res.archetype if match_res.archetype != LocusArchetype.UNKNOWN else default_arch
                align_score = float(match_res.combined_score)
            elif from_untagged_pmid_pool:
                resolved_asm_id = b""

        paired_records.append(
            replace(
                rec,
                assembly_id=resolved_asm_id,
                archetype=matched_arch,
                locus=matched_locus,
                alignment_score=align_score,
            )
        )

    return GlycanRecords.build(paired_records)
