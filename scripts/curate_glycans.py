# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "aiohttp>=3.14.3",
#     "altair>=5.5.0",
#     "cassetter>=0.14.0",
#     "gb-io",
#     "h2>=4.4.1",
#     "httpx>=0.28.1",
#     "httpx2>=2.13.0",
#     "kbank",
#     "marimo>=0.23.1",
#     "numpy>=2.4.4",
#     "pyarrow>=25.0.1",
#     "scipy>=1.17.1",
#     "serogen",
# ]
#
# [tool.uv.sources]
# serogen = { path = "../../SeroGen", editable = true }
# kbank = { path = "..", editable = true }
# ///

"""Unified KBank Bacterial Surface Glycan & Biosynthetic Locus Curation Marimo Slide Deck."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    app_title="KBank — Bacterial Surface Glycan & Biosynthetic Locus Curation",
    layout_file="layouts/curate_glycans.slides.json",
)


@app.cell
def _():
    import asyncio
    from collections import Counter
    import html
    import os
    from pathlib import Path
    import time

    import altair as alt
    from cassetter import Cassetter, RawRequest, RawResponse, SkipRecording
    import marimo as mo
    import numpy as np
    from serogen.genes.roles import GeneRole
    from serogen.loci.archetypes import LocusArchetype

    from kbank.db.curation import (
        DEFAULT_LOCI_CACHE_PATH,
        build_curated_glycan_records,
        normalize_raw_record_taxonomy,
        resolve_and_predict_loci,
    )
    from kbank.db.records import DEFAULT_DB_PATH, GlycanRecords
    from kbank.glycan import Anomer, LinkageType
    from kbank.io.csdb import CSDBParser
    from kbank.io.iupac import IUPACParser
    from kbank.io.sources import (
        DEFAULT_CACHE_DIR,
        DEFAULT_CSDB_CACHE_PATH,
        fetch_ecodab_records,
        fetch_ek3d_records,
        fetch_epsdb_records,
        fetch_glytoucan_records,
        fetch_kegg_glycan_records,
        stream_csdb_database,
    )
    from kbank.sugars import Sugars

    alt.data_transformers.disable_max_rows()

    CASSETTE_DIR = DEFAULT_CACHE_DIR / "cassettes"
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)

    def _skip_efetch_requests(request: RawRequest) -> RawRequest:
        if "efetch.fcgi" in request.uri:
            raise SkipRecording
        return request

    def _skip_error_responses(response: RawResponse) -> RawResponse:
        if response.status >= 400:
            raise SkipRecording
        return response

    RECORDER = Cassetter(
        cassette_library_dir=CASSETTE_DIR,
        record_mode=os.environ.get("KBANK_RECORD_MODE", "new_episodes"),
        match_on=["method", "uri", "body"],
        filter_query_parameters=["api_key"],
        ignore_hosts=["ftp.ncbi.nlm.nih.gov", "osf.io", "*.osf.io"],
        before_record_request=_skip_efetch_requests,
        before_record_response=_skip_error_responses,
    )

    return (
        Anomer,
        CSDBParser,
        Counter,
        DEFAULT_CSDB_CACHE_PATH,
        DEFAULT_DB_PATH,
        DEFAULT_LOCI_CACHE_PATH,
        GeneRole,
        GlycanRecords,
        IUPACParser,
        LinkageType,
        LocusArchetype,
        Path,
        RECORDER,
        Sugars,
        alt,
        asyncio,
        build_curated_glycan_records,
        fetch_ecodab_records,
        fetch_ek3d_records,
        fetch_epsdb_records,
        fetch_glytoucan_records,
        fetch_kegg_glycan_records,
        html,
        mo,
        normalize_raw_record_taxonomy,
        np,
        resolve_and_predict_loci,
        stream_csdb_database,
        time,
    )


@app.cell
async def _(
    Anomer,
    CSDBParser,
    Counter,
    DEFAULT_CSDB_CACHE_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_LOCI_CACHE_PATH,
    GeneRole,
    GlycanRecords,
    IUPACParser,
    LinkageType,
    LocusArchetype,
    Path,
    RECORDER,
    Sugars,
    alt,
    asyncio,
    build_curated_glycan_records,
    fetch_ecodab_records,
    fetch_ek3d_records,
    fetch_epsdb_records,
    fetch_glytoucan_records,
    fetch_kegg_glycan_records,
    html,
    normalize_raw_record_taxonomy,
    np,
    resolve_and_predict_loci,
    stream_csdb_database,
    time,
):
    t0 = time.perf_counter()
    _repo_root = Path(__file__).resolve().parent.parent
    csdb_parser = CSDBParser()
    iupac_parser = IUPACParser()

    if DEFAULT_DB_PATH.exists():
        glycan_records = GlycanRecords.load(DEFAULT_DB_PATH)
        csdb_index_count = 16908
        assembly_cache_count = int(len(np.unique(glycan_records.assembly_ids)) - 1)
    else:
        csdb_records, csdb_index = await asyncio.to_thread(
            stream_csdb_database,
            csdb_parser,
            iupac_parser,
            cache_path=DEFAULT_CSDB_CACHE_PATH,
        )
        ecodab_records, ek3d_records, epsdb_records, kegg_records, canonical_to_gtc = (
            await asyncio.gather(
                fetch_ecodab_records(csdb_index, csdb_parser, iupac_parser, recorder=RECORDER),
                fetch_ek3d_records(csdb_index, csdb_parser, iupac_parser, recorder=RECORDER),
                fetch_epsdb_records(csdb_index, csdb_parser, iupac_parser, recorder=RECORDER),
                fetch_kegg_glycan_records(recorder=RECORDER, allow_composition=False),
                fetch_glytoucan_records(csdb_parser, iupac_parser, recorder=RECORDER),
            )
        )
        all_raw_records = normalize_raw_record_taxonomy(
            csdb_records + ecodab_records + ek3d_records + epsdb_records + kegg_records
        )
        assembly_loci_cache, strain_to_assembly, pmid_to_doi = await resolve_and_predict_loci(
            all_raw_records,
            max_assemblies=100,
            recorder=RECORDER,
            loci_cache_path=DEFAULT_LOCI_CACHE_PATH,
        )
        glycan_records = build_curated_glycan_records(
            all_raw_records,
            assembly_loci_cache,
            strain_to_assembly,
            pmid_to_doi,
            canonical_to_gtc=canonical_to_gtc,
        )
        csdb_index_count = len(csdb_index)
        assembly_cache_count = len(assembly_loci_cache)

    curation_elapsed_s = time.perf_counter() - t0

    npz_path = glycan_records.save(DEFAULT_DB_PATH)
    parquet_path = glycan_records.to_parquet(_repo_root / "data" / "kbank.parquet")
    _reloaded = GlycanRecords.load(npz_path)
    npz_parity = bool(
        len(_reloaded) == len(glycan_records)
        and np.array_equal(_reloaded.compound_keys, glycan_records.compound_keys)
        and np.array_equal(_reloaded.glycans.node_types, glycan_records.glycans.node_types)
        and np.array_equal(_reloaded.loci.node_types, glycan_records.loci.node_types)
    )
    npz_kb = npz_path.stat().st_size / 1024.0
    parquet_kb = parquet_path.stat().st_size / 1024.0

    # 1. Source Database x Antigen Class Stacked Bar Chart
    _src_ag_counts: Counter[tuple[str, str]] = Counter()
    _src_totals: Counter[str] = Counter()
    for _db_b, _ag_b in zip(glycan_records.source_dbs, glycan_records.antigen_types):
        _db_s = _db_b.decode("utf-8", errors="replace")
        _ag_s = _ag_b.decode("utf-8", errors="replace") or "Other"
        _src_ag_counts[(_db_s, _ag_s)] += 1
        _src_totals[_db_s] += 1

    _source_rows = [
        {
            "source_db": _db_s,
            "antigen_type": _ag_s,
            "count": _cnt,
            "db_total": _src_totals[_db_s],
        }
        for (_db_s, _ag_s), _cnt in _src_ag_counts.items()
    ]
    source_chart = (
        alt.Chart(alt.Data(values=_source_rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "source_db:N",
                sort=alt.EncodingSortField(field="db_total", op="max", order="descending"),
                title="Source Database",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("count:Q", title="Curated GlycanRecords"),
            color=alt.Color(
                "antigen_type:N",
                title="Surface Antigen Class",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip("source_db:N", title="Source DB"),
                alt.Tooltip("antigen_type:N", title="Antigen Class"),
                alt.Tooltip("count:Q", title="Records", format=","),
                alt.Tooltip("db_total:Q", title="DB Total", format=","),
            ],
        )
        .properties(title="Curated GlycanRecords by Source Database & Antigen Class", width=360, height=320)
    )

    # 2. Top 15 Bacterial Species Horizontal Stacked Bar Chart
    _orgs, _org_counts = np.unique(glycan_records.organisms, return_counts=True)
    _org_order = np.argsort(-_org_counts)[:15]
    _species_rows = []
    for _idx in _org_order:
        _org_b = _orgs[_idx]
        _org_name = _org_b.decode("utf-8", errors="replace")
        _tot_c = int(_org_counts[_idx])
        _mask = glycan_records.organisms == _org_b
        _paired_cnt = int(np.sum(glycan_records.has_locus[_mask]))
        _unpaired_cnt = _tot_c - _paired_cnt
        _species_rows.append(
            {
                "organism": _org_name,
                "status": "Paired SeroGen Locus",
                "records": _paired_cnt,
                "total_count": _tot_c,
                "paired_loci": _paired_cnt,
            }
        )
        _species_rows.append(
            {
                "organism": _org_name,
                "status": "Glycan Structure Only",
                "records": _unpaired_cnt,
                "total_count": _tot_c,
                "paired_loci": _paired_cnt,
            }
        )

    species_chart = (
        alt.Chart(alt.Data(values=_species_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "organism:N",
                sort=alt.EncodingSortField(field="total_count", op="max", order="descending"),
                title="Bacterial Species / Taxon",
            ),
            x=alt.X("records:Q", title="Curated GlycanRecords"),
            color=alt.Color(
                "status:N",
                title="Locus Pairing Status",
                scale=alt.Scale(
                    domain=["Paired SeroGen Locus", "Glycan Structure Only"],
                    range=["#059669", "#a7f3d0"],
                ),
            ),
            order=alt.Order("status:N", sort="descending"),
            tooltip=[
                alt.Tooltip("organism:N", title="Organism"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("records:Q", title="Records", format=","),
                alt.Tooltip("paired_loci:Q", title="Paired SeroGen Loci", format=","),
                alt.Tooltip("total_count:Q", title="Species Total", format=","),
            ],
        )
        .properties(title="Top 15 Bacterial Species & Paired SeroGen Loci", width=380, height=320)
    )

    # 3. Top 18 Monosaccharides & Modifiers Horizontal Bar Chart
    _node_types = glycan_records.glycans.node_types
    _sugar_ids = _node_types & 0xFF
    _u_sugars, _s_counts = np.unique(_sugar_ids, return_counts=True)
    _s_order = np.argsort(-_s_counts)[:18]
    _sugar_rows = []
    for _idx in _s_order:
        _sid = int(_u_sugars[_idx])
        _sname = Sugars.names[_sid].decode("ascii", errors="replace")
        _is_mod = bool(Sugars.classes[_sid] == b"Mod")
        _sugar_rows.append(
            {
                "residue": _sname,
                "count": int(_s_counts[_idx]),
                "kind": "Modifier / Aglycon" if _is_mod else "Monosaccharide",
            }
        )

    sugar_chart = (
        alt.Chart(alt.Data(values=_sugar_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "residue:N",
                sort=alt.EncodingSortField(field="count", order="descending"),
                title="Residue / Modifier",
            ),
            x=alt.X("count:Q", title="Total Node Occurrences"),
            color=alt.Color(
                "kind:N",
                title="Node Type",
                scale=alt.Scale(domain=["Monosaccharide", "Modifier / Aglycon"], range=["#059669", "#d97706"]),
            ),
            tooltip=[
                alt.Tooltip("residue:N", title="Residue"),
                alt.Tooltip("kind:N", title="Node Type"),
                alt.Tooltip("count:Q", title="Occurrences", format=","),
            ],
        )
        .properties(title="Top 18 Monosaccharides & Modifiers across KBank", width=360, height=350)
    )

    # 4. Top 18 Glycosidic Linkages Horizontal Bar Chart
    _edge_types = glycan_records.glycans.edge_types
    _u_edges, _e_counts = np.unique(_edge_types, return_counts=True)
    _e_order = np.argsort(-_e_counts)[:18]
    _linkage_rows = []
    for _idx in _e_order:
        _lt = LinkageType(int(_u_edges[_idx]))
        _lname = _lt.name.decode("ascii", errors="replace")
        _anom_label = (
            "Alpha (a)"
            if _lt.anomer == Anomer.ALPHA
            else ("Beta (b)" if _lt.anomer == Anomer.BETA else "Unknown / Non-glycosidic")
        )
        _linkage_rows.append(
            {
                "linkage": _lname,
                "count": int(_e_counts[_idx]),
                "anomer": _anom_label,
            }
        )

    linkage_chart = (
        alt.Chart(alt.Data(values=_linkage_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "linkage:N",
                sort=alt.EncodingSortField(field="count", order="descending"),
                title="Glycosidic Linkage",
            ),
            x=alt.X("count:Q", title="Total Edge Occurrences"),
            color=alt.Color(
                "anomer:N",
                title="Anomeric Configuration",
                scale=alt.Scale(
                    domain=["Alpha (a)", "Beta (b)", "Unknown / Non-glycosidic"],
                    range=["#d97706", "#059669", "#0284c7"],
                ),
            ),
            tooltip=[
                alt.Tooltip("linkage:N", title="Linkage"),
                alt.Tooltip("anomer:N", title="Anomer"),
                alt.Tooltip("count:Q", title="Occurrences", format=","),
            ],
        )
        .properties(title="Top 18 Glycosidic Linkages across KBank", width=380, height=350)
    )

    # 5. Assigned SeroGen LocusArchetype Distribution Bar Chart
    _u_arch, _a_counts = np.unique(glycan_records.archetypes, return_counts=True)
    _a_order = np.argsort(-_a_counts)
    _arch_rows = []
    for _idx in _a_order:
        _arch = LocusArchetype(int(_u_arch[_idx]))
        _arch_rows.append(
            {
                "archetype": _arch.name,
                "category": _arch.category.value,
                "count": int(_a_counts[_idx]),
            }
        )

    archetype_chart = (
        alt.Chart(alt.Data(values=_arch_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "archetype:N",
                sort=alt.EncodingSortField(field="count", order="descending"),
                title="SeroGen LocusArchetype",
            ),
            x=alt.X("count:Q", title="Curated GlycanRecords"),
            color=alt.Color(
                "category:N",
                title="Archetype Category",
                scale=alt.Scale(scheme="set2"),
            ),
            tooltip=[
                alt.Tooltip("archetype:N", title="LocusArchetype"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Records", format=","),
            ],
        )
        .properties(title="Assigned SeroGen LocusArchetype Distribution", width=360, height=320)
    )

    # 6. GeneRole -> Glycan Component Assignments (GeneGlycanAssignments SoA)
    _assigns = glycan_records.assignments
    _role_target_counts: Counter[tuple[int, str]] = Counter()
    _role_aligned_counts: Counter[int] = Counter()
    _role_totals: Counter[int] = Counter()
    _target_names = ["NONE", "REDUCING_END", "LINKAGE", "MODIFIED_MONO", "MONOSACCHARIDE", "ANY"]
    for _r_id, _t_id, _ev_id in zip(_assigns.roles, _assigns.target_types, _assigns.evidences):
        _r_int = int(_r_id)
        _t_str = _target_names[int(_t_id)] if 0 <= int(_t_id) < len(_target_names) else "ANY"
        _role_target_counts[(_r_int, _t_str)] += 1
        _role_totals[_r_int] += 1
        if int(_ev_id) > 0:
            _role_aligned_counts[_r_int] += 1

    _role_rows = []
    for (_r_int, _t_str), _cnt in _role_target_counts.items():
        _role = GeneRole(_r_int)
        _role_rows.append(
            {
                "abbreviation": _role.abbreviation,
                "role_name": _role.name,
                "target_type": _t_str,
                "count": int(_cnt),
                "role_total": int(_role_totals[_r_int]),
                "aligned_count": int(_role_aligned_counts[_r_int]),
            }
        )

    roles_chart = (
        alt.Chart(alt.Data(values=_role_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "abbreviation:N",
                sort=alt.EncodingSortField(field="role_total", op="max", order="descending"),
                title="GeneRole (Abbrev)",
            ),
            x=alt.X("count:Q", title="Assigned Glycan Components"),
            color=alt.Color(
                "target_type:N",
                title="Glycan Target Type",
                scale=alt.Scale(
                    domain=["LINKAGE", "MONOSACCHARIDE", "MODIFIED_MONO", "REDUCING_END"],
                    range=["#0284c7", "#059669", "#e15759", "#d97706"],
                ),
            ),
            tooltip=[
                alt.Tooltip("abbreviation:N", title="Abbrev"),
                alt.Tooltip("role_name:N", title="GeneRole"),
                alt.Tooltip("target_type:N", title="Target Component"),
                alt.Tooltip("count:Q", title="Assignments", format=","),
                alt.Tooltip("aligned_count:Q", title="Locus-Aligned ORFs", format=","),
            ],
        )
        .properties(title="GeneRole -> Glycan Component Assignments across KBank", width=380, height=320)
    )

    _dbs, _db_counts = np.unique(glycan_records.source_dbs, return_counts=True)
    _db_order = np.argsort(-_db_counts)
    source_summary_rows = []
    for _idx in _db_order:
        _db_b = _dbs[_idx]
        _db_name = _db_b.decode("utf-8", errors="replace")
        _sub = glycan_records.where_source_db(_db_b)
        source_summary_rows.append(
            {
                "db": _db_name,
                "records": len(_sub),
                "nodes": len(_sub.glycans.node_types),
                "edges": len(_sub.glycans.edge_types),
                "assigns": _sub.assignments.total_assignments,
                "species": len(np.unique(_sub.organisms)),
                "paired": int(np.sum(_sub.has_locus)),
            }
        )

    _paired = glycan_records.where_has_locus()
    _paired_order = np.argsort(-_paired.alignment_scores) if len(_paired) > 0 else np.empty(0, dtype=np.int64)
    top_paired_rows = []
    for _rank_idx in _paired_order[:25]:
        _rec = _paired[int(_rank_idx)]
        _loc = _rec.locus
        top_paired_rows.append(
            {
                "key": _rec.compound_key.decode("utf-8", errors="replace"),
                "organism": _rec.organism.decode("utf-8", errors="replace"),
                "serotype": _rec.serotype.decode("utf-8", errors="replace") or "—",
                "antigen": _rec.antigen_type.decode("utf-8", errors="replace"),
                "assembly": _rec.assembly_id.decode("utf-8", errors="replace"),
                "archetype": _rec.archetype.name,
                "genes": len(_loc.nodes) if _loc is not None else 0,
                "completeness": float(_loc.completeness) * 100.0 if _loc is not None else 0.0,
                "nodes": len(_rec.glycan.nodes) if _rec.glycan is not None else 0,
                "edges": len(_rec.glycan.edges) if _rec.glycan is not None else 0,
                "score": float(_rec.alignment_score),
            }
        )

    return (
        archetype_chart,
        assembly_cache_count,
        csdb_index_count,
        curation_elapsed_s,
        glycan_records,
        linkage_chart,
        npz_kb,
        npz_parity,
        npz_path,
        parquet_kb,
        parquet_path,
        roles_chart,
        source_chart,
        source_summary_rows,
        species_chart,
        sugar_chart,
        top_paired_rows,
    )


@app.cell
def _(
    assembly_cache_count,
    csdb_index_count,
    curation_elapsed_s,
    glycan_records,
    mo,
    np,
    npz_kb,
    npz_parity,
    parquet_kb,
):
    # Slide 1: Hero Title, Executive Summary & KPI Cards
    _n_total = len(glycan_records)
    _n_loci = int(np.sum(glycan_records.has_locus))
    _n_orgs = len(np.unique(glycan_records.organisms))
    _tot_nodes = len(glycan_records.glycans.node_types)
    _tot_edges = len(glycan_records.glycans.edge_types)
    _tot_assigns = glycan_records.assignments.total_assignments
    _tot_aligned = int(np.sum(glycan_records.assignments.evidences > 0))

    mo.vstack(
        [
            mo.md(
                """
                # `KBank` — Bacterial Surface Glycan & Biosynthetic Locus Curation
                Streaming, parsing, and canonicalizing bacterial surface polysaccharides across **CSDB**, **KEGG Glycan**,
                **ECODAB**, **CERMAV EPS-DB**, **EK3D**, and **GlyTouCan** via [`kbank.io`][kbank.io], paired *de novo* with
                [`SeroGen`][serogen.loci.LocusPredictor] biosynthetic gene clusters and [`BiosyntheticAssigner`][kbank.biosynthesis.BiosyntheticAssigner] alignments.
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Curated GlycanRecords",
                        value=f"{_n_total:,}",
                        caption=f"5 Databases | {csdb_index_count:,} CSDB indexed ({curation_elapsed_s:.1f}s)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Glycan CSR Topology",
                        value=f"{_tot_nodes:,} Nodes",
                        caption=f"{_tot_edges:,} Linkages | {_n_orgs:,} Species",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Paired SeroGen Loci",
                        value=f"{_n_loci:,} Loci",
                        caption=f"{len(glycan_records.loci.node_types):,} BGC ORFs ({assembly_cache_count:,} Assemblies)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Bipartite Assignments",
                        value=f"{_tot_assigns:,}",
                        caption=f"ORF-Aligned: {_tot_aligned:,} | Rule: {_tot_assigns - _tot_aligned:,}",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Zero-Pickle SoA Archive",
                        value=f"{npz_kb:.1f} KB",
                        caption=f"Parquet: {parquet_kb:.1f} KB | Parity: {'PASS' if npz_parity else 'FAIL'}",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.callout(
                mo.md(
                    f"**Exported Production Database (`src/kbank/data/glycans.npz` & `data/kbank.parquet`)**: "
                    f"`{_n_total:,}` canonicalized repeat/oligosaccharide graphs paired with `{_n_loci:,}` segmented "
                    f"biosynthetic gene clusters (`allow_pickle=False`, round-trip bit parity: `{npz_parity}`)."
                ),
                kind="success",
            ),
        ],
        gap=1.2,
    )
    return


@app.cell
def _(alt, mo, source_chart, species_chart):
    # Slide 2: Source Database Coverage & Top 15 Bacterial Species
    mo.vstack(
        [
            mo.md(
                """
                ## 1. Multi-Database Surface Antigen Coverage & Taxonomic Distribution
                Consolidating repeating O-antigens, capsular polysaccharides (CPS/K-antigens), exopolysaccharides (EPS),
                and lipooligosaccharides (LOS) across **CSDB**, **KEGG**, **ECODAB**, **EPS-DB**, and **EK3D**, with
                4-tier NCBI Assembly and nucleotide BGC resolution.
                """
            ),
            alt.hconcat(source_chart, species_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(alt, linkage_chart, mo, sugar_chart):
    # Slide 3: Monosaccharide, Modifier & Glycosidic Linkage Repertoire
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Monosaccharide, Substituent Modifier & Glycosidic Linkage Repertoire
                Distribution of the top 18 monosaccharide/modifier nodes (`Sugars`) and glycosidic linkages (`LinkageType`)
                encoded in KBank's packed 16-bit CSR chemical graph representation.
                """
            ),
            alt.hconcat(sugar_chart, linkage_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(alt, archetype_chart, mo, roles_chart):
    # Slide 4: SeroGen LocusArchetype & Bipartite GeneRole -> Glycan Component Assignments
    mo.vstack(
        [
            mo.md(
                """
                ## 3. `SeroGen` `LocusArchetype` & Bipartite `GeneRole` → Glycan Component Alignment
                Automated assignment of biosynthetic pathway archetypes (`CPS_WZY`, `LPS_WZY`, `LPS_ABC`, `CPS_ABC`, `CPS_SYNTHASE`)
                and stoichiometric enzyme-to-glycan mappings across `LINKAGE`, `MONOSACCHARIDE`, `MODIFIED_MONO`, and `REDUCING_END` targets.
                """
            ),
            alt.hconcat(archetype_chart, roles_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(glycan_records, mo, np, source_summary_rows):
    # Slide 5: Multi-Database Coverage & Pan-Graph Evidence Breakdown
    _ev = glycan_records.assignments.evidences
    _n_pan = int(np.sum(_ev == 3))
    _n_graph = int(np.sum(_ev == 2))
    _n_comp = int(np.sum(_ev == 4))
    _n_rule = int(np.sum(_ev == 1))
    _paired = glycan_records.where_has_locus()
    _mean_score = float(np.mean(_paired.alignment_scores)) if len(_paired) > 0 else 0.0

    _db_md = "\n".join(
        f"| `{r['db']}` | **{r['records']:,}** | {r['nodes']:,} | {r['edges']:,} | {r['assigns']:,} | {r['species']:,} | **{r['paired']:,}** |"
        for r in source_summary_rows
    )
    _db_table = "\n".join(
        [
            "| Source Database | Curated Records | Residue / Modifier Nodes | Glycosidic Linkages | Gene-Glycan Assignments | Unique Species | Paired `SeroGen` Loci |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _db_md,
        ]
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 4. Source Database Synthesis & Bipartite Alignment Evidence Tiers
                Cross-referencing structural databases with `PanGraphConsensusPrior` homology-transfer and Hungarian bipartite graph alignment.
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="PAN_GRAPH_CONSENSUS",
                        value=f"{_n_pan:,}",
                        caption="Multi-record homology & synteny consensus",
                        bordered=True,
                    ),
                    mo.stat(
                        label="GRAPH_ALIGNED",
                        value=f"{_n_graph:,}",
                        caption="Bipartite Hungarian locus-to-glycan match",
                        bordered=True,
                    ),
                    mo.stat(
                        label="LOCUS_COMPLETED",
                        value=f"{_n_comp:,}",
                        caption="Tail modifier ORF completion",
                        bordered=True,
                    ),
                    mo.stat(
                        label="RULE_INFERRED",
                        value=f"{_n_rule:,}",
                        caption="Stoichiometric rule registry fallback",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Mean Locus Alignment",
                        value=f"{_mean_score:.3f}",
                        caption=f"Across {len(_paired):,} paired BGCs",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.md(_db_table),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo, source_summary_rows, top_paired_rows):
    # Slide 6: Reference Tables (Source Databases & Top Paired SeroGen Loci)
    _db_md = "\n".join(
        f"| `{r['db']}` | **{r['records']:,}** | {r['nodes']:,} | {r['edges']:,} | {r['assigns']:,} | {r['species']:,} | **{r['paired']:,}** |"
        for r in source_summary_rows
    )
    _db_table = "\n".join(
        [
            "| Source DB | Curated Records | Residue Nodes | Linkages | Assignments | Species | Paired Loci |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _db_md,
        ]
    )

    _paired_md = "\n".join(
        f"| `{r['key']}` | *{r['organism']}* | `{r['serotype']}` | {r['antigen']} | `{r['assembly']}` | `{r['archetype']}` | {r['genes']} | {r['completeness']:.1f}% | {r['nodes']}/{r['edges']} | **{r['score']:.3f}** |"
        for r in top_paired_rows
    )
    _paired_table = "\n".join(
        [
            "| Record Key | Organism | Serotype | Antigen | NCBI Accession | Archetype | BGC ORFs | Completeness | Nodes/Edges | Alignment Score |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
            _paired_md,
        ]
    )

    mo.vstack(
        [
            mo.md("## 5. Reference Tables: Top De Novo Paired `SeroGen` Loci & Source Coverage"),
            mo.ui.tabs(
                {
                    "Top 25 Paired SeroGen Loci": mo.md(_paired_table),
                    "Source Database Coverage": mo.md(_db_table),
                }
            ),
        ],
        gap=0.8,
    )
    return


if __name__ == "__main__":
    app.run()
