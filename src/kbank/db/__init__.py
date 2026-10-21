"""Consolidated Structure-of-Arrays (SoA) database containers and curation pipeline for bacterial surface glycans."""

from kbank.db.curation import (
    DEFAULT_LOCI_CACHE_PATH,
    NCBI_EUTILS_URL,
    OGROUP_TO_SALMONELLA_SEROVARS,
    antigen_family,
    build_curated_glycan_records,
    consolidate_curated_records,
    load_predicted_loci_cache,
    merge_record_cluster,
    normalize_raw_record_taxonomy,
    record_rank_key,
    resolve_and_predict_loci,
    save_predicted_loci_cache,
)
from kbank.db.models import GlycanRecord
from kbank.db.records import DEFAULT_DB_PATH, GlycanRecords

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_LOCI_CACHE_PATH",
    "GlycanRecord",
    "GlycanRecords",
    "NCBI_EUTILS_URL",
    "OGROUP_TO_SALMONELLA_SEROVARS",
    "antigen_family",
    "build_curated_glycan_records",
    "consolidate_curated_records",
    "load_predicted_loci_cache",
    "merge_record_cluster",
    "normalize_raw_record_taxonomy",
    "record_rank_key",
    "resolve_and_predict_loci",
    "save_predicted_loci_cache",
]
