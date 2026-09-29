# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pyarrow>=18.0.0",
# ]
# ///

import itertools
from pathlib import Path
import urllib.request
import pyarrow as pa
import pyarrow.parquet as pq

# CSDB tag to snake_case column mapping
TAG_MAP: dict[str, str] = {
    "ID": "record_id",
    "TH": "is_elucidated",
    "AU": "authors",
    "TI": "title",
    "JN": "journal",
    "PY": "publication_year",
    "VL": "volume",
    "PG": "pages",
    "RL": "references",
    "EA": "email",
    "AD": "affiliations",
    "AB": "abstract",
    "ST1": "structure_linear",
    "ST1ORIG": "structure_linear_orig",
    "ST2": "structure_type",
    "ST3": "polymerization_degree",
    "SL": "structure_location",
    "AG": "aglycon",
    "MF": "molecular_formula",
    "NMRH": "nmr_1h",
    "NMRC": "nmr_13c",
    "NMRS": "nmr_solvent",
    "NMRT": "nmr_temperature",
    "SO": "biological_source",
    "KD": "taxonomic_domain",
    "HO": "host_organism",
    "OTI": "organ_tissue",
    "DSS": "disease",
    "NC": "compound_name",
    "CC": "compound_class",
    "MT": "methods",
    "BA": "biological_activity",
    "EI": "enzymes",
    "BG": "biosynthetic_genetic_data",
    "SY": "synthesis_data",
    "KW": "keywords",
    "NT": "comments",
    "3D": "structure_3d_data",
    "RR": "related_records",
    "DB": "cross_references",
    "TAX": "ncbi_taxonomy",
    "U1": "annotator",
    "U2": "submission_date",
    "U3": "refman_id",
    "U4": "related_refman_ids",
    "U5": "errors",
    "U6": "publication_filename",
}

SCHEMA = pa.schema([
    pa.field(
        col,
        pa.bool_() if col == "is_elucidated" else pa.string()
    )
    for col in TAG_MAP.values()
])


def _coerce_bool(val: str | None) -> bool | None:
    if not val:
        return None
    v = val.lower().strip()
    if v.lstrip("-").isdigit():
        return int(v) > 0
    return v not in ("0", "o", "false", "f", "no", "n")


class CSDBStreamParser:
    """Streams and maps CSDB entries directly into schema-ready dicts."""

    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        current_record: dict[str, str | bool | None] = {}

        for line in self.stream:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")

            line = line.strip()
            if not line or line.startswith("#"):
                if not line and current_record:
                    yield current_record
                    current_record = {}
                continue

            parts = line.split(":", 1)
            if len(parts) == 2:
                raw_key, raw_val = parts[0].strip(), parts[1].strip()
                field_name = TAG_MAP.get(raw_key)
                if not field_name:
                    continue

                if raw_val == "":
                    current_record[field_name] = None
                elif field_name == "is_elucidated":
                    current_record[field_name] = _coerce_bool(raw_val)
                else:
                    current_record[field_name] = raw_val

        if current_record:
            yield current_record


def main():
    source = "https://osf.io/download/2tjrp"
    parquet_path = Path("notebooks") / "csdb.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 10_000

    with urllib.request.urlopen(source) as response:
        parser = CSDBStreamParser(response)

        with pq.ParquetWriter(parquet_path, SCHEMA) as writer:
            for batch in itertools.batched(parser, chunk_size):
                # from_pylist fills missing keys with nulls based on SCHEMA
                table = pa.Table.from_pylist(batch, schema=SCHEMA)
                writer.write_table(table)


if __name__ == "__main__":
    main()