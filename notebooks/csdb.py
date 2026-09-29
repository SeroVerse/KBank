# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx2[http2]==2.10.0",
#     "kaptive @ file:///Users/tsta0015/Programming/Kaptive",
#     "marimo>=0.23.16",
#     "polars==1.43.2",
#     "pyarrow==25.0.1",
#     "pydantic==2.13.4",
#     "scikit-learn==1.9.0",
#     "kbank @ file:///Users/tsta0015/Programming/KaptiveBank",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")

with app.setup:
    from typing import Optional, Dict, Any, Iterable, Union, get_args, get_origin
    import marimo as mo
    import urllib.request
    import itertools
    from pathlib import Path
    from pydantic import BaseModel, Field, ConfigDict, model_validator
    import pyarrow.parquet as pq
    import polars as pl
    import httpx2 as httpx
    import asyncio
    from kaptive.db.manager import DatabaseManager
    from kbank.glycan.glycan import Glycan, Glycans


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Define resuable classes
    """)
    return


@app.class_definition
class CSDBRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    @model_validator(mode='before')
    @classmethod
    def _coerce_bools(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field_name, field_info in cls.model_fields.items():
                alias = field_info.alias or field_name
                if alias in data:
                    val = data[alias]
                    if val in (None, ''):
                        continue
            
                    ann = field_info.annotation
                    origin = get_origin(ann)
                    if origin is Union:
                        args = get_args(ann)
                        types = [t for t in args if t is not type(None)]
                        if types:
                            ann = types[0]
                
                    if ann is bool and isinstance(val, str):
                        v_lower = val.lower().strip()
                        if v_lower.lstrip('-').isdigit():
                            data[alias] = int(v_lower) > 0
                        elif v_lower in ('o', 'false', 'f', 'no', 'n'):
                            data[alias] = False
                        else:
                            data[alias] = True
        return data

    record_id: Optional[str] = Field(None, alias='ID')
    is_elucidated: Optional[bool] = Field(None, alias='TH')
    authors: Optional[str] = Field(None, alias='AU')
    title: Optional[str] = Field(None, alias='TI')
    journal: Optional[str] = Field(None, alias='JN')
    publication_year: Optional[str] = Field(None, alias='PY')
    volume: Optional[str] = Field(None, alias='VL')
    pages: Optional[str] = Field(None, alias='PG')
    references: Optional[str] = Field(None, alias='RL')
    email: Optional[str] = Field(None, alias='EA')
    affiliations: Optional[str] = Field(None, alias='AD')
    abstract: Optional[str] = Field(None, alias='AB')
    structure_linear: Optional[str] = Field(None, alias='ST1')
    structure_linear_orig: Optional[str] = Field(None, alias='ST1ORIG')
    structure_type: Optional[str] = Field(None, alias='ST2')
    polymerization_degree: Optional[str] = Field(None, alias='ST3')
    structure_location: Optional[str] = Field(None, alias='SL')
    aglycon: Optional[str] = Field(None, alias='AG')
    molecular_formula: Optional[str] = Field(None, alias='MF')
    nmr_1h: Optional[str] = Field(None, alias='NMRH')
    nmr_13c: Optional[str] = Field(None, alias='NMRC')
    nmr_solvent: Optional[str] = Field(None, alias='NMRS')
    nmr_temperature: Optional[str] = Field(None, alias='NMRT')
    biological_source: Optional[str] = Field(None, alias='SO')
    taxonomic_domain: Optional[str] = Field(None, alias='KD')
    host_organism: Optional[str] = Field(None, alias='HO')
    organ_tissue: Optional[str] = Field(None, alias='OTI')
    disease: Optional[str] = Field(None, alias='DSS')
    compound_name: Optional[str] = Field(None, alias='NC')
    compound_class: Optional[str] = Field(None, alias='CC')
    methods: Optional[str] = Field(None, alias='MT')
    biological_activity: Optional[str] = Field(None, alias='BA')
    enzymes: Optional[str] = Field(None, alias='EI')
    biosynthetic_genetic_data: Optional[str] = Field(None, alias='BG')
    synthesis_data: Optional[str] = Field(None, alias='SY')
    keywords: Optional[str] = Field(None, alias='KW')
    comments: Optional[str] = Field(None, alias='NT')
    structure_3d_data: Optional[str] = Field(None, alias='3D')
    related_records: Optional[str] = Field(None, alias='RR')
    cross_references: Optional[str] = Field(None, alias='DB')
    ncbi_taxonomy: Optional[str] = Field(None, alias='TAX')
    annotator: Optional[str] = Field(None, alias='U1')
    submission_date: Optional[str] = Field(None, alias='U2')
    refman_id: Optional[str] = Field(None, alias='U3')
    related_refman_ids: Optional[str] = Field(None, alias='U4')
    errors: Optional[str] = Field(None, alias='U5')
    publication_filename: Optional[str] = Field(None, alias='U6')

    @classmethod
    def get_polars_schema(cls) -> Dict[str, Any]:
        schema_definition = {}
        for field_name, field_info in cls.model_fields.items():
            ann = field_info.annotation
            origin = get_origin(ann)
            if origin is Union:
                args = get_args(ann)
                types = [t for t in args if t is not type(None)]
                if types:
                    ann = types[0]
            if ann is bool:
                schema_definition[field_name] = pl.Boolean
            else:
                schema_definition[field_name] = pl.String
        return schema_definition


@app.class_definition
class CSDBStreamParser:
    """
    Takes an iterable stream of text or bytes and yields validated CSDBRecord models.
    """
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        current_record = {}

        for line in self.stream:
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            line = line.strip()

            if line.startswith('#'):
                continue

            if not line:
                if current_record:
                    yield CSDBRecord.model_validate(current_record)
                    current_record = {}
                continue

            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                current_record[key] = value if value != '' else None

        if current_record:
            yield CSDBRecord.model_validate(current_record)


@app.class_definition
class CSDBPipeline:
    """
    Orchestrates the downloading, streaming, and Parquet caching of CSDB datasets.
    """
    def __init__(self, source: str | Path, parquet_path: str | Path, chunk_size: int = 10000):
        self.source = str(source) if isinstance(source, Path) else source
        self.parquet_path = Path(parquet_path)
        self.chunk_size = chunk_size

    def _process_stream(self, stream: Iterable[Union[str, bytes]]):
        schema_definition = CSDBRecord.get_polars_schema()
        writer = None

        parser = CSDBStreamParser(stream)
        for batch in itertools.batched(parser, self.chunk_size):
            valid_records = [model.model_dump() for model in batch]
            df = pl.DataFrame(valid_records, schema=schema_definition)
            arrow_table = df.to_arrow()

            if writer is None:
                writer = pq.ParquetWriter(self.parquet_path, arrow_table.schema)

            writer.write_table(arrow_table)

        if writer is not None:
            writer.close()

    def load(self) -> pl.DataFrame:
        if self.parquet_path.exists():
            return pl.read_parquet(self.parquet_path)

        print(f"Parquet file not found. Fetching from {self.source} and writing to {self.parquet_path}...")

        if self.source.startswith('http://') or self.source.startswith('https://'):
            with urllib.request.urlopen(self.source) as response:
                self._process_stream(response)
        else:
            local_path = Path(self.source)
            with local_path.open('r', encoding='utf-8') as f:
                self._process_stream(f)

        print("Finished writing Parquet file.")
        return pl.read_parquet(self.parquet_path)


@app.class_definition
class StrainSearcher:
    def __init__(self, db='nuccore', species=None, email='tomdstanton@gmail.com', apikey='1a95b525f1c95886f639231e3d517b917f08'):
        self.db = db
        self.species = species
        self.base_params = {
            'db': self.db,
            'retmode': 'json',
            'email': email,
            'api_key': apikey
        }

    async def _get_uid(self, client: httpx.AsyncClient, query: str, semaphore: asyncio.Semaphore) -> dict:
        """Asynchronously search for a top UID, limited by a concurrency semaphore."""
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {**self.base_params, 'term': query, 'retmax': 1}

        async with semaphore:
            for _ in range(3):
                try:
                    response = await client.get(url, params=params, timeout=10.0)
                    response.raise_for_status()
                    idlist = response.json().get('esearchresult', {}).get('idlist', [])
                    return {"query": query, "uid": idlist[0] if idlist else None}
                except httpx.HTTPError:
                    await asyncio.sleep(1) # Asynchronous sleep to avoid blocking the loop
            
        return {"query": query, "uid": None}

    async def fetch_dataframe(self, queries: list[str], max_concurrency: int = 5) -> pl.DataFrame:
        """
        Main async method. Concurrently resolves searches via asyncio, 
        batches summaries, and returns a merged Polars DataFrame.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        # 1. Asynchronously fetch all search results concurrently over HTTP/2
        limits = httpx.Limits(max_connections=max_concurrency)
        async with httpx.AsyncClient(http2=True, limits=limits) as client:
            tasks = [self._get_uid(client, q, semaphore) for q in queries]
            search_results = await asyncio.gather(*tasks)

        # 2. Extract unique UIDs using Polars
        df_queries = pl.DataFrame(search_results).with_columns(pl.col("uid").cast(pl.String))
        unique_uids = df_queries.get_column("uid").drop_nulls().unique().to_list()

        if not unique_uids:
            return df_queries

        # 3. Batch fetch summaries in a single POST request
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        data = {**self.base_params, 'id': ",".join(unique_uids)}

        async with httpx.AsyncClient(http2=True) as client:
            response = await client.post(url, data=data, timeout=30.0)
            response.raise_for_status()
            result_dict = response.json().get('result', {})

        # 4. Parse summary dictionary into Polars and perform join
        uids_returned = result_dict.get('uids', [])
        summary_list = [result_dict[str(uid)] for uid in uids_returned if str(uid) in result_dict]

        df_summaries = pl.DataFrame(summary_list).with_columns(pl.col("uid").cast(pl.String))

        return df_queries.join(df_summaries, on="uid", how="left")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load CSDB data
    """)
    return


@app.cell
def _():
    csdb = CSDBPipeline('https://osf.io/download/2tjrp', 'notebooks/csdb.parquet').load()
    return (csdb,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### _A. baumanni_ K-antigens
    """)
    return


@app.cell
def _(csdb):
    ab_k_csdb = (
        csdb
        .filter(
            pl.col('is_elucidated'),
            pl.col("biological_source").str.contains("Acinetobacter baumannii"),
            pl.col('compound_class').str.contains("K-antigen|CPS"),
            pl.col('structure_type') == "CHEM"
        )
        .pipe(lambda df: df.select([s.name for s in df if s.null_count() < df.height]))
        .with_columns(
            pl.col('compound_name').str.replace_all(r"KL", "K").str.extract_all(r"K[0-9]+").list.unique().alias('a'),
            pl.col('title').str.replace_all(r"KL", "K").str.extract_all(r"K[0-9]+").list.unique().alias('b'),
            pl.col('abstract').str.replace_all(r"KL", "K").str.extract_all(r"K[0-9]+").list.unique().alias('c'),
            pl.col('biological_source').str.extract_all(r"K[0-9]+").list.unique().alias('d'),
        )
        .with_columns(
            pl.concat_list([pl.coalesce(c, pl.lit([])) for c in ['a', 'b', 'c', 'd']])
            .list.eval(pl.element().flatten().mode())
            .list.first()
            .alias("sero")
        )
        .drop(['a', 'b', 'c', 'd'])
        .with_columns(pl.col('sero').str.extract(r"(\d+)").cast(pl.Int32).alias('sero_n'))
        .sort(by='sero_n')
        .filter(pl.col("sero_n").is_not_null())
    )
    return (ab_k_csdb,)


@app.cell
def _(ab_k_csdb):
    (
        ab_k_csdb
        # .filter(pl.col("sero_n").is_duplicated())
        .sort(pl.col("structure_linear").str.len_chars(), descending=True)
        .unique(subset=["sero_n"], keep="first")
    )
    return


@app.cell
def _():
    ab_k = DatabaseManager.load('ab_k')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## _Klebsiella pnuemoniae_ K-antigen assignment
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load Kaptive Data
    """)
    return


@app.cell
def _():
    def load_kpsc_k_kaptive():
        kpsc_k = DatabaseManager.load('kpsc_k')
        return (
             pl.DataFrame({'locus': kpsc_k.loci.ids})
            .with_columns(
                pl.col('locus').str.extract('KL([0-9]+)').alias('locus_n').cast(pl.Int16),
                pl.lit(None).alias("structure")
            )
            .sort("locus_n")
        )
    kpsc_k_data = load_kpsc_k_kaptive()
    kpsc_editor = mo.ui.data_editor(kpsc_k_data)
    kpsc_editor
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Filter Structures
    """)
    return


@app.cell
def _(csdb):
    kpsc_k_csdb = (
        csdb
        .filter(
            pl.col('is_elucidated'),
            pl.col("biological_source").str.contains("Klebsiella"),
            pl.col('compound_class').str.contains("K-antigen|CPS"),
            pl.col('structure_type') == "CHEM"
        )
        .pipe(lambda df: df.select([s.name for s in df if s.null_count() < df.height]))
        .with_columns(
            pl.col('compound_name').str.replace_all(r"KL", "K").str.extract_all(r"K[0-9]+").list.unique().alias('a'),
            pl.col('title').str.replace_all(r"KL", "K").str.extract_all(r"K[0-9]+").list.unique().alias('b'),
            pl.col('abstract').str.replace_all(r"KL", "K").str.extract_all(r"K[0-9]+").list.unique().alias('c'),
            pl.col('biological_source').str.extract_all(r"K[0-9]+").list.unique().alias('d'),
        )
        .with_columns(
            pl.concat_list([pl.coalesce(c, pl.lit([])) for c in ['a', 'b', 'c', 'd']])
            .list.eval(pl.element().flatten().mode())
            .list.first()
            .alias("sero")
        )
        .drop(['a', 'b', 'c', 'd'])
        .with_columns(pl.col('sero').str.extract(r"(\d+)").cast(pl.Int32).alias('sero_n'))
        .sort(by='sero_n')
        .filter(pl.col("sero_n").is_not_null())
    )
    return (kpsc_k_csdb,)


@app.cell
def _(kpsc_k_csdb):
    kpsc_k_glycans =  [Glycan.parse(i, 'CSDB') for i in kpsc_k_csdb.get_column('structure_linear')]
    return


app._unparsable_cell(
    r"""
    sp_k_csdb_linear = set(sp_k_csdb.get_column("structure_linear"))
    sp_k_csdb_encoded = [Glycan.parse(i, 'CSDB') for i in sp_k_csdb_linear]
    sp_k_csdb_batch = Glycans.from_list(sp_k_csdb_encoded)
    """,
    name="_"
)


@app.cell
def _(clusterer, kpsc_k_csdb, kpsc_k_glycan_sim):
    pl.DataFrame({
        'source': kpsc_k_csdb.get_column('biological_source'),
        'sero': kpsc_k_csdb.get_column('biological_source').str.extract(r".*(K[0-9]+)"),
        'clust': clusterer.fit_predict(kpsc_k_glycan_sim),
        'comp': kpsc_k_csdb.get_column('compound_class'),
        'type': kpsc_k_csdb.get_column('structure_type'),
        'comm': kpsc_k_csdb.get_column('comments'),
        'struct': kpsc_k_csdb.get_column('structure_linear')
    }).unique()
    return


@app.cell
async def _(kp_k):
    searcher = StrainSearcher()
    kp_k_strains = await searcher.fetch_dataframe(kp_k.get_column("biological_source").drop_nulls().unique().to_list())
    kp_k_strains
    return


if __name__ == "__main__":
    app.run()
