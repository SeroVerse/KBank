r"""Asynchronous multi-source data fetcher for bacterial biosynthetic gene roles.

Ingests non-redundant reference proteins from UniProtKB Swiss-Prot and NCBI Identical
Protein Groups (IPG) using high-throughput connection pooling (aiohttp) and semaphore
rate limiting, with local SHA-256 disk caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiohttp
from kaptive.core.seq import Sequences

from kbank.genes.roles import GeneRoles

# ----------------------------------------------------------------------------------------------------------------------
# Role Query Registry for UniProtKB and NCBI IPG (Roles 0 to 28)
# ----------------------------------------------------------------------------------------------------------------------

ROLE_QUERIES: dict[int, dict[str, str]] = {
    # 0: FLIPPASE
    0: {
        "uniprot": '(gene:wzx OR protein_name:"O-antigen flippase" OR protein_name:"capsular polysaccharide flippase") AND taxonomy_id:2',
        "ncbi_ipg": '("O-antigen flippase"[Product] OR wzx[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 1: POLYMERASE
    1: {
        "uniprot": '(gene:wzy OR protein_name:"O-antigen polymerase" OR protein_name:"capsular polysaccharide polymerase") AND taxonomy_id:2',
        "ncbi_ipg": '("O-antigen polymerase"[Product] OR wzy[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 2: COPOLYMERASE
    2: {
        "uniprot": '(gene:wzz OR gene:fepE OR gene_exact:cld OR protein_name:"chain length determinant" OR protein_name:"polysaccharide copolymerase") AND taxonomy_id:2',
        "ncbi_ipg": '("chain length determinant"[Product] OR wzz[Gene Name] OR fepE[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 3: KINASE
    3: {
        "uniprot": '(gene:wzc OR gene:ptk OR gene:etk OR protein_name:"tyrosine-protein kinase Wzc") AND taxonomy_id:2',
        "ncbi_ipg": '("tyrosine-protein kinase Wzc"[Product] OR wzc[Gene Name] OR ptk[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 4: PHOSPHATASE
    4: {
        "uniprot": '(gene:wzb OR gene:ptp OR gene:etp OR protein_name:"phosphotyrosine protein phosphatase") AND taxonomy_id:2',
        "ncbi_ipg": '("low molecular weight phosphotyrosine protein phosphatase"[Product] OR wzb[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 5: ABC_PERMEASE
    5: {
        "uniprot": '(gene:wzm OR gene:kpsM OR protein_name:"capsule export system permease") AND taxonomy_id:2',
        "ncbi_ipg": '("capsule export system permease"[Product] OR wzm[Gene Name] OR kpsM[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 6: ABC_ATPASE
    6: {
        "uniprot": '(gene:wzt OR gene:kpsT OR protein_name:"capsule export system ATP-binding") AND taxonomy_id:2',
        "ncbi_ipg": '("capsule export system ATP-binding"[Product] OR wzt[Gene Name] OR kpsT[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 7: LIGASE
    7: {
        "uniprot": '(gene:waaL OR gene:rfaL OR protein_name:"O-antigen ligase") AND taxonomy_id:2',
        "ncbi_ipg": '("O-antigen ligase"[Product] OR waaL[Gene Name] OR rfaL[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 8: TRANSLOCON
    8: {
        "uniprot": '(gene:wza OR gene:kpsD OR gene:kpsE OR protein_name:"polysaccharide export protein Wza") AND taxonomy_id:2',
        "ncbi_ipg": '("outer membrane polysaccharide export"[Product] OR wza[Gene Name] OR kpsD[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 9: CAPSULE_ANCHOR
    9: {
        "uniprot": '(gene:wzi OR gene:kpsC OR protein_name:"capsule attachment protein" OR protein_name:"capsule assembly protein Wzi" OR protein_name:"capsular polysaccharide assembly protein Wzi") AND taxonomy_id:2',
        "ncbi_ipg": '("capsule attachment protein"[Product] OR "capsule assembly protein Wzi"[Product] OR wzi[Gene Name] OR kpsC[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 10: PYRUVYLTRANSFERASE
    10: {
        "uniprot": '(gene:wcaK OR protein_name:pyruvyltransferase) AND taxonomy_id:2',
        "ncbi_ipg": '(pyruvyltransferase[Product] OR wcaK[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 11: METHYLTRANSFERASE
    11: {
        "uniprot": '(protein_name:"polysaccharide O-methyltransferase" OR protein_name:"sugar O-methyltransferase") AND taxonomy_id:2',
        "ncbi_ipg": '("polysaccharide O-methyltransferase"[Product] OR "sugar O-methyltransferase"[Product]) AND "Bacteria"[Organism]',
    },
    # 12: ACETYLTRANSFERASE
    12: {
        "uniprot": '(gene:wecD OR gene:oatA OR gene:atr OR protein_name:"O-acetyltransferase") AND taxonomy_id:2',
        "ncbi_ipg": '("O-acetyltransferase"[Product] OR wecD[Gene Name] OR oatA[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 13: PHOSPHOTRANSFERASE
    13: {
        "uniprot": '(gene:licD OR gene:pgt OR gene:wcfF OR protein_name:"phosphocholine transferase") AND taxonomy_id:2',
        "ncbi_ipg": '("phosphocholine transferase"[Product] OR licD[Gene Name] OR pgt[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 14: FORMYLTRANSFERASE
    14: {
        "uniprot": '(gene:wbqD OR gene:fmt OR protein_name:"N-formyltransferase" OR protein_name:"formyltransferase") AND taxonomy_id:2',
        "ncbi_ipg": '("N-formyltransferase"[Product] OR "formyltransferase"[Product] OR wbqD[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 15: ALANINE_TRANSFERASE
    15: {
        "uniprot": '(gene:alt1 OR gene_exact:dltA OR gene:aat OR protein_name:"alanine--poly(phosphoribitol) ligase" OR protein_name:"D-alanine transferase" OR protein_name:"D-alanyl-lipoteichoic acid") AND taxonomy_id:2',
        "ncbi_ipg": '("alanine--poly(phosphoribitol) ligase"[Product] OR "D-alanine transferase"[Product] OR "D-alanyl-lipoteichoic acid biosynthesis"[Product] OR alt1[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 16: AMINOTRANSFERASE
    16: {
        "uniprot": '(gene:wecE OR gene:degT OR gene:pseC OR gene:rhnb OR gene:rhnc OR protein_name:"transaminase") AND taxonomy_id:2',
        "ncbi_ipg": '("dTDP-4-amino-4,6-dideoxygalactose transaminase"[Product] OR wecE[Gene Name] OR pseC[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 17: ALDOLASE
    17: {
        "uniprot": '(gene:neuB OR gene:pseI OR gene:kdsA OR protein_name:"N-acetylneuraminate synthase") AND taxonomy_id:2',
        "ncbi_ipg": '("N-acetylneuraminate synthase"[Product] OR neuB[Gene Name] OR kdsA[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 18: DEACETYLASE
    18: {
        "uniprot": '(gene:pgaB OR gene:pdaA OR protein_name:"poly-beta-1,6-N-acetyl-D-glucosamine N-deacetylase") AND taxonomy_id:2',
        "ncbi_ipg": '("poly-beta-1,6-N-acetyl-D-glucosamine N-deacetylase"[Product] OR pgaB[Gene Name] OR pdaA[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 19: EPIMERASE
    19: {
        "uniprot": '(gene:gne OR gene:rmlC OR gene:galE OR protein_name:"UDP-glucose 4-epimerase") AND taxonomy_id:2',
        "ncbi_ipg": '("UDP-glucose 4-epimerase"[Product] OR galE[Gene Name] OR rmlC[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 20: OXIDOREDUCTASE
    20: {
        "uniprot": '(gene:rmlB OR gene:rmlD OR gene:ugd OR protein_name:"UDP-glucose 6-dehydrogenase" OR protein_name:"dTDP-glucose 4,6-dehydratase") AND taxonomy_id:2',
        "ncbi_ipg": '("UDP-glucose 6-dehydrogenase"[Product] OR "dTDP-glucose 4,6-dehydratase"[Product] OR "dTDP-4-dehydrorhamnose reductase"[Product] OR rmlB[Gene Name] OR ugd[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 21: TRANSMEMBRANE_SYNTHASE
    21: {
        "uniprot": '(gene:hasA OR gene:pgaC OR (gene:bcsA AND protein_name:"catalytic") OR protein_name:"hyaluronan synthase" OR protein_name:"cellulose synthase catalytic subunit") AND taxonomy_id:2',
        "ncbi_ipg": '("hyaluronan synthase"[Product] OR "cellulose synthase catalytic subunit"[Product] OR "cellulose synthase subunit A"[Product] OR hasA[Gene Name] OR pgaC[Gene Name] OR bcsA[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 22: NUCLEOTIDYLTRANSFERASE
    22: {
        "uniprot": '(gene:rmlA OR gene:galU OR gene:glmM OR protein_name:"glucose-1-phosphate thymidylyltransferase") AND taxonomy_id:2',
        "ncbi_ipg": '("glucose-1-phosphate thymidylyltransferase"[Product] OR rmlA[Gene Name] OR galU[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 23: NUCLEOTIDASE
    23: {
        "uniprot": '(gene:gmm OR gene:psaD OR gene:lgaF OR protein_name:"GDP-mannose mannosyl hydrolase" OR protein_name:"sugar-nucleotide hydrolase") AND taxonomy_id:2',
        "ncbi_ipg": '("GDP-mannose mannosyl hydrolase"[Product] OR "GDP-mannose hydrolase"[Product] OR gmm[Gene Name] OR psaD[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 24: INITIATOR
    24: {
        "uniprot": '(gene:wbaP OR gene:wcaJ OR gene:wecA OR protein_name:"undecaprenyl-phosphate galactose phosphotransferase") AND taxonomy_id:2',
        "ncbi_ipg": '("undecaprenyl-phosphate galactose phosphotransferase"[Product] OR wbaP[Gene Name] OR wcaJ[Gene Name] OR wecA[Gene Name]) AND "Bacteria"[Organism]',
    },
    # 25: GLYCOSYLHYDROLASE
    25: {
        "uniprot": '(protein_name:"alginate lyase" OR protein_name:"polysaccharide lyase" OR protein_name:"depolymerase") AND taxonomy_id:2',
        "ncbi_ipg": '("polysaccharide lyase"[Product] OR "alginate lyase"[Product]) AND "Bacteria"[Organism]',
    },
    # 26: GLYCOSYLTRANSFERASE
    26: {
        "uniprot": 'protein_name:"glycosyltransferase" AND taxonomy_id:2',
        "ncbi_ipg": '(glycosyltransferase[Product]) AND "Bacteria"[Organism]',
    },
    # 27: TRANSPOSASE
    27: {
        "uniprot": '(protein_name:transposase NOT protein_name:integrase NOT protein_name:recombinase) AND reviewed:true AND taxonomy_id:2',
        "ncbi_ipg": '("IS family transposase"[Product] OR "insertion sequence transposase"[Product] OR (transposase[Product] NOT integrase[Product])) AND "Bacteria"[Organism]',
    },
    # 28: INTEGRASE
    28: {
        "uniprot": '(protein_name:integrase OR protein_name:"site-specific recombinase" NOT protein_name:transposase) AND reviewed:true AND taxonomy_id:2',
        "ncbi_ipg": '("site-specific integrase"[Product] OR "phage integrase"[Product] OR "tyrosine recombinase XerC"[Product] OR "tyrosine recombinase XerD"[Product]) AND "Bacteria"[Organism]',
    },
}


# ----------------------------------------------------------------------------------------------------------------------
# MultiSourceFetcher Engine
# ----------------------------------------------------------------------------------------------------------------------

class MultiSourceFetcher:
    r"""Asynchronous multi-source sequence fetcher with disk caching and rate limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        concurrency_limit: int = 10,
        timeout_seconds: float = 30.0,
    ) -> None:
        # Resolve NCBI API key
        if api_key is None:
            api_key = os.environ.get("NCBI_API_KEY")
            if not api_key:
                env_path = Path.home() / ".env"
                if env_path.is_file():
                    for line in env_path.read_text().splitlines():
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            if k.strip() == "NCBI_API_KEY":
                                api_key = v.strip()
                                break
        self.api_key = api_key

        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "kbank" / "fetchers"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.concurrency_limit = concurrency_limit
        self.timeout_seconds = timeout_seconds
        self.roles = GeneRoles()

    def _cache_path(self, key_str: str, ext: str = ".dat") -> Path:
        """Computes deterministic cache path based on SHA-256 digest."""
        h = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}{ext}"

    async def _http_get(
        self,
        session: aiohttp.ClientSession,
        url: str,
        semaphore: asyncio.Semaphore,
        max_retries: int = 3,
    ) -> bytes:
        """Performs semaphore-limited GET request with exponential backoff on rate limits."""
        cache_file = self._cache_path(url, ext=".dat")
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            return cache_file.read_bytes()

        backoff = 1.0
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            cache_file.write_bytes(content)
                            return content
                        elif resp.status in (429, 500, 502, 503, 504):
                            await asyncio.sleep(backoff)
                            backoff *= 2.0
                        else:
                            text = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status} for {url}: {text[:200]}")
                except (TimeoutError, aiohttp.ClientError) as err:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts: {err}") from err
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

        raise RuntimeError(f"Exhausted retries fetching {url}")

    async def fetch_uniprot(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int = 60,
    ) -> list[tuple[str, str, bytes]]:
        """Queries UniProtKB REST API and parses returned FASTA sequences."""
        encoded_q = urllib.parse.quote(query)
        url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_q}&format=fasta&size={limit}"
        sem = asyncio.Semaphore(5)
        raw_fasta = await self._http_get(session, url, sem)

        results: list[tuple[str, str, bytes]] = []
        if not raw_fasta or not raw_fasta.strip():
            return results

        try:
            seq_objs = Sequences.from_fasta(raw_fasta)
            for i in range(len(seq_objs)):
                sid = seq_objs.ids[i].decode("ascii", errors="replace").split()[0]
                seq_b = bytes(seq_objs[i].seq)
                if len(seq_b) >= 30:
                    results.append((sid, f"uniprot_{sid}", seq_b))
        except Exception:
            # Fallback manual parse if needed
            pass

        return results

    async def fetch_ncbi_ipg(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int = 60,
    ) -> list[tuple[str, str, bytes]]:
        """Queries NCBI IPG db and batch downloads RefSeq WP_... FASTA sequences."""
        sem = asyncio.Semaphore(self.concurrency_limit)
        api_param = f"&api_key={self.api_key}" if self.api_key else ""

        # 1. ESearch IPG
        encoded_q = urllib.parse.quote(query)
        esearch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=ipg&term={encoded_q}&retmax={limit}&retmode=xml{api_param}"
        )
        search_xml = await self._http_get(session, esearch_url, sem)
        try:
            root = ET.fromstring(search_xml)
            id_list = [e.text for e in root.findall(".//IdList/Id") if e.text]
        except Exception:
            id_list = []

        if not id_list:
            return []

        # 2. ESummary IPG to extract non-redundant RefSeq WP_ accessions
        summary_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            f"db=ipg&id={','.join(id_list)}&retmode=xml{api_param}"
        )
        sum_xml = await self._http_get(session, summary_url, sem)
        wp_accs: list[str] = []
        try:
            sum_root = ET.fromstring(sum_xml)
            for doc in sum_root.findall(".//DocumentSummary"):
                acc = doc.findtext("Accession")
                if acc and (acc.startswith("WP_") or acc.startswith("NP_") or re.match(r"^[A-Z]{2,}_", acc)):
                    wp_accs.append(acc)
        except Exception:
            pass

        # Deduplicate accessions while preserving order
        unique_accs = list(dict.fromkeys(wp_accs))
        if not unique_accs:
            return []

        # 3. Batch EFetch in protein db
        results: list[tuple[str, str, bytes]] = []
        chunk_size = 50
        for chunk_idx in range(0, len(unique_accs), chunk_size):
            chunk = unique_accs[chunk_idx : chunk_idx + chunk_size]
            efetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                f"db=protein&id={','.join(chunk)}&rettype=fasta&retmode=text{api_param}"
            )
            fasta_bytes = await self._http_get(session, efetch_url, sem)
            if fasta_bytes and fasta_bytes.strip():
                try:
                    seq_objs = Sequences.from_fasta(fasta_bytes)
                    for i in range(len(seq_objs)):
                        sid = seq_objs.ids[i].decode("ascii", errors="replace").split()[0]
                        seq_b = bytes(seq_objs[i].seq)
                        if len(seq_b) >= 30:
                            results.append((sid, f"ncbi_ipg_{sid}", seq_b))
                except Exception:
                    pass

        return results

    async def fetch_for_role(
        self,
        session: aiohttp.ClientSession,
        role_idx: int,
        target_count: int = 60,
    ) -> list[dict[str, Any]]:
        """Fetches non-redundant sequences for a given role across UniProtKB and NCBI IPG."""
        if role_idx not in ROLE_QUERIES:
            return []

        queries = ROLE_QUERIES[role_idx]
        role_name = self.roles.names[role_idx].decode("ascii")
        category = self.roles.categories[role_idx].decode("ascii")

        records: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()

        # 1. Fetch from UniProtKB Swiss-Prot
        uniprot_q = queries.get("uniprot")
        if uniprot_q:
            try:
                up_results = await self.fetch_uniprot(session, uniprot_q, limit=target_count)
                for sid, lid, seq_b in up_results:
                    h = hashlib.sha256(seq_b).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        records.append({
                            "gene_id": f"uniprot_{sid}",
                            "locus_id": f"uniprot_{role_name}_{sid}",
                            "source": "uniprot",
                            "role_index": role_idx,
                            "role_name": role_name,
                            "category": category,
                            "aa_seq": seq_b,
                            "length": len(seq_b),
                        })
            except Exception as e:
                print(f"Warning: UniProt fetch failed for role {role_name}: {e}")

        # 2. Fetch from NCBI IPG if needed to reach target_count
        if len(records) < target_count:
            ncbi_q = queries.get("ncbi_ipg")
            if ncbi_q:
                try:
                    needed = target_count - len(records) + 20
                    ipg_results = await self.fetch_ncbi_ipg(session, ncbi_q, limit=max(needed, 50))
                    for sid, lid, seq_b in ipg_results:
                        h = hashlib.sha256(seq_b).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            records.append({
                                "gene_id": f"ncbi_{sid}",
                                "locus_id": f"ncbi_{role_name}_{sid}",
                                "source": "ncbi_ipg",
                                "role_index": role_idx,
                                "role_name": role_name,
                                "category": category,
                                "aa_seq": seq_b,
                                "length": len(seq_b),
                            })
                except Exception as e:
                    print(f"Warning: NCBI IPG fetch failed for role {role_name}: {e}")

        return records

    async def async_fetch_roles(
        self,
        role_indices: Sequence[int],
        target_count: int = 60,
    ) -> dict[int, list[dict[str, Any]]]:
        """Concurrently fetches sequences for multiple roles."""
        conn = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
        async with aiohttp.ClientSession(
            connector=conn,
            headers={"User-Agent": "KBank-Curation/1.0 (tom.stanton@monash.edu)"},
        ) as session:
            tasks = [self.fetch_for_role(session, r_idx, target_count=target_count) for r_idx in role_indices]
            results = await asyncio.gather(*tasks)
            return dict(zip(role_indices, results))

    def fetch_roles(
        self,
        role_indices: Sequence[int],
        target_count: int = 60,
    ) -> dict[int, list[dict[str, Any]]]:
        """Synchronous entry point for fetching external sequences."""
        return asyncio.run(self.async_fetch_roles(role_indices, target_count=target_count))
