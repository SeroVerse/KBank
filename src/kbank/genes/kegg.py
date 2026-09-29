r"""KEGG Orthology (KO) client and reference sequence fetcher.

Provides programmatic retrieval of verified bacterial ortholog sequences for biosynthetic
gene roles ([`KeggOrthologyFetcher`][kbank.loci.kegg.KeggOrthologyFetcher]) to augment rare
or under-represented classes in the BGC training dataset.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from kaptive.core.seq import Sequences

# Canonical KEGG KO mapping to KBank GeneRoles -------------------------------------------------------------------------
KEGG_KO_MAP: dict[str, bytes] = {
    # Ligase & Polymer Assembly
    "K02847": b"LIGASE",  # waaL, rfaL O-antigen ligase
    "K02853": b"POLYMERASE",  # wzy, rfc O-antigen polymerase
    "K13008": b"POLYMERASE",  # wzyB capsular polysaccharide polymerase
    "K02854": b"COPOLYMERASE",  # wzz, cld chain length determinant
    "K04771": b"COPOLYMERASE",  # fepE very long O-antigen chain length regulator
    # Transport Machinery
    "K16693": b"FLIPPASE",  # wzxE O-antigen flippase
    "K16695": b"FLIPPASE",  # wzx flippase
    "K16692": b"KINASE",  # wzc, etk tyrosine-protein kinase
    "K25307": b"PHOSPHATASE",  # wzb, etp low molecular weight phosphotyrosine phosphatase
    "K16690": b"ABC_PERMEASE",  # wzm, kpsM ABC transporter permease
    "K16691": b"ABC_ATPASE",  # wzt, kpsT ABC transporter ATPase
    "K01991": b"TRANSLOCON",  # wza outer membrane polysaccharide export pore
    "K09695": b"CAPSULE_ANCHOR",  # wzi outer membrane capsule attachment protein
    # Initiators
    "K02841": b"INITIATOR",  # wbaP undecaprenyl-phosphate galactose phosphotransferase
    "K03606": b"INITIATOR",  # wcaJ undecaprenyl-phosphate glucose-1-phosphate transferase
    "K02843": b"INITIATOR",  # wecA UDP-GlcNAc:undecaprenyl-phosphate GlcNAc-1-phosphate transferase
    # Modifiers & Precursor Synthesis
    "K00607": b"FORMYLTRANSFERASE",  # fmt, GDP-perosamine N-formyltransferase
    "K16560": b"PYRUVYLTRANSFERASE",  # wcaK pyruvyltransferase
    "K00558": b"METHYLTRANSFERASE",  # polysaccharide O-methyltransferase
    "K01443": b"DEACETYLASE",  # pgaB poly-beta-1,6-GlcNAc deacetylase
    "K01444": b"DEACETYLASE",  # pdaA peptidoglycan deacetylase
    "K01654": b"ALDOLASE",  # neuB sialic acid synthase
    "K05304": b"ALDOLASE",  # pseI pseudaminic acid synthase
    "K01627": b"ALDOLASE",  # kdsA KDO 8-phosphate synthase
    "K00832": b"AMINOTRANSFERASE",  # wecE UDP-4-amino-4,6-dideoxy-GlcNAc aminotransferase
    "K01791": b"EPIMERASE",  # gne UDP-GlcNAc 4-epimerase
    "K01790": b"EPIMERASE",  # rmlC dTDP-4-dehydrorhamnose 3,5-epimerase
    "K02473": b"EPIMERASE",  # mnaA UDP-GlcNAc 2-epimerase
    "K00012": b"OXIDOREDUCTASE",  # ugd UDP-glucose 6-dehydrogenase
    "K00067": b"OXIDOREDUCTASE",  # rmlD dTDP-4-dehydrorhamnose reductase
    "K02377": b"OXIDOREDUCTASE",  # gmd GDP-mannose 4,6-dehydratase
    "K00697": b"TRANSMEMBRANE_SYNTHASE",  # hasA hyaluronan synthase
    "K01795": b"TRANSMEMBRANE_SYNTHASE",  # pgaC poly-beta-1,6-N-acetylglucosamine synthase
    "K00703": b"GLYCOSYLTRANSFERASE",  # capsular polysaccharide glycosyltransferase
    "K01186": b"GLYCOSYLHYDROLASE",  # capsule lyase / depolymerase
    "K07483": b"TRANSPOSASE",  # IS element transposase
    "K04763": b"INTEGRASE",  # site-specific integrase / recombinase
}


class KeggOrthologyFetcher:
    r"""Client to query the KEGG REST API and fetch verified bacterial protein sequences."""

    BASE_URL = "https://rest.kegg.jp"

    def __init__(self, cache_dir: Path | str | None = None, timeout: float = 15.0) -> None:
        r"""Initialise the KEGG fetcher with optional disk cache directory.

        Args:
            cache_dir: Optional path for storing downloaded FASTA files. Defaults to `~/.cache/kbank/kegg`.
            timeout: HTTP request timeout in seconds. Defaults to 15.0.
        """
        if cache_dir is None:
            cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            self.cache_dir = cache_root / "kbank" / "kegg"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)

    def _http_get(self, endpoint: str) -> str:
        r"""Execute an HTTP GET request against the KEGG REST API."""
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        req = Request(url, headers={"User-Agent": "KBank/1.0 (BGC Classifier Research)"})
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except URLError as e:
            raise RuntimeError(f"Failed to query KEGG API at {url}: {e}") from e

    def fetch_ko_gene_ids(self, ko_id: str, max_genes: int = 50) -> list[str]:
        r"""Query linked bacterial gene identifiers for a given KEGG Orthology ID.

        Args:
            ko_id: KEGG KO identifier (e.g. `"K02847"`).
            max_genes: Maximum number of gene IDs to return. Defaults to 50.

        Returns:
            list[str]: Gene identifiers formatted as `<org>:<gene_id>`.
        """
        clean_ko = ko_id.strip().upper()
        if not clean_ko.startswith("K"):
            clean_ko = f"K{clean_ko}"

        raw_text = self._http_get(f"link/genes/{clean_ko}")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        gene_ids: list[str] = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2:
                target_gene = parts[1].strip()
                # Exclude eukaryotic KEGG identifiers (e.g. hsa, mmu, dme, cel)
                if not any(target_gene.startswith(prefix) for prefix in ("hsa:", "mmu:", "rno:", "dme:", "cel:")):
                    gene_ids.append(target_gene)
            if len(gene_ids) >= max_genes:
                break

        return gene_ids

    def fetch_gene_sequence(self, gene_id: str) -> tuple[bytes, bytes] | None:
        r"""Download amino acid sequence for a single KEGG gene identifier.

        Args:
            gene_id: Gene identifier (e.g. `"eco:b3622"`).

        Returns:
            tuple[bytes, bytes] | None: `(gene_id_bytes, aa_sequence_bytes)`, or None if empty.
        """
        raw_fasta = self._http_get(f"get/{gene_id}/aaseq")
        lines = [line.strip() for line in raw_fasta.splitlines() if line.strip()]
        if not lines or not lines[0].startswith(">"):
            return None

        # Extract sequence lines
        seq_parts = [line.encode("ascii", errors="ignore").upper() for line in lines[1:]]
        seq_bytes = b"".join(seq_parts)
        # Strip internal stops or non-standard characters
        seq_bytes = re.sub(rb"[^ARNDCQEGHILKMFPSTWYVBJZX]", b"", seq_bytes)

        if not seq_bytes:
            return None

        return gene_id.encode("utf-8"), seq_bytes

    def get_ko_sequences(self, ko_id: str, max_seqs: int = 50) -> Sequences:
        r"""Retrieve amino acid sequences for a KEGG KO identifier with local disk caching.

        Args:
            ko_id: KEGG KO identifier (e.g. `"K02847"`).
            max_seqs: Maximum number of sequences to retrieve. Defaults to 50.

        Returns:
            Sequences: Collection of fetched amino acid sequences.
        """
        clean_ko = ko_id.strip().upper()
        cache_file = self.cache_dir / f"{clean_ko}_{max_seqs}.fasta"

        # Check local cache first
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            return Sequences.from_fasta(cache_file.read_bytes())

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Download from KEGG REST API
        gene_ids = self.fetch_ko_gene_ids(clean_ko, max_genes=max_seqs)
        records: list[tuple[bytes, bytes]] = []

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self.fetch_gene_sequence, gid): gid for gid in gene_ids}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res is not None:
                        records.append(res)
                except Exception:
                    continue

        if not records:
            return Sequences.empty()

        # Write to local cache file for offline reusability
        with open(cache_file, "wb") as f:
            for gid_bytes, seq_bytes in records:
                f.write(b">" + gid_bytes + b"\n" + seq_bytes + b"\n")

        return Sequences.from_fasta(cache_file.read_bytes())
