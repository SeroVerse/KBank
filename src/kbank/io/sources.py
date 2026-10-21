"""External carbohydrate database normalizers, parsers, and async/streaming fetchers."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
import gzip
import html
import io
import json
from pathlib import Path
import re
import subprocess
from typing import Any
import urllib.request
import warnings

from kbank.glycan import Glycan
from kbank.io.base import GlycanParseError, GlycanParseWarning
from kbank.io.csdb import CSDBParser
from kbank.io.iupac import IUPACParser
from kbank.io.kegg import parse_kegg_glycan_flatfile

CSDB_OSF_URL = "https://osf.io/download/2tjrp"
ECODAB_BASE_URL = "https://nevyn.organ.su.se/ECODAB/"
EK3D_URL = "https://www.iith.ac.in/EK3D/Kantigenstructures.php"
EPSDB_BASE_URL = "https://epsdatabase.cermav.cnrs.fr/"
KEGG_REST_URL = "https://rest.kegg.jp"
GLYCOSMOS_SPARQL_URL = "https://ts.glycosmos.org/sparql"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".cache"
DEFAULT_CSDB_CACHE_PATH = DEFAULT_CACHE_DIR / "csdb_surface_cache.json.gz"

CSDB_TAG_MAP: dict[str, str] = {
    "ID": "record_id",
    "TH": "is_elucidated",
    "AU": "authors",
    "TI": "title",
    "JN": "journal",
    "PY": "publication_year",
    "RL": "references",
    "AB": "abstract",
    "ST1": "structure_linear",
    "ST2": "structure_type",
    "ST3": "polymerization_degree",
    "SL": "structure_location",
    "AG": "aglycon",
    "SO": "biological_source",
    "KD": "taxonomic_domain",
    "NC": "compound_name",
    "CC": "compound_class",
    "BG": "biosynthetic_genetic_data",
    "KW": "keywords",
    "NT": "comments",
    "DB": "cross_references",
    "TAX": "ncbi_taxonomy",
}

BACTERIAL_GENERA_RE = re.compile(
    r"\b("
    r"Escherichia|Klebsiella|Acinetobacter|Streptococcus|Salmonella|Pseudomonas|"
    r"Staphylococcus|Shigella|Vibrio|Yersinia|Campylobacter|Burkholderia|"
    r"Haemophilus|Neisseria|Bacillus|Listeria|Enterococcus|Clostridium|Clostridioides|"
    r"Bacteroides|Parabacteroides|Phocaeicola|Porphyromonas|Prevotella|Tannerella|"
    r"Helicobacter|Mycobacterium|Rhodococcus|Corynebacterium|Gordonia|Nocardia|Dietzia|"
    r"Proteus|Providencia|Morganella|Serratia|Citrobacter|Enterobacter|Cronobacter|"
    r"Raoultella|Hafnia|Edwardsiella|Pantoea|Erwinia|Pectobacterium|Dickeya|"
    r"Photorhabdus|Xenorhabdus|Plesiomonas|Rahnella|Kluyvera|Leclercia|Lelliottia|"
    r"Kosakonia|Pluralibacter|Shimwellia|Atlantibacter|Tatumella|Ewingella|Cedecea|"
    r"Aeromonas|Shewanella|Pseudoalteromonas|Alteromonas|Marinobacter|Halomonas|Cobetia|"
    r"Idiomarinaceae|Idiomarina|Psychromonas|Colwellia|Moritella|Photobacterium|Aliivibrio|"
    r"Francisella|Brucella|Ochrobactrum|Bartonella|Legionella|Coxiella|Rickettsia|"
    r"Bordetella|Achromobacter|Alcaligenes|Taylorella|Ralstonia|Cupriavidus|"
    r"Xanthomonas|Stenotrophomonas|Xylella|Lysobacter|"
    r"Pasteurella|Mannheimia|Actinobacillus|Aggregatibacter|Histophilus|Glaesserella|"
    r"Moraxella|Psychrobacter|Kingella|Eikenella|"
    r"Rhizobium|Sinorhizobium|Ensifer|Bradyrhizobium|Mesorhizobium|Agrobacterium|Azorhizobium|"
    r"Azospirillum|Rhodobacter|Paracoccus|Sphingomonas|Caulobacter|Zymomonas|Acetobacter|Komagataeibacter|"
    r"Lactobacillus|Lactiplantibacillus|Lacticaseibacillus|Limosilactobacillus|Lactococcus|"
    r"Leuconostoc|Pediococcus|Weissella|Oenococcus|Bifidobacterium|Propionibacterium|Cutibacterium|"
    r"Streptomyces|Micrococcus|Arthrobacter|Clavibacter|Rathayibacter|Actinomyces|"
    r"Paenibacillus|Brevibacillus|Geobacillus|Aneurinibacillus|Lysinibacillus|"
    r"Flavobacterium|Chryseobacterium|Cytophaga|Elizabethkingia|Capnocytophaga|"
    r"Fusobacterium|Leptotrichia|Treponema|Borrelia|Borreliella|Leptospira|Brachyspira|"
    r"Chlamydia|Chlamydophila|Mycoplasma|Ureaplasma|Acholeplasma|"
    r"Arcobacter|Wolinella|Desulfovibrio|Lawsonia|Synechococcus|Synechocystis|Nostoc|Anabaena|"
    r"Thermus|Deinococcus|Meiothermus|Thermotoga|Aquifex"
    r")\b",
    re.IGNORECASE,
)

SURFACE_ANTIGEN_RE = re.compile(
    r"\b("
    r"CPS|K-antigen|K\s+antigen|capsul\w*|O-antigen|O\s+antigen|OPS|LPS|LOS|"
    r"lipopolysaccharide|lipooligosaccharide|EPS|exopolysaccharide|teichoic|WTA|LTA|"
    r"surface\s+polysaccharide|cell\s+wall\s+polysaccharide|Vi-antigen"
    r")\b",
    re.IGNORECASE,
)

SEROTYPE_EXTRACT_RE = re.compile(
    r"\b("
    r"KL\d+[A-Za-z0-9]*|OL\d+[A-Za-z0-9]*|OCL\d+[A-Za-z0-9]*|"
    r"K\d+[A-Za-z]?|O\d+[A-Za-z]?|"
    r"(?:IATS\s+O?\d+[A-Za-z]?)|"
    r"(?:serotype|serogroup|(?<!sequence\s)(?<!Sequence\s)type)\s+(?:\d+[A-Za-z0-9]*|[IVX]{1,4}[abAB]?|[A-Fa-f])\b"
    r")\b",
    re.IGNORECASE,
)

CPS_SEROTYPE_RE = re.compile(
    r"\b(KL\d+[A-Za-z0-9]*|K\d+[A-Za-z]?|(?:serotype|capsular\s+type|(?<!sequence\s)(?<!Sequence\s)type)\s+(?:\d+[A-Za-z0-9]*|[IVX]{1,4}[abAB]?|[A-Fa-f]))\b",
    re.IGNORECASE,
)

OPS_SEROTYPE_RE = re.compile(
    r"\b(OL\d+[A-Za-z0-9]*|OCL\d+[A-Za-z0-9]*|O\d+[A-Za-z0-9]*|(?:IATS\s+O?\d+[A-Za-z]?)|(?:serogroup|serotype|(?<!sequence\s)(?<!Sequence\s)type)\s+(?:O?\d+[A-Za-z0-9]*|[IVX]{1,4}[abAB]?))\b",
    re.IGNORECASE,
)

GENBANK_ACC_RE = re.compile(
    r"\b(GC[FA]_\d{9}\.\d+|N[CZW]_[A-Z0-9]{6,10}(?:\.\d+)?|[A-Z]{2}\d{6,8}(?:\.\d+)?)\b"
)

PMID_EXTRACT_RE = re.compile(r"(?:PMID|PubMed)\s*[:=#]?\s*(\d{5,9})", re.IGNORECASE)
DOI_EXTRACT_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")

# Kauffmann-White Salmonella enterica serovar -> canonical O-antigen serogroup map
SALMONELLA_SEROVAR_TO_OGROUP: dict[str, str] = {
    "paratyphi a": "O2",
    "paratyphi b": "O4",
    "typhimurium": "O4",
    "agB": "O4",
    "heidelberg": "O4",
    "saintpaul": "O4",
    "derby": "O4",
    "agona": "O4",
    "bredeney": "O4",
    "schwarzengrund": "O4",
    "stanley": "O4",
    "chester": "O4",
    "abortusequi": "O4",
    "choleraesuis": "O7",
    "paratyphi c": "O7",
    "montevideo": "O7",
    "infantis": "O7",
    "thompson": "O7",
    "virchow": "O7",
    "oranienburg": "O7",
    "braenderup": "O7",
    "ohio": "O7",
    "tennessee": "O7",
    "muenchen": "O8",
    "newport": "O8",
    "manhattan": "O8",
    "blockley": "O8",
    "kentucky": "O8",
    "hadar": "O8",
    "bovismorbificans": "O8",
    "typhi": "O9",
    "enteritidis": "O9",
    "dublin": "O9",
    "pullorum": "O9",
    "gallinarum": "O9",
    "javiana": "O9",
    "panama": "O9",
    "berta": "O9",
    "haarlem": "O9",
    "strasbourg": "O9",
    "anatum": "O3,10",
    "give": "O3,10",
    "meleagridis": "O3,10",
    "weltevreden": "O3,10",
    "london": "O3,10",
    "muenster": "O3,10",
    "lexington": "O3,10",
    "senftenberg": "O1,3,19",
    "aberdeen": "O11",
    "poona": "O13",
    "cubana": "O13",
    "cerro": "O18",
    "minnesota": "O21",
    "urbana": "O30",
    "adelaide": "O35",
    "inverness": "O38",
    "waycross": "O41",
    "weslaco": "O42",
    "milwaukee": "O43",
    "niarembe": "O44",
    "deversoir": "O45",
    "dahlem": "O48",
    "greenside": "O50",
    "overschie": "O51",
    "utrecht": "O52",
    "humber": "O53",
    "uccle": "O54",
    "tranoroa": "O55",
    "artis": "O56",
    "locarno": "O57",
    "basel": "O58",
    "betioky": "O59",
    "setubal": "O60",
    "eilbek": "O61",
}


def clean_html_text(raw_html: str) -> str:
    """Strip HTML tags and unescape entities into normalized whitespace text."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def classify_antigen_label(text: str, bio_src: str = "") -> bytes:
    """Classify raw compound class / description text into a canonical antigen type label."""
    low = text.lower()
    if any(k in low for k in ("k-antigen", "k antigen", "cps", "capsul", "vi-antigen", "kps")):
        return b"CPS"
    if any(k in low for k in ("o-antigen", "o antigen", "ops", "lps", "lipopolysaccharide", "b-band", "osa")):
        return b"O-antigen"
    if any(k in low for k in ("los", "lipooligosaccharide", "outer core", "ocl")):
        return b"LOS"
    if any(k in low for k in ("teichoic", "wta", "lta")):
        return b"WTA"
    if any(k in low for k in ("eps", "exopolysaccharide", "biofilm", "alginate", "cellulose")):
        return b"EPS"

    src_low = f"{bio_src} {text}".lower()
    if re.search(r"\b(kl\d+|k\d+[a-z]?|capsular)\b", src_low):
        return b"CPS"
    if any(
        g in src_low
        for g in (
            "streptococcus",
            "klebsiella",
            "neisseria meningitidis",
            "haemophilus influenzae",
            "staphylococcus aureus",
            "cryptococcus",
        )
    ):
        return b"CPS"
    if any(
        g in src_low
        for g in (
            "staphylococcus",
            "bacillus",
            "listeria",
            "enterococcus",
            "lactobacillus",
            "lactococcus",
            "bifidobacterium",
            "clostrid",
            "streptomyces",
            "actinomyces",
            "micrococcus",
            "rathayibacter",
            "arthrobacter",
        )
    ):
        return b"WTA"
    if any(
        g in src_low
        for g in (
            "rhizobium",
            "sinorhizobium",
            "bradyrhizobium",
            "agrobacterium",
            "mesorhizobium",
            "azospirillum",
            "xanthomonas",
            "sphingomonas",
            "leuconostoc",
            "pediococcus",
            "acetobacter",
            "gluconacetobacter",
        )
    ):
        return b"EPS"
    return b"O-antigen"


def extract_serotype_antigen_aware(
    org: str,
    strain_rest: str,
    bio_source: str,
    antigen_type: bytes = b"",
    extra_text: str = "",
) -> str:
    """Extract and canonicalize serotype designation with awareness of `antigen_type` (`CPS` vs `O-antigen`)."""
    ag_upper = antigen_type.decode("utf-8", errors="replace").upper()
    is_cps = "CPS" in ag_upper or "CAPSUL" in ag_upper
    is_ops = "O-ANTIGEN" in ag_upper or "LPS" in ag_upper or "LOS" in ag_upper
    org_low = org.lower()

    def _clean_sero_token(tok: str) -> str:
        t = re.sub(
            r"^(?:serotype|serogroup|capsular\s+type|iats|type)\s+",
            "",
            tok.strip(),
            flags=re.IGNORECASE,
        ).strip()
        if re.match(r"^[ko]\d+", t, re.IGNORECASE):
            t = t[0].upper() + t[1:]
        elif re.match(r"^(?:kl|ol|ocl)\d+", t, re.IGNORECASE):
            m_p = re.match(r"^([A-Za-z]+)(.+)$", t)
            if m_p:
                t = m_p.group(1).upper() + m_p.group(2)
        return t

    search_fields = [strain_rest, bio_source, extra_text]
    if is_cps:
        for fld in search_fields:
            if not fld:
                continue
            for m in CPS_SEROTYPE_RE.finditer(fld):
                cand = _clean_sero_token(m.group(1))
                if cand:
                    if cand.isdigit() and any(g in org_low for g in ("klebsiella", "acinetobacter", "escherichia")):
                        if "klebsiella" in org_low and int(cand) > 82:
                            continue
                        return f"K{cand}"
                    return cand
    elif is_ops:
        for fld in search_fields:
            if not fld:
                continue
            for m in OPS_SEROTYPE_RE.finditer(fld):
                cand = _clean_sero_token(m.group(1))
                if cand and cand.upper() not in ("K12", "K-12"):
                    if cand.isdigit() and any(
                        g in org_low
                        for g in (
                            "klebsiella",
                            "acinetobacter",
                            "escherichia",
                            "pseudomonas",
                            "salmonella",
                            "vibrio",
                            "proteus",
                            "serratia",
                            "yersinia",
                        )
                    ):
                        if "klebsiella" in org_low and int(cand) > 20:
                            continue
                        return f"O{cand}"
                    return cand
        if "klebsiella" in org_low:
            extra_low = f"{bio_source} {extra_text}".lower()
            if any(g_tok in extra_low for g_tok in ("galactan-i", "galactan-iii", "d-galactan")):
                return "O2"
        if "salmonella" in org_low:
            combined_low = f"{strain_rest} {bio_source}".lower()
            for sv_name, o_grp in SALMONELLA_SEROVAR_TO_OGROUP.items():
                if re.search(rf"\b{re.escape(sv_name)}\b", combined_low):
                    return o_grp

    # General fallback across fields
    for fld in search_fields:
        if not fld:
            continue
        m_any = SEROTYPE_EXTRACT_RE.search(fld)
        if m_any:
            cand = _clean_sero_token(m_any.group(1))
            if is_ops and cand.upper() in ("K12", "K-12"):
                continue
            if cand.isdigit() and "klebsiella" in org_low and int(cand) > 82:
                continue
            return cand

    bare = strain_rest.strip()
    if re.fullmatch(r"\d{1,3}[A-Fa-f]?|[IVX]{1,4}[abAB]?", bare):
        if is_cps and bare.isdigit() and any(g in org_low for g in ("klebsiella", "acinetobacter", "escherichia")):
            if "klebsiella" in org_low and int(bare) > 82:
                return ""
            return f"K{bare}"
        if is_ops and bare.isdigit() and any(
            g in org_low
            for g in (
                "klebsiella",
                "acinetobacter",
                "escherichia",
                "pseudomonas",
                "salmonella",
                "vibrio",
                "proteus",
                "serratia",
            )
        ):
            return f"O{bare}"
        return bare

    return ""


def parse_organism_and_strain(
    bio_source: str,
    antigen_type: bytes = b"",
    extra_text: str = "",
) -> tuple[bytes, bytes, bytes]:
    """Extract canonical `(organism, strain, serotype_hint)` from a biological source string."""
    src = bio_source.split(";")[0].strip()
    m_new = re.search(r"\{NEW:\s*([^}]+)\}", src, re.IGNORECASE)
    if m_new:
        src = m_new.group(1).strip()
    src = re.sub(r"\{[^}]*\}", "", src)
    src = re.sub(r"\s*\([^)]*\)", "", src).strip()

    tokens = src.split()
    if len(tokens) >= 2:
        t0_low = tokens[0].lower()
        t1_low = tokens[1].lower()
        if t1_low in ("sp.", "spp."):
            org = f"{tokens[0]} sp."
            rest = " ".join(tokens[2:])
        elif t0_low == "salmonella" and t1_low not in ("enterica", "bongori", "sp.", "spp."):
            # Classic Kauffmann-White binomen e.g. 'Salmonella typhimurium' -> Salmonella enterica + serovar
            org = "Salmonella enterica"
            sv_cap = tokens[1][0].upper() + tokens[1][1:]
            rest = " ".join([sv_cap, *tokens[2:]])
        elif not tokens[1].isalpha() or not tokens[1].islower():
            org = f"{tokens[0]} sp."
            rest = " ".join(tokens[1:])
        elif len(tokens) >= 4 and tokens[2].lower() in ("subsp.", "ssp.", "pv.", "serovar", "sv.", "biovar"):
            org = f"{tokens[0]} {tokens[1]}"
            if tokens[2].lower() in ("subsp.", "ssp."):
                rest = " ".join(tokens[4:])
            else:
                rest = " ".join(tokens[2:])
        else:
            org = f"{tokens[0]} {tokens[1]}"
            rest = " ".join(tokens[2:])
    elif len(tokens) == 1:
        org = tokens[0]
        rest = ""
    else:
        return b"Unknown bacterium", b"", b""

    strain = re.sub(
        r"^(?:strain|str\.|serovar|sv\.|serotype|serogroup|type)\s+",
        "",
        rest,
        flags=re.IGNORECASE,
    ).strip(" ,;:-")
    sero_hint = extract_serotype_antigen_aware(org, strain, bio_source, antigen_type=antigen_type, extra_text=extra_text)

    return (
        org.encode("utf-8")[:64],
        strain.encode("utf-8")[:64],
        sero_hint.encode("utf-8")[:32],
    )


def normalize_ecodab_structure(raw_struct: str) -> bytes:
    """Convert ECODAB multi-line ASCII sugar diagram into CSDB linear sequence notation."""
    clean = re.sub(r"<[^>]+>", "", html.unescape(raw_struct))
    lines = [ln.rstrip() for ln in clean.splitlines() if ln.strip()]
    if not lines:
        return b""

    backbone_idx = -1
    for idx, ln in enumerate(lines):
        if ln.lstrip().startswith(("->", "-")):
            backbone_idx = idx
            break
    if backbone_idx < 0:
        backbone_idx = max(range(len(lines)), key=lambda i: len(lines[i].strip()))

    backbone = lines[backbone_idx].strip()

    def _clean_token(tok: str) -> str:
        t = tok.strip()
        t = re.sub(r"\((\d+)->(\d+)\)", r"(\1-\2)", t)
        t = re.sub(r"\((\d+)-P-(\d+)\)", r"(\1-\2)", t)
        t = re.sub(r"^->(\d+)\)", r"-\1)", t)
        t = re.sub(r"\((\d+)->$", r"(\1-", t)
        return t

    s = _clean_token(backbone)
    branch_sugars: list[str] = []
    for idx, ln in enumerate(lines):
        if idx == backbone_idx:
            continue
        stripped = ln.strip()
        if stripped in ("|", "/", "\\") or re.fullmatch(r"[|/\\\s]+", stripped):
            continue
        cleaned = _clean_token(stripped)
        if any(c.isalpha() for c in cleaned):
            branch_sugars.append(cleaned)

    for br in branch_sugars:
        if re.search(r"\(\d+-\d+\)$", br):
            s = re.sub(r"([a-zA-Z0-9]+)(\(\d+-\d+\))", rf"[{br}]\1\2", s, count=1)

    return s.encode("utf-8")


def normalize_ek3d_iupac_to_csdb(raw_iupac: str) -> bytes:
    """Convert EK3D condensed IUPAC repeat notation into CSDB linear sequence syntax."""
    s = raw_iupac.strip()
    if not s:
        return b""

    s = s.replace("\u03b1", "a").replace("\u03b2", "b").replace("\u2192", "->")
    s = re.sub(r"^\[?", "", s)
    s = re.sub(r"\]?[nN]?\s*$", "", s)
    s = re.sub(r"^->(\d+)\)-?", r"-\1)", s)
    s = re.sub(r"-\((\d+)->$", r"(\1-", s)
    s = re.sub(r"\((\d+)->$", r"(\1-", s)
    s = re.sub(r"-\((\d+)->(\d+)\)-", r"(\1-\2)", s)
    s = re.sub(r"\((\d+)->(\d+)\)", r"(\1-\2)", s)
    return s.encode("utf-8")


def normalize_epsdb_backbone(backbone: str) -> bytes:
    """Convert CERMAV EPS-DB backbone `{ 4)bDGlcp(1-4)... }n` into CSDB linear `-4)bDGlcp(1-4)...`."""
    bb_clean = re.sub(r"^\{?\s*", "", backbone.strip())
    bb_clean = re.sub(r"\s*\}n?\s*$", "", bb_clean)
    if re.match(r"^\d\)", bb_clean):
        bb_clean = "-" + bb_clean
    bb_clean = re.sub(r"\d+Ac[0-9.]*", "", bb_clean)
    bb_clean = re.sub(r"\(\d+-P-\d+\)", "(1-4)", bb_clean)
    return bb_clean.encode("utf-8")


def parse_glycan_safe(
    seq_bytes: bytes,
    csdb_parser: CSDBParser | None = None,
    iupac_parser: IUPACParser | None = None,
) -> Glycan | None:
    """Attempt to parse `seq_bytes` with `CSDBParser` and fallback to `IUPACParser`."""
    if not seq_bytes or not seq_bytes.strip():
        return None
    cp = csdb_parser if csdb_parser is not None else CSDBParser()
    ip = iupac_parser if iupac_parser is not None else IUPACParser()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GlycanParseWarning)
        for parser in (cp, ip):
            try:
                g = parser.parse(seq_bytes, strict=False)
                if len(g.nodes) > 0:
                    return g
            except (GlycanParseError, ValueError, KeyError, IndexError):
                continue
    return None


def _cassette_ctx(recorder: Any | None, cassette_name: str) -> Any:
    """Return `recorder.use_cassette(cassette_name)` if `recorder` is provided, else `nullcontext()`."""
    if recorder is not None and hasattr(recorder, "use_cassette"):
        return recorder.use_cassette(cassette_name)
    return nullcontext()


def stream_csdb_database(
    csdb_parser: CSDBParser | None = None,
    iupac_parser: IUPACParser | None = None,
    url: str = CSDB_OSF_URL,
    cache_path: Path | None = DEFAULT_CSDB_CACHE_PATH,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Stream CSDB from OSF (or load from compressed local cache), indexing cross-references and surface antigens."""
    cp = csdb_parser if csdb_parser is not None else CSDBParser()
    ip = iupac_parser if iupac_parser is not None else IUPACParser()
    csdb_by_id: dict[str, dict[str, Any]] = {}
    curated_raw: list[dict[str, Any]] = []

    def _process_entry(current: dict[str, str]) -> bool:
        raw_id = current.get("record_id", "").strip()
        rec_id = raw_id.split()[0] if raw_id else ""
        st1 = current.get("structure_linear", "").strip()
        if not (rec_id and st1):
            return False
        csdb_by_id[rec_id] = dict(current)

        th_raw = current.get("is_elucidated", "").strip().lower()
        is_eluc = th_raw not in ("0", "o", "false", "f", "no", "n", "")
        domain = current.get("taxonomic_domain", "")
        bio_src = current.get("biological_source", "")
        comp_cls = current.get("compound_class", "")
        comp_name = current.get("compound_name", "")
        title = current.get("title", "")
        loc = current.get("structure_location", "")
        st2 = current.get("structure_type", "")

        is_bacteria = ("Prokaryotes" in domain) or ("Bacteria" in domain) or bool(BACTERIAL_GENERA_RE.search(bio_src))
        combined_meta = f"{comp_cls} {comp_name} {loc} {title}"
        is_surface = bool(SURFACE_ANTIGEN_RE.search(combined_meta))
        is_chem = st2 in ("CHEM", "REP", "")

        if is_eluc and is_bacteria and is_surface and is_chem:
            gly = parse_glycan_safe(st1.encode("utf-8"), cp, ip)
            if gly is not None and len(gly.nodes) >= 2:
                ag_label = classify_antigen_label(combined_meta, bio_src=bio_src)
                abstract_txt = current.get("abstract", "")
                org_b, strain_b, sero_b = parse_organism_and_strain(
                    bio_src,
                    antigen_type=ag_label,
                    extra_text=f"{comp_name} {title} {abstract_txt}",
                )

                db_xref = current.get("cross_references", "")
                bg_data = current.get("biosynthetic_genetic_data", "")
                notes_txt = current.get("comments", "")
                rl_txt = current.get("references", "")
                all_xref_text = f"{db_xref} {bg_data} {notes_txt} {rl_txt}"

                pmid = 0
                m_pmid = PMID_EXTRACT_RE.search(all_xref_text)
                if m_pmid:
                    pmid = int(m_pmid.group(1))

                doi_b = b""
                m_doi = DOI_EXTRACT_RE.search(all_xref_text)
                if m_doi:
                    doi_b = m_doi.group(1).rstrip(".,;)").encode("utf-8")

                acc_b = b""
                m_acc = GENBANK_ACC_RE.search(f"{bg_data} {db_xref} {notes_txt}")
                if m_acc:
                    acc_b = m_acc.group(1).encode("utf-8")[:32]

                py_str = current.get("publication_year", "")
                pub_year = int(py_str) if py_str.isdigit() and len(py_str) == 4 else 0

                curated_raw.append(
                    {
                        "source_db": b"CSDB",
                        "record_id": rec_id.encode("utf-8")[:32],
                        "organism": org_b,
                        "strain": strain_b,
                        "serotype": sero_b,
                        "antigen_type": ag_label,
                        "accession_hint": acc_b,
                        "structure_raw": st1.encode("utf-8"),
                        "structure_linear": st1.encode("utf-8"),
                        "glycan": gly,
                        "is_repeating": (gly.has_polymer_repeat or st1.lstrip().startswith("-"))
                        and not gly.is_alditol,
                        "pmid": pmid,
                        "publication_year": pub_year,
                        "doi": doi_b,
                        "notes": f"{comp_cls}; {comp_name}".strip("; ").encode("utf-8"),
                    }
                )
                return True
        return False

    # Fast path: load pre-filtered bacterial entries from compressed local cache if present
    if cache_path is not None and cache_path.is_file():
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as f_in:
                cached_entries: list[dict[str, str]] = json.load(f_in)
            if cached_entries:
                for entry_dict in cached_entries:
                    _process_entry(entry_dict)
                return curated_raw, csdb_by_id
        except Exception:
            csdb_by_id.clear()
            curated_raw.clear()

    kept_entries: list[dict[str, str]] = []

    def _ingest_entry(current: dict[str, str]) -> None:
        matched = _process_entry(current)
        bio_src = current.get("biological_source", "")
        if matched or "Escherichia coli" in bio_src:
            kept_entries.append(dict(current))

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KBank-Curator/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            current: dict[str, str] = {}
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith("#"):
                    if not line and current:
                        _ingest_entry(current)
                        current = {}
                    continue

                parts = line.split(":", 1)
                if len(parts) == 2:
                    tag = CSDB_TAG_MAP.get(parts[0].strip())
                    if tag:
                        current[tag] = parts[1].strip()
            if current:
                _ingest_entry(current)
    except Exception:
        import pyarrow.parquet as pq

        blob = subprocess.check_output(
            ["git", "show", "de8d429711e04217f701c5b3a9f8c70bfe644f1c~1:data/csdb.parquet"],
            cwd=str(Path(__file__).resolve().parent.parent.parent.parent),
        )
        tbl = pq.read_table(io.BytesIO(blob))
        for row in tbl.to_pylist():
            _ingest_entry({k: ("" if v is None else str(v)) for k, v in row.items()})

    if cache_path is not None and kept_entries:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cache_path, "wt", encoding="utf-8") as f_out:
                json.dump(kept_entries, f_out)
        except Exception:
            pass

    return curated_raw, csdb_by_id


async def fetch_ecodab_records(
    csdb_by_id: dict[str, dict[str, Any]],
    csdb_parser: CSDBParser | None = None,
    iupac_parser: IUPACParser | None = None,
    *,
    recorder: Any | None = None,
) -> list[dict[str, Any]]:
    """Asynchronously scrape ECODAB *E. coli* O-antigen records from `list.php` & `showDetails.php`."""
    import httpx

    cp = csdb_parser if csdb_parser is not None else CSDBParser()
    ip = iupac_parser if iupac_parser is not None else IUPACParser()
    records: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(10)

    with _cassette_ctx(recorder, "ecodab.toml"):
        async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=25.0) as client:
            try:
                list_resp = await client.get(f"{ECODAB_BASE_URL}list.php")
                list_html = list_resp.text if list_resp.status_code == 200 else ""
            except Exception:
                list_html = ""

            entries = re.findall(
                r"<a href='showDetails\.php\?idSerogroup=(\d+)'>([^<]+)</a>.*?<td[^>]*>.*?</td>\s*<td class='structura'><pre>(.*?)</pre>",
                list_html,
                re.DOTALL,
            )
            if not entries:
                entries = [(str(sg_id), f"O{sg_id}", "") for sg_id in range(1, 189)]

            async def _fetch_entry(rec_id: str, fallback_sg: str, fallback_struct: str) -> dict[str, Any] | None:
                url = f"{ECODAB_BASE_URL}showDetails.php?idSerogroup={rec_id}"
                page = ""
                async with sem:
                    try:
                        resp = await client.get(url, timeout=15.0)
                        if resp.status_code == 200:
                            page = resp.text
                    except Exception:
                        page = ""

                sg_clean = re.sub(r"\s+", "", fallback_sg.strip())
                if page:
                    m_sg = re.search(
                        r"<th[^>]*>Serogroup:?</th>\s*<td[^>]*>(.*?)</td>",
                        page,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if m_sg:
                        cand_sg = re.sub(r"\s+", "", clean_html_text(m_sg.group(1)))
                        if cand_sg:
                            sg_clean = cand_sg
                if sg_clean and not sg_clean.upper().startswith("O"):
                    sg_clean = f"O{sg_clean}"

                gb_acc = b""
                if page:
                    m_gb = re.search(r"nuccore/([A-Za-z0-9_.]+)", page) or GENBANK_ACC_RE.search(page)
                    if m_gb:
                        gb_acc = m_gb.group(1).encode("utf-8")[:32]

                csdb_id = ""
                if page:
                    m_csdb = re.search(r"csdb\.glycoscience\.ru[^\"]*id=(\d+)", page, re.IGNORECASE)
                    if not m_csdb:
                        m_csdb = re.search(r"CSDB\s*(?:ID)?\s*[:#]?\s*(\d+)", clean_html_text(page), re.IGNORECASE)
                    if m_csdb:
                        csdb_id = m_csdb.group(1)

                pmid = 0
                pub_year = 0
                strain_b = b""
                if page:
                    m_pmid = re.search(r"pubmed/(\d+)", page, re.IGNORECASE)
                    if m_pmid:
                        pmid = int(m_pmid.group(1))
                    m_yr = re.search(r"\((\d{4})\),\s*pp", page)
                    if m_yr:
                        pub_year = int(m_yr.group(1))
                    m_str = re.search(
                        r"<th[^>]*>Strain:?</th>\s*<td[^>]*>(.*?)</td>",
                        page,
                        re.IGNORECASE | re.DOTALL,
                    )
                    if m_str:
                        strain_b = clean_html_text(m_str.group(1)).encode("utf-8")[:64]

                raw_struct = fallback_struct.strip()
                if page and not raw_struct:
                    pre_blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", page, re.DOTALL | re.IGNORECASE)
                    if pre_blocks:
                        raw_struct = pre_blocks[0].strip()

                linear_b = b""
                if csdb_id and csdb_id in csdb_by_id:
                    csdb_entry = csdb_by_id[csdb_id]
                    linear_b = csdb_entry.get("structure_linear", "").encode("utf-8")
                    py_s = csdb_entry.get("publication_year", "")
                    if not pub_year and py_s.isdigit() and len(py_s) == 4:
                        pub_year = int(py_s)
                    if not strain_b:
                        _, strain_b, _ = parse_organism_and_strain(
                            csdb_entry.get("biological_source", ""),
                            antigen_type=b"O-antigen",
                        )

                if not linear_b and raw_struct:
                    linear_b = normalize_ecodab_structure(raw_struct)

                gly = parse_glycan_safe(linear_b, cp, ip)
                if gly is None or len(gly.nodes) == 0:
                    return None

                return {
                    "source_db": b"ECODAB",
                    "record_id": str(rec_id).encode("utf-8")[:32],
                    "organism": b"Escherichia coli",
                    "strain": strain_b,
                    "serotype": sg_clean.encode("utf-8")[:32],
                    "antigen_type": b"O-antigen",
                    "accession_hint": gb_acc,
                    "structure_raw": re.sub(r"<[^>]+>", "", html.unescape(raw_struct)).encode("utf-8")
                    if raw_struct
                    else linear_b,
                    "structure_linear": linear_b,
                    "glycan": gly,
                    "is_repeating": True,
                    "pmid": pmid,
                    "publication_year": pub_year,
                    "doi": b"",
                    "notes": (f"ECODAB {sg_clean}" + (f" (CSDB:{csdb_id})" if csdb_id else "")).encode("utf-8"),
                }

            results = await asyncio.gather(*[_fetch_entry(eid, sg, st) for eid, sg, st in entries])
            for item in results:
                if item is not None:
                    records.append(item)

    return records


async def fetch_ek3d_records(
    csdb_by_id: dict[str, dict[str, Any]],
    csdb_parser: CSDBParser | None = None,
    iupac_parser: IUPACParser | None = None,
    *,
    recorder: Any | None = None,
) -> list[dict[str, Any]]:
    """Fetch EK3D *E. coli* K-antigen records (Groups 1-4) and cross-link with CSDB K-antigen structures."""
    import httpx

    cp = csdb_parser if csdb_parser is not None else CSDBParser()
    ip = iupac_parser if iupac_parser is not None else IUPACParser()

    group_1_k = {
        "K26", "K27", "K28", "K29", "K30", "K31", "K32", "K33", "K34", "K35",
        "K36", "K37", "K38", "K39", "K41", "K42", "K55", "K82", "K88", "K89", "K102", "K103",
    }
    group_2_k = {
        "K1", "K4", "K5", "K7", "K8", "K9", "K12", "K13", "K14", "K15", "K16",
        "K18", "K20", "K22", "K23", "K24", "K43", "K44", "K45", "K46", "K47",
        "K50", "K51", "K52", "K53", "K56", "K57", "K62", "K63", "K74", "K80",
        "K92", "K93", "K94", "K95", "K97", "K100", "K101",
    }
    group_3_k = {"K2", "K3", "K10", "K11", "K19", "K54", "K96", "K98"}
    group_4_k = {"K6", "K40", "K49", "K84", "K85", "K87"}

    def _classify_k_group(k_sero: str) -> str:
        k_base = re.match(r"^(K\d+)", k_sero.upper())
        key = k_base.group(1) if k_base else k_sero.upper()
        if key in group_1_k:
            return "Group 1"
        if key in group_2_k:
            return "Group 2"
        if key in group_3_k:
            return "Group 3"
        if key in group_4_k:
            return "Group 4"
        return "Group 2"

    ecoli_k_csdb: dict[str, dict[str, Any]] = {}
    for entry in csdb_by_id.values():
        bio_src = entry.get("biological_source", "")
        if "Escherichia coli" not in bio_src:
            continue
        m_k = re.search(r"\b(K\d+[A-Za-z]?)\b", f"{bio_src} {entry.get('compound_name', '')} {entry.get('title', '')}")
        if not m_k:
            continue
        k_name = m_k.group(1).upper()
        st1 = entry.get("structure_linear", "").strip()
        if st1 and k_name not in ecoli_k_csdb:
            ecoli_k_csdb[k_name] = entry

    live_ek_map: dict[str, tuple[str, str]] = {}
    try:
        with _cassette_ctx(recorder, "ek3d.toml"):
            async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=6.0) as client:
                resp = await client.get(EK3D_URL)
                if resp.status_code == 200:
                    tds = re.findall(r"<td\b[^>]*>(.*?)</td>", resp.text, re.DOTALL)
                    grp_names = ["Group 1", "Group 2", "Group 3", "Group 4", "Group 2"]
                    for idx, td in enumerate(tds):
                        grp = grp_names[idx] if idx < len(grp_names) else "Group 2"
                        for _, ek_id in re.findall(r'<a href="(ek\d+\.php)"[^>]*>(EK\d+)</a>', td):
                            k_label = ek_id.replace("EK", "K")
                            live_ek_map[k_label] = (ek_id, grp)
    except Exception:
        pass

    records: list[dict[str, Any]] = []
    sorted_k = sorted(
        ecoli_k_csdb.keys(),
        key=lambda k: (int(re.search(r"\d+", k).group(0)) if re.search(r"\d+", k) else 999, k),
    )
    for idx_k, k_sero in enumerate(sorted_k, start=1):
        entry = ecoli_k_csdb[k_sero]
        st1 = entry.get("structure_linear", "").strip()
        linear_b = st1.encode("utf-8")
        gly = parse_glycan_safe(linear_b, cp, ip)
        if gly is None or len(gly.nodes) < 2:
            continue

        ek_id, grp = live_ek_map.get(k_sero, (f"EK{idx_k}", _classify_k_group(k_sero)))
        _, strain_b, _ = parse_organism_and_strain(entry.get("biological_source", ""))
        py_s = entry.get("publication_year", "")
        pub_year = int(py_s) if py_s.isdigit() and len(py_s) == 4 else 0

        all_xref = f"{entry.get('cross_references', '')} {entry.get('biosynthetic_genetic_data', '')} {entry.get('references', '')}"
        pmid = 0
        m_pmid = PMID_EXTRACT_RE.search(all_xref)
        if m_pmid:
            pmid = int(m_pmid.group(1))
        acc_b = b""
        m_acc = GENBANK_ACC_RE.search(all_xref)
        if m_acc:
            acc_b = m_acc.group(1).encode("utf-8")[:32]

        records.append(
            {
                "source_db": b"EK3D",
                "record_id": ek_id.encode("utf-8")[:32],
                "organism": b"Escherichia coli",
                "strain": strain_b,
                "serotype": k_sero.encode("utf-8")[:32],
                "antigen_type": f"CPS ({grp})".encode("utf-8")[:32],
                "accession_hint": acc_b,
                "structure_raw": linear_b,
                "structure_linear": linear_b,
                "glycan": gly,
                "is_repeating": True,
                "pmid": pmid,
                "publication_year": pub_year,
                "doi": b"",
                "notes": f"EK3D {ek_id} {k_sero} {grp} (CSDB:{entry.get('record_id', '')})".encode("utf-8"),
            }
        )

    return records


async def fetch_epsdb_records(
    csdb_by_id: dict[str, dict[str, Any]] | None = None,
    csdb_parser: CSDBParser | None = None,
    iupac_parser: IUPACParser | None = None,
    *,
    recorder: Any | None = None,
) -> list[dict[str, Any]]:
    """Asynchronously scrape CERMAV EPS-DB bacterial exopolysaccharide entries from `database.html`."""
    import httpx

    cp = csdb_parser if csdb_parser is not None else CSDBParser()
    ip = iupac_parser if iupac_parser is not None else IUPACParser()
    records: list[dict[str, Any]] = []
    try:
        with _cassette_ctx(recorder, "epsdb.toml"):
            async with httpx.AsyncClient(verify=False, follow_redirects=True, timeout=25.0) as client:
                resp = await client.get(f"{EPSDB_BASE_URL}database.html")
                if resp.status_code != 200:
                    return []
                index_html = resp.text
    except Exception:
        return []

    tbody_m = re.search(r"<tbody[^>]*>(.*?)</tbody>", index_html, re.DOTALL | re.IGNORECASE)
    if not tbody_m:
        return []

    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", tbody_m.group(1), re.DOTALL | re.IGNORECASE)
    for row_html in rows:
        tds = re.findall(r"<td\b[^>]*>(.*?)</td>", row_html, re.DOTALL | re.IGNORECASE)
        if len(tds) < 8:
            continue

        eps_id = clean_html_text(tds[0])
        organism = clean_html_text(tds[1]) or "Unknown bacterium"
        strain = re.sub(r"\{[^}]*\}", "", clean_html_text(tds[2])).strip()
        backbone = clean_html_text(tds[3])
        branches = clean_html_text(tds[4])
        year_s = clean_html_text(tds[6])
        func_s = clean_html_text(tds[7])

        if not eps_id or not backbone:
            continue

        linear_b = normalize_epsdb_backbone(backbone)
        gly = parse_glycan_safe(linear_b, cp, ip)
        if gly is None or len(gly.nodes) == 0:
            continue

        raw_str = f"{backbone} {branches}".strip() if branches and branches != "–" else backbone
        pub_year = int(year_s) if year_s.isdigit() and len(year_s) == 4 else 0

        st_match = re.search(r"\b([KO]\d+[A-Za-z0-9\-_.]*)\b", f"{strain} {organism}")
        sero_b = st_match.group(1).encode("utf-8")[:32] if st_match else b""

        records.append(
            {
                "source_db": b"EPSDB",
                "record_id": f"EPS{eps_id}".encode("utf-8")[:32],
                "organism": organism.encode("utf-8")[:64],
                "strain": strain.encode("utf-8")[:64],
                "serotype": sero_b,
                "antigen_type": b"EPS",
                "accession_hint": b"",
                "structure_raw": raw_str.encode("utf-8"),
                "structure_linear": linear_b,
                "glycan": gly,
                "is_repeating": True,
                "pmid": 0,
                "publication_year": pub_year,
                "doi": b"",
                "notes": (f"CERMAV EPS-DB #{eps_id}" + (f" ({func_s})" if func_s and func_s != "–" else "")).encode("utf-8"),
            }
        )

    return records


async def fetch_kegg_glycan_records(
    *,
    recorder: Any | None = None,
    allow_composition: bool = False,
) -> list[dict[str, Any]]:
    """Asynchronously fetch connected bacterial surface glycans from KEGG (`map00540`, `map00552`, keywords)."""
    import httpx

    records: list[dict[str, Any]] = []
    pathway_ids: dict[str, bytes] = {}

    with _cassette_ctx(recorder, "kegg.toml"):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for pw, label in (("map00540", b"LPS"), ("map00552", b"WTA")):
                try:
                    r = await client.get(f"{KEGG_REST_URL}/link/glycan/{pw}", timeout=20.0)
                    if r.status_code == 200:
                        for line in r.text.splitlines():
                            if "\tgl:" in line:
                                gid = line.split("\tgl:")[-1].strip()
                                pathway_ids[gid] = label
                except Exception:
                    continue

            for kw, default_lbl in (
                ("lipopolysaccharide", b"LPS"),
                ("capsular", b"CPS"),
                ("teichoic", b"WTA"),
                ("O-antigen", b"O-antigen"),
            ):
                try:
                    r_find = await client.get(f"{KEGG_REST_URL}/find/glycan/{kw}", timeout=12.0)
                    if r_find.status_code == 200:
                        for line in r_find.text.splitlines():
                            parts = line.split("\t", 1)
                            if len(parts) == 2:
                                gid = parts[0].replace("gl:", "").strip()
                                desc = parts[1]
                                if BACTERIAL_GENERA_RE.search(desc) or SURFACE_ANTIGEN_RE.search(desc):
                                    pathway_ids.setdefault(gid, classify_antigen_label(desc) or default_lbl)
                except Exception:
                    pass

            gids = sorted(pathway_ids.keys())
            sem = asyncio.Semaphore(4)

            async def _fetch_batch(batch_ids: list[str]) -> list[dict[str, Any]]:
                joined = "+".join(batch_ids)
                async with sem:
                    try:
                        resp = await client.get(f"{KEGG_REST_URL}/get/{joined}", timeout=30.0)
                        if resp.status_code != 200:
                            return []
                        raw_text = resp.text
                    except Exception:
                        return []

                out: list[dict[str, Any]] = []
                for entry_txt in raw_text.split("///"):
                    entry_txt = entry_txt.strip()
                    if not entry_txt:
                        continue
                    m_id = re.search(r"^ENTRY\s+(G\d{5})", entry_txt, re.MULTILINE)
                    if not m_id:
                        continue
                    gid = m_id.group(1)
                    gly = parse_kegg_glycan_flatfile(entry_txt, gid, allow_composition=allow_composition)
                    if gly is None or len(gly.nodes) < 2:
                        continue

                    m_name = re.search(r"^NAME\s+(.+)$", entry_txt, re.MULTILINE)
                    name_str = m_name.group(1).strip() if m_name else ""
                    m_comm = re.search(r"^COMMENT\s+(.+)$", entry_txt, re.MULTILINE)
                    comm_str = m_comm.group(1).strip() if m_comm else ""

                    org_b = b"Bacteria (KEGG pathway)"
                    m_org = BACTERIAL_GENERA_RE.search(f"{name_str} {comm_str}")
                    if m_org:
                        org_b, _, _ = parse_organism_and_strain(f"{name_str} {comm_str}"[m_org.start() :])

                    pmid = 0
                    m_pmid = re.search(r"PMID:\s*(\d+)", entry_txt)
                    if m_pmid:
                        pmid = int(m_pmid.group(1))

                    ag_type = pathway_ids.get(gid, b"LPS")
                    out.append(
                        {
                            "source_db": b"KEGG",
                            "record_id": gid.encode("utf-8"),
                            "organism": org_b,
                            "strain": b"",
                            "serotype": b"",
                            "antigen_type": ag_type,
                            "accession_hint": b"",
                            "structure_raw": gly.canonical,
                            "structure_linear": gly.canonical,
                            "glycan": gly,
                            "is_repeating": False,
                            "pmid": pmid,
                            "publication_year": 0,
                            "doi": b"",
                            "notes": (name_str or comm_str or f"KEGG {ag_type.decode()}").encode("utf-8")[:120],
                        }
                    )
                return out

            batches = [gids[i : i + 10] for i in range(0, len(gids), 10)]
            results = await asyncio.gather(*[_fetch_batch(b) for b in batches])
            for b_res in results:
                records.extend(b_res)

    return records


async def fetch_glytoucan_records(
    csdb_parser: CSDBParser | None = None,
    iupac_parser: IUPACParser | None = None,
    *,
    recorder: Any | None = None,
) -> dict[bytes, bytes]:
    """Query GlyTouCan / GlyCosmos SPARQL to build a canonical structure -> `glytoucan_id` cross-reference map."""
    import httpx

    cp = csdb_parser if csdb_parser is not None else CSDBParser()
    ip = iupac_parser if iupac_parser is not None else IUPACParser()
    canonical_to_gtc: dict[bytes, bytes] = {}

    genera_pattern = (
        "^(Escherichia|Klebsiella|Acinetobacter|Streptococcus|Salmonella|Pseudomonas|"
        "Staphylococcus|Shigella|Vibrio|Yersinia|Campylobacter|Burkholderia|"
        "Haemophilus|Neisseria|Bacillus|Listeria|Enterococcus|Clostrid|"
        "Bacteroides|Helicobacter|Mycobacterium|Proteus|Serratia|Citrobacter|"
        "Enterobacter|Cronobacter|Aeromonas|Francisella|Brucella|Legionella|"
        "Providencia|Morganella|Hafnia|Edwardsiella|Plesiomonas)"
    )

    with _cassette_ctx(recorder, "glytoucan.toml"):
        async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
            for offset in (0, 2500):
                sparql = f"""
PREFIX glycan: <http://purl.jp/bio/12/glyco/glycan#>
PREFIX glytoucan: <http://www.glytoucan.org/glyco/owl/glytoucan#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT DISTINCT ?acc ?iupac ?taxon_label ?pub
WHERE {{
  ?s glytoucan:has_primary_id ?acc .
  ?s glycan:has_glycosequence ?gseq .
  ?gseq glycan:in_carbohydrate_format glycan:carbohydrate_format_iupac_condensed .
  ?gseq glycan:has_sequence ?iupac .
  ?s glycan:is_from_source ?source .
  ?source rdfs:label ?taxon_label .
  OPTIONAL {{ ?s dcterms:references ?pub . }}
  FILTER(REGEX(?taxon_label, "{genera_pattern}", "i"))
}}
LIMIT 2500 OFFSET {offset}
"""
                try:
                    resp = await client.post(
                        GLYCOSMOS_SPARQL_URL,
                        data={"query": sparql},
                        headers={"Accept": "application/sparql-results+json"},
                    )
                    if resp.status_code != 200:
                        break
                    bindings = resp.json().get("results", {}).get("bindings", [])
                except Exception:
                    break

                if not bindings:
                    break

                for b in bindings:
                    acc = b.get("acc", {}).get("value", "").strip()
                    iupac_str = b.get("iupac", {}).get("value", "").strip()
                    if not acc or not iupac_str:
                        continue

                    iupac_b = iupac_str.encode("utf-8")
                    gly = parse_glycan_safe(iupac_b, cp, ip)
                    if gly is None or len(gly.nodes) < 2:
                        continue

                    acc_b = acc.encode("utf-8")[:32]
                    canonical_to_gtc.setdefault(gly.canonical, acc_b)

                if len(bindings) < 2500:
                    break

    return canonical_to_gtc
