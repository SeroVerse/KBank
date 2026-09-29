from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, auto
import re
from typing import Any, Self

import numpy as np

from kbank.core.models import Catalog

_LOCUS_SUFFIX_PATTERN: re.Pattern[bytes] = re.compile(rb"_(?:kl|gi|ph)[0-9a-z_]*$", re.IGNORECASE)


# ----------------------------------------------------------------------------------------------------------------------
# Target Type Enum for Alignment Contract
# ----------------------------------------------------------------------------------------------------------------------

class TargetType(IntEnum):
    """
    Biological target scope for a biosynthetic gene role in glycan alignment.
    """
    LINKAGE = auto()          # Glycosidic linkages (GTs, Polymerases)
    MONOSACCHARIDE = auto()   # Core sugar precursors (Synthases, Epimerases, Reductases)
    MODIFIED_MONO = auto()    # Sugar harboring a specific modification (Acetyltransferases, etc.)
    REDUCING_END = auto()     # The initial lipid-linked reducing-end sugar (Initiators)
    ANY = auto()              # Uncharacterized / background (UNKNOWN)
    NONE = auto()             # Non-biosynthetic / Mobile elements (Transposases, Integrases)


# ----------------------------------------------------------------------------------------------------------------------
# Canonical Gene Roles Database Manifest
# ----------------------------------------------------------------------------------------------------------------------

# (name, abbreviation, category, symbol_regex, desc_regex, description, matches_linkages, allowed_names, allowed_classes, disallowed_classes, aliases, trusted_prefixes)
_GENE_DATA: list[tuple[
    bytes, bytes, bytes, bytes, bytes, bytes, bool,
    tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...]
]] = [
    # Transport & Polymer Regulation
    (
        b"FLIPPASE", b"Wzx", b"Transport",
        rb"^(?:wzx|gmla)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"\bflippase\b|\bwzx(?:_[a-z0-9_]+|[a-z0-9]+)?\b|o-antigen flippase|capsular polysaccharide flippase|gtra-like|prophage bactoprenol-linked glucose translocase",
        b"Wzx flippase; translocates completed lipid-linked repeat units across inner membrane.",
        False, (), (), (b"Mod",), (b"wzx", b"gmla"),
        (b"wzx", b"gmla"),
    ),
    (
        b"POLYMERASE", b"Wzy", b"PolymerAssembly",
        rb"^wzy(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"\bpolymerase\b|\bwzy(?:_[a-z0-9_]+|[a-z0-9]+)?\b|o-antigen polymerase|capsular polysaccharide polymerase",
        b"Wzy polymerase; forms the polymer repeat unit boundary glycosidic linkage.",
        True, (), (), (), (b"wzy",),
        (b"wzy",),
    ),
    (
        b"COPOLYMERASE", b"Wzz", b"Transport",
        rb"^(?:wzz|cld|cps[0-9]*c)(?:_[0-9]+|[a-z0-9])?$",
        rb"\bcopolymerase\b|co-polymerase|chain[\s-]length determinant|\bwzz\b",
        b"Wzz polysaccharide co-polymerase; modal chain length regulator.",
        False, (), (), (), (b"wzz", b"cld"),
        (b"wzz", b"cld"),
    ),
    (
        b"KINASE", b"Kin", b"Transport",
        rb"^(?:wzc|cps[0-9]*d|exop|ptk)(?:_[0-9]+|[a-z0-9])?$",
        rb"tyrosine kinase|autokinase|\bwzc\b|by-kinase|protein-tyrosine kinase",
        b"Wzc / BY-kinase; tyrosine autokinase coordinating polymer export.",
        False, (), (), (), (b"wzc", b"ptk"),
        (b"wzc",),
    ),
    (
        b"PHOSPHATASE", b"Wzb", b"Transport",
        rb"^(?:wzb|cps[0-9]*b|ptp|cpsacp)(?:_[0-9]+|[a-z0-9])?$",
        rb"phosphotyrosine phosphatase|tyrosine phosphatase|\bwzb\b|\bptp\b|acid\s*phosphatase|pap2|had[\s-]*superfamily|haloacid\s*dehalogenase|capsule.*(?:biosynthesis\s*)?phosphatase|inositol\s*monophosphatase",
        b"Wzb / PTP / CpsACP; low-molecular-weight phosphotyrosine and acid phosphatases.",
        False, (), (), (), (b"wzb", b"ptp", b"cpsacp"),
        (b"wzb", b"cpsacp"),
    ),
    (
        b"ABC_PERMEASE", b"Wzm", b"Transport",
        rb"^(?:wzm|kpsm|cpsu)(?:_[0-9]+|[a-z0-9])?$",
        rb"abc transporter permease|capsule export system permease|\bwzm\b|\bkpsm\b",
        b"Wzm ABC transporter permease; transmembrane translocation channel.",
        False, (), (), (b"Mod",), (b"wzm", b"kpsm"),
        (b"wzm", b"kpsm"),
    ),
    (
        b"ABC_ATPASE", b"Wzt", b"Transport",
        rb"^(?:wzt|kpst|cpsv)(?:_[0-9]+|[a-z0-9])?$",
        rb"abc transporter atpase|capsule export system atpase|\bwzt\b|\bkpst\b|abc transporter nucleotide-binding",
        b"Wzt ABC transporter ATPase; nucleotide-binding domain powering export.",
        False, (), (), (b"Mod",), (b"wzt", b"kpst"),
        (b"wzt", b"kpst"),
    ),
    (
        b"LIGASE", b"WaaL", b"PolymerAssembly",
        rb"^(?:waal|rfal)(?:_[0-9]+|[a-z0-9])?$",
        rb"o-antigen ligase|\bwaal\b|\brfal\b|lipid a ligase",
        b"WaaL ligase; attaches periplasmic polysaccharide to Lipid A core.",
        False, (), (), (), (b"waal", b"rfal"),
        (b"waal", b"rfal"),
    ),
    (
        b"TRANSLOCON", b"Wza", b"Transport",
        rb"^(?:wza|kps[de]|exoa|opx|gfc[b-d]|yjbh|ymcc)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"outer membrane.*(?:export|capsule)|\bwza\b|\bopx\b|translocon|export pore|\bgfc[b-d]\b|yjbh|ymcc|capsule.*(?:export|translocation).*(?:inner-membrane|periplasmic|protein kpse)",
        b"Wza / KpsD / KpsE outer membrane and periplasmic translocon complex for capsular polysaccharide surface export.",
        False, (), (), (), (b"wza", b"gfcd", b"gfcb", b"gfcc", b"kpsd", b"kpse"),
        (b"wza", b"kpsd", b"kpse", b"gfcd", b"gfcb", b"gfcc"),
    ),
    (
        b"CAPSULE_ANCHOR", b"Wzi", b"Transport",
        rb"^(?:wzi|kpsc)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"\bwzi\b|\bkpsc\b|capsule\s*(?:attachment|anchor|assembly)|outer\s*membrane\s*lectin|wzi\s+outer\s+membrane|capsule.*export.*protein\s*kpsc",
        b"Wzi outer membrane lectin / KpsC lipid anchor transferase; anchors exported capsular polysaccharide to the cell surface.",
        False, (), (), (), (b"wzi", b"anchor", b"capsule_anchor", b"kpsc"),
        (b"wzi", b"kpsc"),
    ),

    # Modifiers & Specific Substitutions
    (
        b"PYRUVYLTRANSFERASE", b"PT", b"Modifier",
        rb"^(?:wcak|ptr[0-9]*|cps[0-9]*e|pvt[a-z0-9_]*|wbmx)(?:_[0-9]+|[a-z0-9])?$",
        rb"pyruv(?:yl|ate)[\s-]?transferase|pyruvyl|\bwcak\b|\bwbmx\b",
        b"Pyruvyltransferase; adds pyruvate acetal (Pyr) modifications.",
        False, (b"Pyr",), (), (), (b"wcak", b"ptr", b"wbmx"),
        (b"ptr", b"wbmx"),
    ),
    (
        b"METHYLTRANSFERASE", b"MetT", b"Modifier",
        rb"^(?:met[0-9]+|omt[0-9]+)(?:_[0-9]+|[a-z0-9])?$",
        rb"methyl[\s-]?transferase|o-methyl",
        b"Methyltransferase; adds O-methyl (OMe) groups.",
        False, (b"OMe",), (), (), (),
        (),
    ),
    (
        b"ACETYLTRANSFERASE", b"AT", b"Modifier",
        rb"^(?:oat[a-z]*|wecd|wb[ub][a-z]|atr[0-9]*[a-z]?|neud|qhb[ac]|at[0-9]+|fdtc|fdtd|psah|qdte|wcio)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"(?:acetyl|acyl|o-acetyl|n-acetyl)[\s-]?transferase|acetyl-?\s*(?:or|/)\s*acyl|\bwecd\b|\boata\b|\batr[0-9]*[a-z]?\b|\bwcio\b",
        b"Acetyltransferase / Acyltransferase; adds O-acetyl (OAc) or hydroxyacyl (3HOBut, 3HOMyr) groups.",
        False, (b"OAc", b"3HOBut", b"3HOMyr", b"3HOLau", b"3HODco", b"3HOSte", b"3HOCap", b"Lau", b"Myr", b"Ste", b"Lipid"), (), (), (b"oata", b"wecd", b"atr", b"qdte", b"wcio"),
        (b"atr", b"qdte"),
    ),
    (
        b"PHOSPHOTRANSFERASE", b"PhoT", b"Modifier",
        rb"^(?:wcf[a-z]|pt[0-9]+|pgt[0-9]*|licd)(?:_[0-9]+|[a-z0-9])?$",
        rb"(?:phospho(?:ethanolamine|choline|glycerol)?|cholinephospho|glycerolphospho)[\s-]?transferase|\bpgt\b|\blicd\b|phosphocholine|choline\s*phosphotransferase",
        b"Phosphotransferase; adds phosphate (P), phosphoethanolamine (PEtn), or phosphocholine (ChoP) groups.",
        False, (b"P", b"PEtn", b"ChoP", b"GroP"), (), (), (b"licd",),
        (),
    ),
    (
        b"FORMYLTRANSFERASE", b"ForT", b"Modifier",
        rb"^(?:fmt[0-9]+|for[a-z0-9_]*)(?:_[0-9]+|[a-z0-9])?$",
        rb"formyl[\s-]?transferase|n-formyl",
        b"Formyltransferase; adds formyl (Fo) modifications.",
        False, (b"Fo",), (), (), (),
        (),
    ),
    (
        b"ALANINE_TRANSFERASE", b"Alt", b"Modifier",
        rb"^(?:alt[0-9]*|aat[0-9]*)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"alanine\s*(?:amide\s*)?transferase|\balt[0-9]*\b|aminoacyl[\s-]transferase",
        b"Alt1 alanine transferase; transfers D-alanine to the carboxyl group of GalA/GalNAcA forming an amide.",
        False, (b"DAla", b"LAla", b"Ala"), (b"HexA", b"HexNA"), (), (b"alt", b"alt1", b"ala_transferase"),
        (),
    ),
    (
        b"AMINOTRANSFERASE", b"ATn", b"PrecursorSynthesis",
        rb"^(?:wece|psec|legc|degt|qhbb|qnr[a-z0-9_]*|qdtb|psab|rhn[bc]|wei[pq])(?:_[0-9]+|[a-z0-9])?$",
        rb"amino[\s-]?transferase|transaminase|\bwece\b|\bpsec\b|\blegc\b|aminase|\brhn[bc]\b",
        b"Aminotransferase / Transaminase (e.g. WecE, RhnB, RhnC); converts keto-intermediates to amino sugars.",
        False, (b"GlcN", b"GalN", b"QuiN", b"QuiN4N", b"FucN", b"FucN4N", b"Pse", b"Leg", b"Bac", b"Aci", b"RhaN", b"RhaN3N"), (b"HexN", b"dHexN"), (), (b"wece", b"rhnb", b"rhnc"),
        (),
    ),
    (
        b"ALDOLASE", b"Ald", b"PrecursorSynthesis",
        rb"^(?:neub|psei|legi|kds[ab]|nana|psaf)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"\baldolase\b|\bcondensase\b|(?:sialic|pseudaminic|legionaminic|neuraminic|n-acetylneuraminic)[\s-]acid[\s-]?synthase|(?:neu5ac|n-acetylneuraminate)[\s-]?synthase|kdo(?:-8-phosphate)?[\s-]?synthase|\bneub\b|\bpsei\b|\blegi\b",
        b"PEP-condensing aldolase/synthase (e.g. NeuB, PseI, LegI); synthesizes nonulosonic acids.",
        False, (b"Neu5Ac", b"Pse", b"Leg", b"Aci", b"Kdo"), (b"HigherSugar", b"gPse", b"gLeg", b"g4eLeg", b"gAci", b"gNeu", b"NeuAc"), (), (b"neub", b"psei", b"legi"),
        (),
    ),
    (
        b"DEACETYLASE", b"DAc", b"Modifier",
        rb"^(?:pgab|pda[0-9a-z_]*|dac[a-z0-9_]*|ahy)(?:_[0-9]+|[a-z0-9])?$",
        rb"de-?acetylase|\bpgab\b|\bpdaa?\b|\bpda[0-9a-z]*\b|\bahy\b|acylhydrolase",
        b"Deacetylase; generates free amino sugar derivatives.",
        False, (b"GlcN", b"GalN", b"QuiN", b"QuiN4N", b"FucN", b"FucN4N", b"RhaN", b"ManN", b"GalNA", b"GlcNA", b"ManNA", b"GlcN3NA"), (), (), (b"pgab", b"pda1", b"pda2", b"pda2a", b"ahy"),
        (b"pda", b"ahy"),
    ),

    # Pathway & Biosynthesis Enzymes
    (
        b"EPIMERASE", b"Epi", b"PrecursorSynthesis",
        rb"^(?:gale|rmlc|rfbc|wbpp|neuc|wca[gi]|glf|manb|pgm|gpi|mnaa|cps[0-9]*g|rfbk|gne[0-9]*|gne|fdta|fdtb|kpsf|gutq|kdsd)(?:_[0-9]+|[a-z0-9])?$",
        rb"\bepimerase(?!\s*[/\\-]\s*dehydratase)\b|\bracemase\b|\bisomerase\b|ketoisomerase|(?:(?:udp-)?galacto(?:pyranose)?|phospho(?:manno|gluco|hexo)|(?:hexose|sugar)[\s-]phosphate)?\s*mutase|\bgale\b|\bwbpp\b|\brmlc\b|\bneuc\b|\bgpi\b|\bgne\b|dtdp-4-dehydrorhamnose\s*3|arabinose-5-phosphate\s*isomerase",
        b"Epimerase / Isomerase; interconverts UDP-sugar stereoisomers and precursor intermediates.",
        False, (), (b"Hex", b"HexNAc", b"DeoxyHex", b"DeoxyHexNAc"), (b"Mod",), (b"gale", b"rmlc", b"neuc", b"gne", b"rfbc", b"kpsf"),
        (b"rmlc", b"kpsf", b"gpi", b"pgm", b"rfbc"),
    ),
    (
        b"OXIDOREDUCTASE", b"OxR", b"PrecursorSynthesis",
        rb"^(?:rml[bd]|rfb[bd]|gmd|fcl|ugd|hasb|gna[0-9]*|fnr[0-9]*|rhna|weis|epab|epad|mnab)(?:_[0-9]+|[a-z0-9])?$",
        rb"dehydrogenase|reductase|dehydratase|oxidoreductase|epimer(?:ase|ise)/dehydratase|\brmlb\b|\brmld\b|\brhna\b|fucosamine biosynthesis protein|dtdp-glucose\s*4",
        b"Reductase / Dehydratase (e.g. RmlB, RhnA); synthesizes deoxyhexoses, dideoxyhexoses, and alditol reducing ends.",
        False, (b"Fuc", b"Rha", b"Qui", b"6dGul", b"6dAlt", b"6dTal", b"Oli", b"Tyv", b"Abe", b"Par", b"Dig", b"Col", b"ol", b"Gro", b"RhaN", b"RhaN3N"), (b"DeoxyHex", b"DeoxyHexNAc", b"DiDeoxyHex", b"dHexN"), (), (b"rmlb", b"rmld", b"rhna", b"epab", b"epad", b"rfbb", b"rfbd"),
        (b"rmlb", b"rmld", b"ugd", b"gnd", b"rfbb", b"rfbd"),
    ),
    (
        b"TRANSMEMBRANE_SYNTHASE", b"TMS", b"PolymerAssembly",
        rb"^(?:hasa|pgac|icaa|bcsa|alg8|cps3s)(?:_[0-9]+|[a-z0-9])?$",
        rb"(?:hyaluron(?:an|ic\s?acid)|cellulose|poly-.*glucosamine|poly-beta-1,6.*|alginate)[\s-]?(?:synthase|polymerase)|processive[\s-]?glycosyltransferase|type[\s-]3[\s-]?synthase|\bhasa\b|\bpgac\b|\bbcsa\b",
        b"Processive transmembrane glycosyltransferase (e.g. HasA, PgaC, BcsA); polymerizes and extrudes glycan chains.",
        True, (), (), (b"Mod",), (b"hasa", b"pgac", b"bcsa"),
        (),
    ),
    (
        b"NUCLEOTIDYLTRANSFERASE", b"NT", b"PrecursorSynthesis",
        rb"^(?:rmla|rfba|galu|neua|glmm|glmu|man[bc]|hasc|wbp[vw]|psac|lgad|lgag|lgah|epaa|epac)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"(?:nucleotidyl|thymidylyl|uridylyl|cytidylyl|guanylyl|adenylyl)[\s-]?(?:transferase|diphosphorylase)|pyrophosphorylase|\brmla\b|\brfba\b|\bgalu\b|\bneua\b|\bglmm\b|\bglmu\b|\bcytidyltransferase\b|synthetase; involved in synthesis of legionaminic",
        b"Nucleotide-sugar precursor synthase; acts on base monosaccharide pathways.",
        False, (), (), (b"Mod",), (b"rmla", b"galu", b"epaa", b"epac"),
        (b"rmla", b"galu", b"galf"),
    ),
    (
        b"NUCLEOTIDASE", b"Nuc", b"PrecursorSynthesis",
        rb"^(?:psa[dgi]|lgaf|gmm|wcah)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        rb"\bnucleotidase\b|sugar[\s-]nucleotide\s*hydrolase|gdp-mannose\s*(?:mannosyl\s*)?hydrolase",
        b"Sugar-nucleotide nucleotidase / hydrolase (e.g. PsaD, PsaG, PsaI, LgaF, Gmm); cleaves sugar-nucleotide intermediates.",
        False, (b"Pse", b"Leg", b"Aci", b"Neu5Ac", b"Man"), (b"HigherSugar", b"gPse", b"gLeg", b"g4eLeg", b"gAci", b"gNeu", b"NeuAc"), (), (b"psad", b"psag", b"psai", b"lgaf", b"gmm", b"wcah", b"sugar_phosphatase"),
        (b"gmm", b"wcah"),
    ),
    (
        b"INITIATOR", b"Init", b"PolymerAssembly",
        rb"^(?:wbap|wcaj|itr[ab]|tago|mray|cps[0-9]*a|wcfs|weca|wcha)(?:_[0-9]+|[a-z0-9])?$",
        rb"(?:initial|initiating).*transferase|undecaprenyl-phosphate.*transferase|undecaprenyl.*phosphotransferase|polyisoprenyl-teichoic.*transferase|glucosyl[\s-]?(?:1-)?phosphate[\s-]?transferase|\bwcaj\b|\bwbap\b|\bitra\b|\bitrb\b|\bwcha\b",
        b"Initial sugar transferase (e.g. WbaP/WcaJ/WchA); attaches first non-modifier sugar onto undecaprenyl phosphate.",
        False, (), (), (b"Mod",), (b"wbap", b"wcaj", b"wcha", b"initiating_gt"),
        (b"wbap", b"wcaj", b"wcha", b"itra", b"itrb", b"weca"),
    ),

    # Broad/Generic Enzymes
    (
        b"GLYCOSYLHYDROLASE", b"GH", b"Modifier",
        rb"^(?:gh[0-9]+|ghl[0-9]*|gnlb|cps[0-9]*g|orf1\(ghy\)|ghy)(?:_[0-9]+|[a-z0-9])?$",
        rb"glycosyl[\s-]?hydrolase|(?:gluc|galact|mann)osidase|\bheparinase\b|\bgnlb\b|\blyase\b|alginate\s*lyase|pectate\s*lyase|depolymerase",
        b"Glycosylhydrolase / lyase; processes, depolymerizes, or cleaves glycosidic linkages.",
        False, (), (), (), (b"gnlb", b"ghl", b"orf1(ghy)", b"ghy"),
        (b"ghl", b"orf1(ghy)", b"ghy"),
    ),
    (
        b"GLYCOSYLTRANSFERASE", b"GT", b"PolymerAssembly",
        rb"^(?:wb[dsyz][a-z]|gtr[0-9a-z]*|gtroc[0-9]*|glt[0-9]*|gt[0-9]+|wf[efg][a-z]|kfo[a-z]|kfi[ac]|kpss[0-9]*|kpss|htrl[0-9a-z*]*|sps[a-z]|wbm[vw]|gml[bc]|rhamnan|rgpf)(?:_[0-9]+|[a-z0-9])?$",
        rb"(?:glyco(?:syl)?|sugar|sialyl|polysialyl|kdo|(?:gluc|galact(?:ofuran)?|mann|fuc|rhamn|glucuron|galacturon|xyl)osyl|(?:glucos|galactos|mannos)aminyl)[\s-]?transferase|oct-2-ulosonic\s*acid\s*transferase|pseudaminyltransferase|\bhtrl\b|wca[al]\b|rhamnan\s*synthesis\s*protein",
        b"Glycosyltransferase; synthesizes glycosidic linkages and non-modifier monosaccharide additions.",
        True, (), (), (b"Mod",), (b"kpss", b"gtr", b"glt", b"htrl", b"wbmv", b"wbmw", b"gmlc", b"spsa", b"wcaa", b"rhamnan", b"rgpf"),
        (b"gtr", b"htrl", b"wbmv", b"wbmw", b"gmlc", b"spsa", b"wcaa", b"rhamnan", b"rgpf"),
    ),

    # Mobile Genetic Elements
    (
        b"TRANSPOSASE", b"Tnp", b"MobileElement",
        rb"^(?:tnp[0-9a-z_]*|is[0-9a-z_]+|ins[0-9a-z_]*)(?:_[0-9]+|[a-z0-9])?$",
        rb"\btransposase\b|\btransposon\b|transposase-like|\binsertion\s*sequence\b|\bis[0-9a-z_]+(?:\s*family)?\s*(?:transposase|element)?\b|\bis[0-9a-z_]+\s*orf\b",
        b"Transposase / Insertion sequence (IS) element; mobile genetic element non-biosynthetic infiltrator.",
        False, (), (), (), (b"tnp", b"transposase", b"is_element", b"insertion_sequence"),
        (b"tnp",),
    ),
    (
        b"INTEGRASE", b"Int", b"MobileElement",
        rb"^(?:int[0-9a-z_]*|xis[0-9a-z_]*|rec[0-9a-z_]*)(?:_[0-9]+|[a-z0-9])?$",
        rb"\bintegrase\b|\brecombinase\b|\bexcisionase\b|phage\s*integrase|site-specific\s*recombinase|\bresolvase\b",
        b"Integrase / Recombinase / Excisionase; mediates DNA integration/excision, prophage insertions, or mobile cassette movement.",
        False, (), (), (), (b"int", b"integrase", b"recombinase", b"excisionase"),
        (),
    ),

    # Uncharacterized / Background
    (
        b"UNKNOWN", b"?", b"Background",
        rb"^$",
        rb"^$",
        b"Uncharacterized gene; unconstrained matching to any residue or linkage.",
        False, (), (b"Hex", b"HexNAc", b"DeoxyHex", b"DeoxyHexNAc", b"Pentose", b"DiDeoxyHex", b"Mod", b"HigherSugar", b"Generic"), (), (),
        (),
    ),
]


# ----------------------------------------------------------------------------------------------------------------------
# Vectorized SoA GeneRoles Database Class
# ----------------------------------------------------------------------------------------------------------------------

class GeneRoles(Catalog):
    """
    State-less, vectorized Structure-of-Arrays (SoA) db container for BGC gene roles.
    Supports integer and bytes indexing (GeneRoles[b"INITIATOR"]), fast single-pass regex classification,
    and alignment rule compilation.
    """
    _CANONICAL_MANIFEST = _GENE_DATA
    _SOA_ARRAYS = (
        '_names',
        '_abbreviations',
        '_categories',
        '_matches_linkages',
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Vectorized SoA Array Fields with Strict Dimensional & Dtype Type Hints
    # ------------------------------------------------------------------------------------------------------------------
    _names: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    _abbreviations: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    _categories: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    _matches_linkages: np.ndarray[tuple[int], np.dtype[np.bool_]]

    name_to_index: dict[bytes, int]
    abbreviation_to_index: dict[bytes, int]
    abbreviation_map: dict[bytes, bytes]
    role_abbreviations: dict[bytes, bytes]
    alias_map: dict[bytes, bytes]

    _symbol_pattern: re.Pattern[bytes] | None
    _desc_pattern: re.Pattern[bytes] | None
    _group_to_index: dict[str, int]
    _trusted_prefixes: tuple[bytes, ...]
    _trusted_map: dict[bytes, int]

    _target_types: tuple[TargetType, ...]
    _target_modifiers: tuple[tuple[bytes, ...], ...]
    _allowed_names: tuple[tuple[bytes, ...], ...]
    _allowed_classes: tuple[tuple[bytes, ...], ...]
    _disallowed_classes: tuple[tuple[bytes, ...], ...]
    _descriptions: tuple[bytes, ...]
    _symbol_regexes: tuple[bytes, ...]
    _desc_regexes: tuple[bytes, ...]
    _aliases: tuple[tuple[bytes, ...], ...]
    _role_prefixes: tuple[tuple[bytes, ...], ...]

    def init_default(self) -> None:
        self._init_from_manifest(self._CANONICAL_MANIFEST)

    def _init_from_manifest(self, manifest: Sequence[tuple]) -> None:
        N = len(manifest)
        self._names = np.empty(N, dtype="S32")
        self._abbreviations = np.empty(N, dtype="S8")
        self._categories = np.empty(N, dtype="S24")
        self._matches_linkages = np.empty(N, dtype=np.bool_)

        self.name_to_index = {}
        self.abbreviation_to_index = {}
        self.abbreviation_map = {}
        self.role_abbreviations = {}
        self.alias_map = {}
        self._group_to_index = {}
        all_trusted: list[tuple[bytes, int]] = []

        target_types_list: list[TargetType] = []
        target_modifiers_list: list[tuple[bytes, ...]] = []
        allowed_names_list: list[tuple[bytes, ...]] = []
        allowed_classes_list: list[tuple[bytes, ...]] = []
        disallowed_classes_list: list[tuple[bytes, ...]] = []
        descriptions_list: list[bytes] = []
        symbol_regexes_list: list[bytes] = []
        desc_regexes_list: list[bytes] = []
        aliases_list: list[tuple[bytes, ...]] = []
        role_prefixes_list: list[tuple[bytes, ...]] = []

        for i, row in enumerate(manifest):
            (
                name, abbreviation, category, sym_rx, desc_rx, desc,
                matches_link, allowed_names, allowed_classes, disallowed_classes,
                aliases, trusted_prefixes
            ) = row

            self._names[i] = name
            self._abbreviations[i] = abbreviation
            self._categories[i] = category
            self._matches_linkages[i] = matches_link

            self.name_to_index[name] = i
            self.abbreviation_to_index[abbreviation] = i
            self.abbreviation_to_index[abbreviation.upper()] = i
            self.abbreviation_to_index[abbreviation.lower()] = i

            self.abbreviation_map[abbreviation] = name
            self.abbreviation_map[abbreviation.lower()] = name
            self.role_abbreviations[name] = abbreviation

            rp_list = []
            for p in trusted_prefixes:
                bp = bytes(p).strip().lower()
                all_trusted.append((bp, i))
                rp_list.append(bp)
            role_prefixes_list.append(tuple(rp_list))

            if name == b"INITIATOR":
                target_type = TargetType.REDUCING_END
                target_modifiers = ()
            elif name in (
                b"ACETYLTRANSFERASE", b"PYRUVYLTRANSFERASE", b"METHYLTRANSFERASE",
                b"PHOSPHOTRANSFERASE", b"FORMYLTRANSFERASE", b"ALANINE_TRANSFERASE", b"DEACETYLASE"
            ) or category == b"Modifier":
                target_type = TargetType.MODIFIED_MONO
                target_modifiers = tuple(allowed_names)
            elif name in (b"GLYCOSYLTRANSFERASE", b"POLYMERASE", b"TRANSMEMBRANE_SYNTHASE") or matches_link:
                target_type = TargetType.LINKAGE
                target_modifiers = ()
            elif category == b"PrecursorSynthesis":
                target_type = TargetType.MONOSACCHARIDE
                target_modifiers = ()
            elif category in (b"Transport", b"MobileElement", b"Background") or name in (
                b"TRANSPOSASE", b"INTEGRASE", b"LIGASE", b"UNKNOWN",
                b"FLIPPASE", b"COPOLYMERASE", b"KINASE", b"PHOSPHATASE",
                b"ABC_PERMEASE", b"ABC_ATPASE", b"TRANSLOCON", b"CAPSULE_ANCHOR"
            ):
                target_type = TargetType.NONE
                target_modifiers = ()
            else:
                target_type = TargetType.NONE
                target_modifiers = ()

            target_types_list.append(target_type)
            target_modifiers_list.append(target_modifiers)
            allowed_names_list.append(tuple(allowed_names))
            allowed_classes_list.append(tuple(allowed_classes))
            disallowed_classes_list.append(tuple(disallowed_classes))
            descriptions_list.append(desc)
            symbol_regexes_list.append(sym_rx)
            desc_regexes_list.append(desc_rx)
            aliases_list.append(tuple(aliases))

        self._target_types = tuple(target_types_list)
        self._target_modifiers = tuple(target_modifiers_list)
        self._allowed_names = tuple(allowed_names_list)
        self._allowed_classes = tuple(allowed_classes_list)
        self._disallowed_classes = tuple(disallowed_classes_list)
        self._descriptions = tuple(descriptions_list)
        self._symbol_regexes = tuple(symbol_regexes_list)
        self._desc_regexes = tuple(desc_regexes_list)
        self._aliases = tuple(aliases_list)
        self._role_prefixes = tuple(role_prefixes_list)

        self._feature_matrix = None
        sorted_trusted = sorted(all_trusted, key=lambda item: len(item[0]), reverse=True)
        self._trusted_prefixes = tuple(p for p, _ in sorted_trusted)
        self._trusted_map = {p: idx for p, idx in sorted_trusted}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compiles named-group regex patterns and caches string group-to-index mapping for zero-overhead classification."""
        sym_parts = []
        desc_parts = []
        self._group_to_index = {}
        for i in range(len(self._names)):
            name = self._names[i]
            self._group_to_index[name.decode("ascii")] = i
            if name == b"UNKNOWN":
                continue
            sym_parts.append(rb"(?P<" + name + rb">" + self._symbol_regexes[i] + rb")")
            desc_parts.append(rb"(?P<" + name + rb">" + self._desc_regexes[i] + rb")")

        self._symbol_pattern = re.compile(b"|".join(sym_parts), re.IGNORECASE) if sym_parts else None
        self._desc_pattern = re.compile(b"|".join(desc_parts), re.IGNORECASE) if desc_parts else None

    # ------------------------------------------------------------------------------------------------------------------
    # Vectorized Array Properties
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def names(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        return self._names

    @property
    def abbreviations(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        return self._abbreviations

    @property
    def categories(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        return self._categories

    @property
    def matches_linkages(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        return self._matches_linkages

    @property
    def trusted_prefixes(self) -> tuple[bytes, ...]:
        return self._trusted_prefixes

    @property
    def trusted_map(self) -> dict[bytes, int]:
        return self._trusted_map

    @property
    def target_types(self) -> tuple[TargetType, ...]:
        return self._target_types

    @property
    def target_modifiers(self) -> tuple[tuple[bytes, ...], ...]:
        return self._target_modifiers

    @property
    def allowed_names(self) -> tuple[tuple[bytes, ...], ...]:
        return self._allowed_names

    @property
    def allowed_classes(self) -> tuple[tuple[bytes, ...], ...]:
        return self._allowed_classes

    @property
    def disallowed_classes(self) -> tuple[tuple[bytes, ...], ...]:
        return self._disallowed_classes

    @property
    def descriptions(self) -> tuple[bytes, ...]:
        return self._descriptions

    @property
    def aliases(self) -> tuple[tuple[bytes, ...], ...]:
        return self._aliases

    @property
    def role_trusted_prefixes(self) -> tuple[tuple[bytes, ...], ...]:
        return self._role_prefixes

    def to_feature_matrix(self) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """
        Generates or returns cached (N, D) float32 feature matrix for embeddings.
        Features include one-hot encoded categories and boolean flags.
        """
        if self._feature_matrix is not None:
            return self._feature_matrix

        unique_cats = sorted(set(self._categories))
        cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        N = len(self._categories)
        D = len(unique_cats) + 1

        features = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            cat_idx = cat_to_idx[self._categories[i]]
            features[i, cat_idx] = 1.0
            features[i, -1] = 1.0 if self._matches_linkages[i] else 0.0

        self._feature_matrix = features
        return features

    # ------------------------------------------------------------------------------------------------------------------
    # Lookups & Overrides
    # ------------------------------------------------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _slice(self, key: slice | np.ndarray[tuple[int], np.dtype[Any]] | list[int]) -> Self:
        sub = super()._slice(key)
        sub._trusted_prefixes = self._trusted_prefixes
        sub._trusted_map = self._trusted_map
        sub.abbreviation_to_index = self.abbreviation_to_index
        sub.abbreviation_map = self.abbreviation_map
        sub.role_abbreviations = self.role_abbreviations
        sub._symbol_regexes = self._symbol_regexes
        sub._desc_regexes = self._desc_regexes
        sub._target_types = self._target_types
        sub._target_modifiers = self._target_modifiers
        sub._allowed_names = self._allowed_names
        sub._allowed_classes = self._allowed_classes
        sub._disallowed_classes = self._disallowed_classes
        sub._descriptions = self._descriptions
        sub._aliases = self._aliases
        sub._role_prefixes = self._role_prefixes
        sub._compile_patterns()
        return sub


    # ------------------------------------------------------------------------------------------------------------------
    # Single-Pass C-Speed Classification
    # ------------------------------------------------------------------------------------------------------------------
    def _classify_one(
        self,
        symbol_clean: bytes,
        desc_clean: bytes,
        unknown_idx: int,
    ) -> int:
        """Core 4-tier single-item classification engine."""
        # Tier 1: Canonical Trusted Core Symbol Prefixes (fast O(1) prefix check)
        if symbol_clean and symbol_clean.startswith(self._trusted_prefixes):
            for p in self._trusted_prefixes:
                if symbol_clean.startswith(p):
                    return self._trusted_map[p]

        # Tier 2: Informative Product Description Regex (resolves variable enzymes by verified biochemical activity)
        if desc_clean and self._desc_pattern is not None:
            match = self._desc_pattern.search(desc_clean)
            if match and match.lastgroup:
                return self._group_to_index.get(match.lastgroup, unknown_idx)

        # Tier 3: Symbol Regex & Aliases Fallback (for unannotated or missing descriptions)
        if symbol_clean:
            if symbol_clean in self.alias_map:
                return self.name_to_index[self.alias_map[symbol_clean]]
            if self._symbol_pattern is not None:
                match = self._symbol_pattern.search(symbol_clean)
                if match and match.lastgroup:
                    return self._group_to_index.get(match.lastgroup, unknown_idx)
            stripped = _LOCUS_SUFFIX_PATTERN.sub(b"", symbol_clean)
            if stripped != symbol_clean and stripped:
                if stripped in self.alias_map:
                    return self.name_to_index[self.alias_map[stripped]]
                if self._symbol_pattern is not None:
                    match = self._symbol_pattern.search(stripped)
                    if match and match.lastgroup:
                        return self._group_to_index.get(match.lastgroup, unknown_idx)

        # Tier 4: Fallback for empty symbol when description contains candidate gene identifier
        if not symbol_clean and desc_clean:
            candidate = desc_clean.split()[0].rstrip(b"*")
            if candidate.startswith(self._trusted_prefixes):
                for p in self._trusted_prefixes:
                    if candidate.startswith(p):
                        return self._trusted_map[p]
            if candidate in self.alias_map:
                return self.name_to_index[self.alias_map[candidate]]
            if self._symbol_pattern is not None:
                match = self._symbol_pattern.search(candidate)
                if match and match.lastgroup:
                    return self._group_to_index.get(match.lastgroup, unknown_idx)
                stripped = _LOCUS_SUFFIX_PATTERN.sub(b"", candidate)
                if stripped != candidate and stripped:
                    if stripped in self.alias_map:
                        return self.name_to_index[self.alias_map[stripped]]
                    if self._symbol_pattern is not None:
                        m_strip = self._symbol_pattern.search(stripped)
                        if m_strip and m_strip.lastgroup:
                            return self._group_to_index.get(m_strip.lastgroup, unknown_idx)

        return unknown_idx

    def label(
        self,
        symbol: bytes | str | Sequence[bytes | str] | np.ndarray | None = None,
        description: bytes | str | Sequence[bytes | str] | np.ndarray | None = None,
    ) -> int | np.ndarray[tuple[int], np.dtype[np.int8]]:
        """Assigns gene role integer indices for a single gene or batch of genes using the 4-tier strategy:

        1. Tier 1: Fast O(1) canonical trusted core symbol prefixes (immune to submitter typos).
        2. Tier 2: Informative product description regex (prioritizes verified biochemical activity).
        3. Tier 3: Symbol regex and aliases fallback (with locus suffix stripping).
        4. Tier 4: Gene identifier extraction from product description fallback.

        Returns:
            int if inputs are scalar (or single symbol/description).
            np.ndarray[int8] if inputs are a sequence or array of symbols/descriptions.
        """
        sym = symbol
        desc = description
        unknown_idx = self.name_to_index[b"UNKNOWN"]

        # Fast scalar path
        if isinstance(sym, (bytes, bytearray, str)) or (sym is None and isinstance(desc, (bytes, bytearray, str))):
            s = (sym.encode("ascii") if isinstance(sym, str) else bytes(sym)).strip().lower() if sym else b""
            d = (desc.encode("utf-8") if isinstance(desc, str) else bytes(desc)).strip().lower() if desc else b""
            return self._classify_one(s, d, unknown_idx)

        # Batch vectorized path
        if sym is not None:
            sym_arr = sym if isinstance(sym, (list, tuple, np.ndarray)) else list(sym)
            N = len(sym_arr)
        elif desc is not None:
            desc_arr = desc if isinstance(desc, (list, tuple, np.ndarray)) else list(desc)
            N = len(desc_arr)
            sym_arr = ()
        else:
            return unknown_idx

        if N == 0:
            return np.empty(0, dtype=np.int8)

        out = np.empty(N, dtype=np.int8)
        classify = self._classify_one
        has_sym = len(sym_arr) > 0

        if desc is None:
            for i in range(N):
                s_raw = sym_arr[i]
                if s_raw:
                    s = (s_raw.encode("ascii") if isinstance(s_raw, str) else bytes(s_raw)).strip().lower()
                else:
                    s = b""
                out[i] = classify(s, b"", unknown_idx)
        else:
            desc_arr = desc if isinstance(desc, (list, tuple, np.ndarray)) else list(desc)
            if len(desc_arr) != N:
                raise ValueError(f"descriptions length {len(desc_arr)} does not match symbols length {N}")
            for i in range(N):
                s_raw = sym_arr[i] if has_sym else None
                d_raw = desc_arr[i]
                if s_raw:
                    s = (s_raw.encode("ascii") if isinstance(s_raw, str) else bytes(s_raw)).strip().lower()
                else:
                    s = b""
                if d_raw:
                    d = (d_raw.encode("utf-8") if isinstance(d_raw, str) else bytes(d_raw)).strip().lower()
                else:
                    d = b""
                out[i] = classify(s, d, unknown_idx)

        return out


