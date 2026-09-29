from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
import re
import warnings

from kbank.glycan.glycan import Anomer, Chirality, Glycan, LinkageEdge, LinkageType, ResidueType
from kbank.glycan.sugars import Sugars


# Errors and warnings --------------------------------------------------------------------------------------------------
class GlycanParseError(ValueError):
    """Raised when a glycan string violates structural syntax and cannot be parsed."""
    pass


class GlycanParseWarning(UserWarning):
    """Raised when the parser employs heuristics or makes assumptions about an ambiguous structure."""
    pass


# Enums ----------------------------------------------------------------------------------------------------------------
class TokenType(Enum):
    BRANCH_START = auto()
    BRANCH_END = auto()
    LINKAGE = auto()
    SUGAR = auto()


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Token:
    type: TokenType
    value: bytes
    pos: int

@dataclass(slots=True, frozen=True)
class Substitution:
    modifier: int
    carbons: tuple[int, ...] = (0,)
    stoichiometry: float = 1.0


# Parsers --------------------------------------------------------------------------------------------------------------
class GlycanParser(ABC):
    _RE_LEXER = re.compile(br"\]|\[|\([^)]+\)|[^\[\]()]+")

    def __init__(self):
        self.mono_map = dict(Sugars.name_to_index)
        for k, v in Sugars.alias_map.items():
            if v in self.mono_map:
                self.mono_map[k] = self.mono_map[v]

        self.mod_map = {
            b"Ac": self.mono_map[b"OAc"],
            b"OAc": self.mono_map[b"OAc"],
            b"Pyr": self.mono_map[b"Pyr"],
            b"P": self.mono_map[b"P"],
            b"PEtn": self.mono_map[b"PEtn"],
            b"OMe": self.mono_map[b"OMe"],
            b"Me": self.mono_map[b"OMe"],
            b"N": self.mono_map[b"N"],
            b"LAla": self.mono_map[b"LAla"],
            b"DAla": self.mono_map[b"DAla"],
            b"dAla": self.mono_map[b"DAla"],
            b"Ala": self.mono_map[b"Ala"],
            b"Fo": self.mono_map[b"Fo"],
            b"Lac": self.mono_map[b"Lac"],
            b"Gro": self.mono_map[b"Gro"],
            b"ol": self.mono_map[b"ol"],
        }
        
        mono_patterns = sorted(self.mono_map.keys(), key=len, reverse=True)
        mod_patterns = sorted(self.mod_map.keys(), key=len, reverse=True)
        
        self._RE_MOD = re.compile(br"^(" + b"|".join(mod_patterns) + br")")
        self._RE_MOD_SEARCH = re.compile(br"(" + b"|".join(mod_patterns) + br")")
        self._RE_MONO = re.compile(br"^(" + b"|".join(mono_patterns) + br")")

    def _lex(self, seq: bytes) -> Iterator[Token]:
        tokens = []
        for match in self._RE_LEXER.finditer(seq):
            val = match.group(0)
            pos = match.start()
            if val == b"]":
                tokens.append(Token(TokenType.BRANCH_START, val, pos))
            elif val == b"[":
                tokens.append(Token(TokenType.BRANCH_END, val, pos))
            elif val.startswith(b"("):
                tokens.append(Token(TokenType.LINKAGE, val[1:-1], pos))
            else:
                tokens.append(Token(TokenType.SUGAR, val.strip(), pos))
        yield from reversed(tokens)

    @abstractmethod
    def parse(self, sequence: bytes, strict: bool = False) -> Glycan:
        """Parse a canonical glycan sequence into a Glycan core model."""
        raise NotImplementedError


class IUPACParser(GlycanParser):
    @staticmethod
    def _parse_linkage(link: bytes) -> LinkageType:
        try:
            return LinkageType.from_bytes(link)
        except ValueError as e:
            raise GlycanParseError(str(e)) from e

    def _parse_monosaccharide(self, sugar: bytes) -> tuple[ResidueType, list[Substitution]]:
        sugar = sugar.strip(b"-")
        chirality = Chirality.UNKNOWN
        
        # Handle stereodescriptor/anomeric prefixes (e.g. b-D-, a-L-, D-, L-, b-, a-)
        if sugar.startswith((b"D-", b"d-")):
            chirality = Chirality.D
            sugar = sugar[2:]
        elif sugar.startswith((b"L-", b"l-")):
            chirality = Chirality.L
            sugar = sugar[2:]
        elif len(sugar) > 3 and sugar[0] in (b"a"[0], b"b"[0], b"?"[0]) and sugar[1] == b"-"[0]:
            sugar = sugar[2:]
            if sugar.startswith((b"D-", b"d-")):
                chirality = Chirality.D
                sugar = sugar[2:]
            elif sugar.startswith((b"L-", b"l-")):
                chirality = Chirality.L
                sugar = sugar[2:]

        mono_match = self._RE_MONO.match(sugar)
        if not mono_match:
            raise GlycanParseError(f"Unknown sugar base in: {sugar}")
        base_str = mono_match.group(1)
        sugar_idx = self.mono_map[base_str]
        is_furanose = base_str.endswith(b'f')
        if chirality == Chirality.UNKNOWN:
            chirality = Chirality(Sugars.natural_chirality[sugar_idx])
        base_mono = ResidueType.from_sugar(sugar_idx, chirality, is_furanose)
        
        remainder = sugar[mono_match.end():]
        modifications = []
        if remainder:
            idx = 0
            carbons = []
            while idx < len(remainder):
                c_byte = remainder[idx]
                if b'1'[0] <= c_byte <= b'9'[0] or c_byte == b'?'[0]:
                    carbons.append(c_byte - 48 if b'1'[0] <= c_byte <= b'9'[0] else 0)
                    idx += 1
                elif c_byte == b','[0]:
                    idx += 1
                else:
                    mod_match = self._RE_MOD.match(remainder[idx:])
                    if not mod_match:
                        raise GlycanParseError(f"Unknown mod at idx {idx} in {remainder}")
                    matched_mod = self.mod_map[mod_match.group(1)]
                    modifications.append(Substitution(
                        modifier=matched_mod,
                        carbons=tuple(carbons) if carbons else (0,)
                    ))
                    idx += mod_match.end()
                    carbons = []
        return base_mono, modifications

    def parse(self, sequence: bytes, strict: bool = False) -> Glycan:
        if not isinstance(sequence, (bytes, bytearray)):
            raise TypeError(f"sequence must be bytes, got {type(sequence).__name__}")
        sequence = bytes(sequence)

        monosaccharides = []
        edges = []
        stack = []
        active_node_id: int | None = None
        pending_linkage: LinkageType | None = None

        for token in self._lex(sequence):
            if token.type == TokenType.BRANCH_START:
                if active_node_id is None:
                    raise GlycanParseError(f"Dangling branch start ']' at {token.pos}")
                stack.append(active_node_id)
            elif token.type == TokenType.BRANCH_END:
                if not stack:
                    raise GlycanParseError(f"Unmatched '[' at {token.pos}")
                active_node_id = stack.pop()
            elif token.type == TokenType.LINKAGE:
                pending_linkage = self._parse_linkage(token.value)
            elif token.type == TokenType.SUGAR:
                monosaccharide, subs = self._parse_monosaccharide(token.value)
                new_node_id = len(monosaccharides)
                if new_node_id == 0:
                    monosaccharide = monosaccharide.with_reducing_end()
                monosaccharides.append(monosaccharide)
                for sub in subs:
                    mod_node_id = len(monosaccharides)
                    monosaccharides.append(ResidueType.from_sugar(sub.modifier, Chirality.UNKNOWN, False))
                    for carbon in sub.carbons:
                        edges.append(LinkageEdge(source=mod_node_id, target=new_node_id, edge_type=LinkageType(Anomer.UNKNOWN, donor=0, acceptor=carbon)))
                
                if active_node_id is not None and pending_linkage:
                    edges.append(LinkageEdge(source=new_node_id, target=active_node_id, edge_type=pending_linkage))
                active_node_id = new_node_id
                pending_linkage = None

        if stack:
            raise GlycanParseError("Unmatched ']' in sequence (stack not empty)")
            
        return Glycan(canonical=sequence, nodes=tuple(monosaccharides), edges=tuple(edges))


class CSDBParser(GlycanParser):
    _RE_POLYMER_DONOR = re.compile(br"\(([0-9?]+)-?(P)?-?\s*$")
    _RE_POLYMER_ACCEPTOR = re.compile(br"^\s*-([0-9?P-]+)\)")
    _RE_PERCENTAGES = re.compile(br"\d*%+")
    _RE_FUZZY_OR = re.compile(br"<<([^|>]+)[^>]*>>")
    _RE_DANGLING_CAP = re.compile(br"^-\d+\)")
    _RE_COMMENT = re.compile(br"\s*//.*$")
    _RE_NEU = re.compile(br"^(Neu)([pf]?)")
    _RE_LIPID_HEURISTIC = re.compile(br"(Myr|Lau|Dco|Ste|But|Cap|CER|Cer|LI|3HO|Vac|ALK|Und|lX|l\?|Gc|Et)")
    _RE_AMINOACID_HEURISTIC = re.compile(br"(Ala|Ser|Thr|Gly|Lys|Asp|Glu)")

    def _strip(self, sequence: bytes) -> tuple[bytes, int | None, int | None]:
        seq = sequence
        seq = self._RE_COMMENT.sub(b"", seq).strip()
        seq = self._RE_PERCENTAGES.sub(b"", seq)
        seq = self._RE_FUZZY_OR.sub(br"\1", seq)
        
        donor = None
        donor_match = self._RE_POLYMER_DONOR.search(seq)
        if donor_match:
            d_chars = donor_match.group(1)
            donor = (d_chars[0] - 48) if (d_chars and 49 <= d_chars[0] <= 57) else 0
            seq = seq[:donor_match.start()].strip()
            
        acceptor = None
        acc_match = self._RE_POLYMER_ACCEPTOR.search(seq)
        if acc_match:
            a_chars = acc_match.group(1)
            acceptor = (a_chars[0] - 48) if (a_chars and 49 <= a_chars[0] <= 57) else 0
            seq = seq[acc_match.end():].strip()
            
        # Clever comma replace
        res = bytearray()
        parens = curlies = 0
        for i, byte in enumerate(seq):
            if byte == 40: parens += 1
            elif byte == 41: parens = max(0, parens - 1)
            elif byte == 123: curlies += 1
            elif byte == 125: curlies = max(0, curlies - 1)
            elif byte == 44 and parens == 0 and curlies == 0:
                is_bounded = False
                if 0 < i < len(seq) - 1:
                    if 48 <= seq[i-1] <= 57 and 48 <= seq[i+1] <= 57:
                        is_bounded = True
                if not is_bounded:
                    res.extend(b"][")
                    continue
            res.append(byte)
        seq = bytes(res)
        
        return seq, donor, acceptor

    @staticmethod
    def _parse_linkage(link: bytes) -> tuple[int, tuple[int, ...]]:
        def _parse_c(b_char: int) -> int:
            return (b_char - 48) if 49 <= b_char <= 57 else 0

        if b':' in link:
            parts = link.split(b':')
            donor = _parse_c(parts[0].split(b'-')[0][0])
            c1 = _parse_c(parts[0].split(b'-')[1][0])
            c2 = _parse_c(parts[1].split(b'-')[1][0])
            return donor, (c1, c2)
        donor = _parse_c(link[0])
        acceptor = _parse_c(link[-1])
        return donor, (acceptor,)

    def _parse_monosaccharide(self, node: bytes, strict: bool = False) -> tuple[Anomer, ResidueType | None, list[Substitution]]:
        if node.startswith(b"*"):
            node = node.lstrip(b"*")
        if node.startswith(b"-"):
            node = node.lstrip(b"-")
        subs = []
        mod_match = self._RE_MOD_SEARCH.search(node)
        is_false_mod = False
        if mod_match:
            mod_str = mod_match.group(1)
            prefix = node[:mod_match.start()]
            if mod_str == b"N":
                is_false_mod = True
            elif mod_str in (b"Ac", b"OAc") and (b"Aci" in node or b"Leg" in node or b"Pse" in node or b"Bac" in node):
                is_false_mod = True
            elif mod_str == b"Gro" and len(prefix) <= 2:
                is_false_mod = True
            elif mod_str == b"ol" and (b"-ol" in node or b"ol" in node[len(prefix) - 2:]):
                is_false_mod = True
                
        if mod_match and not is_false_mod:
            mod_str = mod_match.group(1)
            prefix = node[:mod_match.start()]
            if not prefix or prefix.strip(b"ab?DxXRSldrs ") == b"":
                return Anomer.UNKNOWN, None, [Substitution(self.mod_map[mod_str])]
            subs.append(Substitution(self.mod_map[mod_str]))
            node = prefix
            
        mono_match = self._RE_MONO.match(node)
        if mono_match:
            base_str = mono_match.group(1)
            if base_str in self.mono_map:
                base_mono = self.mono_map[base_str]
                nat_chi = Chirality(Sugars.natural_chirality[base_mono])
                return Anomer.UNKNOWN, ResidueType.from_sugar(base_mono, nat_chi, base_str.endswith(b'f')), subs
                
        anomer = Anomer.UNKNOWN
        chirality = Chirality.UNKNOWN
        idx = 0
        
        # 1. Anomer prefix (a, b, ?, x, or a?, b?)
        if idx < len(node) and node[idx] in (b'a'[0], b'b'[0], b'?'[0], b'x'[0]):
            if node[idx] == b'a'[0]: anomer = Anomer.ALPHA
            elif node[idx] == b'b'[0]: anomer = Anomer.BETA
            idx += 1
            if idx < len(node) and node[idx] == b'?'[0] and (idx + 1 < len(node) and not (48 <= node[idx+1] <= 57)):
                idx += 1
            if idx < len(node) and node[idx] == b'-'[0]:
                idx += 1
            
        # 2. Chirality prefix (D, L, l, d, X, x, R, S, r, s, and possible trailing ?)
        chi_chars = (b'D'[0], b'L'[0], b'l'[0], b'd'[0], b'?'[0], b'x'[0], b'X'[0], b'R'[0], b'S'[0], b'r'[0], b's'[0])
        if idx < len(node) and node[idx] in chi_chars and not (node[idx] in (b'R'[0], b'r'[0]) and node[idx:].lower().startswith((b"rib", b"rha"))):
            first_chi = node[idx]
            if first_chi in (b'D'[0], b'd'[0]): chirality = Chirality.D
            elif first_chi in (b'L'[0], b'l'[0]): chirality = Chirality.L
            idx += 1
            # Check for compound acyl stereodescriptors (e.g. lS, lR, lX, rS, rR, xS, xR)
            if first_chi in (b'l'[0], b'r'[0], b'x'[0], b'X'[0]):
                if idx < len(node) and node[idx] in (b'S'[0], b'R'[0], b's'[0], b'r'[0], b'X'[0], b'x'[0], b'?'[0]):
                    if node[idx] in (b'S'[0], b's'[0]) and chirality == Chirality.UNKNOWN:
                        chirality = Chirality.L
                    elif node[idx] in (b'R'[0], b'r'[0]) and chirality == Chirality.UNKNOWN:
                        chirality = Chirality.D
                    idx += 1
            elif idx < len(node) and node[idx] == b'?'[0]:
                idx += 1
            if idx < len(node) and node[idx] == b'-'[0]:
                idx += 1
            
        core_sugar = node[idx:]
        core_sugar_clean = core_sugar.rstrip(b"?")
        
        # Handle 'N' modifier properly so it doesn't break 'Neu'
        if core_sugar.startswith(b"Neu"):
            mono_match = self._RE_NEU.match(core_sugar)
        else:
            mono_match = self._RE_MONO.match(core_sugar)
            if not mono_match and core_sugar_clean != core_sugar:
                mono_match = self._RE_MONO.match(core_sugar_clean)
            
        if not mono_match:
            if strict:
                raise GlycanParseError(f"Unknown CSDB sugar base: {core_sugar.decode(errors='replace')}")
            if self._RE_LIPID_HEURISTIC.search(core_sugar):
                sugar_idx = self.mono_map[b"Lipid"]
                generic_name = "Lipid"
            elif self._RE_AMINOACID_HEURISTIC.search(core_sugar):
                sugar_idx = self.mono_map[b"AminoAcid"]
                generic_name = "AminoAcid"
            else:
                sugar_idx = self.mono_map[b"ExoticSugar"]
                generic_name = "ExoticSugar"
            warnings.warn(f"Unknown CSDB sugar base '{core_sugar.decode(errors='replace')}'. "
                          f"Falling back to generic '{generic_name}'.", GlycanParseWarning)
        else:
            sugar_idx = self.mono_map[mono_match.group(1)]
        if chirality == Chirality.UNKNOWN:
            chirality = Chirality(Sugars.natural_chirality[sugar_idx])
            
        return anomer, ResidueType.from_sugar(sugar_idx, chirality, core_sugar.endswith(b'f')), subs

    def parse(self, sequence: bytes, strict: bool = False) -> Glycan:
        if not isinstance(sequence, (bytes, bytearray)):
            raise TypeError(f"sequence must be bytes, got {type(sequence).__name__}")
        sequence = bytes(sequence)

        clean_seq, polymer_donor, polymer_acceptor = self._strip(sequence)
        monosaccharides = []
        edges = []
        stack = []
        active_node_id: int | None = None
        pending_donor: int | None = None
        pending_acceptors: tuple[int, ...] | None = None
        root_anomer: Anomer | None = None

        for token in self._lex(clean_seq):
            if token.type == TokenType.BRANCH_START:
                if active_node_id is None:
                    raise GlycanParseError(f"Dangling ']' at {token.pos}")
                stack.append(active_node_id)
            elif token.type == TokenType.BRANCH_END:
                if not stack:
                    raise GlycanParseError(f"Unmatched '[' at {token.pos}")
                active_node_id = stack.pop()
            elif token.type == TokenType.LINKAGE:
                pending_donor, pending_acceptors = self._parse_linkage(token.value)
            elif token.type == TokenType.SUGAR:
                anomer, node_obj, subs = self._parse_monosaccharide(token.value, strict=strict)
                if node_obj is None:
                    if active_node_id is not None:
                        mod_carbons = pending_acceptors if pending_acceptors else (0,)
                        for sub in subs:
                            mod_node_id = len(monosaccharides)
                            monosaccharides.append(ResidueType.from_sugar(sub.modifier, Chirality.UNKNOWN, False))
                            for carbon in mod_carbons:
                                edges.append(LinkageEdge(source=mod_node_id, target=active_node_id,
                                                         edge_type=LinkageType(Anomer.UNKNOWN, pending_donor or 0, carbon)))
                        pending_donor = None
                        pending_acceptors = None
                else:
                    new_node_id = len(monosaccharides)
                    if new_node_id == 0:
                        node_obj = node_obj.with_reducing_end()
                    monosaccharides.append(node_obj)
                    if active_node_id is None:
                        root_anomer = anomer
                        
                    for sub in subs:
                        mod_node_id = len(monosaccharides)
                        monosaccharides.append(ResidueType.from_sugar(sub.modifier, Chirality.UNKNOWN, False))
                        for carbon in sub.carbons:
                            edges.append(LinkageEdge(source=mod_node_id, target=new_node_id,
                                                     edge_type=LinkageType(Anomer.UNKNOWN, 0, carbon)))
                            
                    if pending_donor is not None and pending_acceptors and active_node_id is not None:
                        link = LinkageType(anomer, donor=pending_donor, acceptor=pending_acceptors[0])
                        edges.append(LinkageEdge(source=new_node_id, target=active_node_id, edge_type=link))
                        for alternative_carbon in pending_acceptors[1:]:
                            alt_link = LinkageType(anomer, donor=pending_donor, acceptor=alternative_carbon)
                            edges.append(LinkageEdge(source=new_node_id, target=active_node_id, edge_type=alt_link))
                            
                    active_node_id = new_node_id
                    pending_donor = None
                    pending_acceptors = None

        if stack:
            raise GlycanParseError("Unmatched ']' in sequence (stack not empty)")
            
        if polymer_donor and polymer_acceptor and active_node_id is not None:
            link_type = LinkageType(root_anomer or Anomer.UNKNOWN, polymer_donor, polymer_acceptor)
            edges.append(LinkageEdge(source=0, target=active_node_id, edge_type=link_type))
            
        return Glycan(canonical=sequence, nodes=tuple(monosaccharides), edges=tuple(edges))


# Format Enum & Registry -----------------------------------------------------------------------------------------------
class GlycanFormat(StrEnum):
    CSDB = "csdb"
    IUPAC = "iupac"
    IUPAC_CONDENSED = "iupac"
    AUTO = "auto"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val = value.lower().strip()
            if val in ("iupac_condensed", "iupac"):
                return cls.IUPAC
            if val == "csdb":
                return cls.CSDB
            if val == "auto":
                return cls.AUTO
        return super()._missing_(value)


_CSDB_RE_INDICATORS = (
    re.compile(br"^-[0-9\?]+\)"),
    re.compile(br"\)[ab\?x]"),
    re.compile(br"[ab\?x][DLX\?][A-Za-z0-9]+[pf]?(NA|N|A)?\("),
)

def detect_format(sequence: bytes) -> str:
    """
    Heuristically detects whether a glycan sequence is formatted in CSDB or IUPAC syntax.
    """
    if not isinstance(sequence, (bytes, bytearray)):
        raise TypeError(f"sequence must be bytes, got {type(sequence).__name__}")
    s = bytes(sequence).strip()

    # Explicit IUPAC indicators
    if b"->" in s or b"alpha-" in s or b"beta-" in s or b"-D-" in s or b"-L-" in s:
        return "iupac"

    # Explicit CSDB indicators
    if s.startswith(b"*") or \
       (s.startswith(b"-") and _CSDB_RE_INDICATORS[0].match(s)) or \
       b"<<" in s or b">>" in s or \
       s.endswith(b"(1-") or s.endswith(b"(2-") or \
       _CSDB_RE_INDICATORS[1].search(s) or \
       _CSDB_RE_INDICATORS[2].search(s):
        return "csdb"

    return "iupac"


# Singletons & Registry ------------------------------------------------------------------------------------------------
_IUPAC_PARSER = IUPACParser()
_CSDB_PARSER = CSDBParser()

_PARSER_REGISTRY: dict[str, GlycanParser] = {
    "csdb": _CSDB_PARSER,
    "iupac": _IUPAC_PARSER,
    "iupac_condensed": _IUPAC_PARSER,
}

def get_parser(
    format: str | GlycanFormat = "auto",
    sequence: bytes | None = None
) -> GlycanParser:
    """
    Returns the appropriate GlycanParser singleton based on format or auto-detection.
    """
    if isinstance(format, GlycanFormat):
        fmt = format.value
    else:
        fmt = str(format).lower().strip()

    if fmt == "auto":
        if sequence is None:
            raise ValueError("Sequence must be provided when format='auto'.")
        fmt = detect_format(sequence)

    if fmt in ("iupac_condensed", "iupac"):
        return _IUPAC_PARSER
    if fmt == "csdb":
        return _CSDB_PARSER

    raise ValueError(f"Unknown glycan format '{format}'. Supported formats: 'auto', 'csdb', 'iupac'.")

