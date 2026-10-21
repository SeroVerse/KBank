import re
import warnings

from kbank.glycan import Anomer, Chirality, Glycan, LinkageEdge, LinkageType, ResidueType
from kbank.io.base import GlycanParseError, GlycanParser, GlycanParseWarning, Substitution, TokenType
from kbank.sugars import Sugars


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
            if byte == 40:
                parens += 1
            elif byte == 41:
                parens = max(0, parens - 1)
            elif byte == 123:
                curlies += 1
            elif byte == 125:
                curlies = max(0, curlies - 1)
            elif byte == 44 and parens == 0 and curlies == 0:
                is_bounded = False
                if 0 < i < len(seq) - 1:
                    if 48 <= seq[i - 1] <= 57 and 48 <= seq[i + 1] <= 57:
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

    def __init__(self) -> None:
        super().__init__()
        self._oac_idx: int = self.mod_map[b"OAc"]
        self._neu_idx: int = self.mono_map[b"Neu"]
        self._neu5ac_idx: int = self.mono_map[b"Neu5Ac"]
        self._n_acetyl_map: dict[int, int] = {
            self.mono_map[b"GlcN"]: self.mono_map[b"GlcNAc"],
            self.mono_map[b"GalN"]: self.mono_map[b"GalNAc"],
            self.mono_map[b"ManN"]: self.mono_map[b"ManNAc"],
            self.mono_map[b"GulN"]: self.mono_map[b"GulNAc"],
            self.mono_map[b"AltN"]: self.mono_map[b"AltNAc"],
            self.mono_map[b"AllN"]: self.mono_map[b"AllNAc"],
            self.mono_map[b"TalN"]: self.mono_map[b"TalNAc"],
            self.mono_map[b"IdoN"]: self.mono_map[b"IdoNAc"],
            self.mono_map[b"FucN"]: self.mono_map[b"FucNAc"],
            self.mono_map[b"QuiN"]: self.mono_map[b"QuiNAc"],
            self.mono_map[b"RhaN"]: self.mono_map[b"RhaNAc"],
            self.mono_map[b"Mur"]: self.mono_map[b"MurNAc"],
        }

    def _parse_monosaccharide(self, node: bytes, strict: bool = False) -> tuple[
        Anomer, ResidueType | None, list[Substitution]]:
        if node.startswith(b"*"):
            node = node.lstrip(b"*")
        if node.startswith(b"-"):
            node = node.lstrip(b"-")
        subs = []
        if node.endswith(b"ol") and not node.endswith(b"Col"):
            subs.append(Substitution(self.mod_map[b"ol"]))
            node = node[:-3] if node.endswith(b"-ol") else node[:-2]
            if not node or node.strip(b"ab?DxXRSldrs ") == b"":
                return Anomer.UNKNOWN, None, subs

        mod_match = self._RE_MOD_SEARCH.search(node)
        is_false_mod = False
        if mod_match:
            mod_str = mod_match.group(1)
            prefix = node[:mod_match.start()]
            if mod_str == b"N":
                is_false_mod = True
            elif mod_str in (b"Ac", b"OAc") and (
                b"Aci" in node
                or b"Leg" in node
                or b"Pse" in node
                or b"Bac" in node
                or b"Neu5Ac" in node
                or prefix.endswith(b"N")
            ):
                is_false_mod = True
            elif mod_str == b"Gro" and len(prefix) <= 2:
                is_false_mod = True
            elif mod_str == b"P" and node[mod_match.start() :].startswith((b"Par", b"Pse")):
                is_false_mod = True
            elif mod_str == b"ol":
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
            if node[idx] == b'a'[0]:
                anomer = Anomer.ALPHA
            elif node[idx] == b'b'[0]:
                anomer = Anomer.BETA
            idx += 1
            if idx < len(node) and node[idx] == b'?'[0] and (idx + 1 < len(node) and not (48 <= node[idx + 1] <= 57)):
                idx += 1
            if idx < len(node) and node[idx] == b'-'[0]:
                idx += 1

        # 2. Chirality prefix (D, L, l, d, X, x, R, S, r, s, and possible trailing ?)
        chi_chars = (b'D'[0], b'L'[0], b'l'[0], b'd'[0], b'?'[0], b'x'[0], b'X'[0], b'R'[0], b'S'[0], b'r'[0], b's'[0])
        protected_sugar_prefixes = (
            b"rib",
            b"rha",
            b"rul",
            b"xyl",
            b"xul",
            b"lyx",
            b"sor",
            b"sia",
            b"sug",
            b"sub",
            b"dig",
            b"dha",
            b"ddhex",
            b"dhex",
            b"ldmanhep",
            b"ddmanhep",
            b"dmanhep",
        )
        if idx < len(node) and node[idx] in chi_chars and not node[idx:].lower().startswith(protected_sugar_prefixes):
            first_chi = node[idx]
            if first_chi in (b'D'[0], b'd'[0]):
                chirality = Chirality.D
            elif first_chi in (b'L'[0], b'l'[0]):
                chirality = Chirality.L
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
        mono_match = self._RE_MONO.match(core_sugar)
        if not mono_match and core_sugar_clean != core_sugar:
            mono_match = self._RE_MONO.match(core_sugar_clean)
        if not mono_match and core_sugar.startswith(b"Neu"):
            mono_match = self._RE_NEU.match(core_sugar)

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

        is_fur = core_sugar.endswith(b"f") or b"fN" in core_sugar
        return anomer, ResidueType.from_sugar(sugar_idx, chirality, is_fur), subs

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
                            rem_carbons = list(mod_carbons)
                            if sub.modifier == self._oac_idx:
                                active_res = monosaccharides[active_node_id]
                                active_s = active_res.sugar
                                if active_s in self._n_acetyl_map and (2 in rem_carbons or 0 in rem_carbons):
                                    monosaccharides[active_node_id] = ResidueType.from_sugar(
                                        self._n_acetyl_map[active_s],
                                        active_res.chirality,
                                        active_res.is_furanose,
                                        active_res.is_reducing_end,
                                    )
                                    if 2 in rem_carbons:
                                        rem_carbons.remove(2)
                                    else:
                                        rem_carbons.remove(0)
                                elif active_s == self._neu_idx and (5 in rem_carbons or 0 in rem_carbons):
                                    monosaccharides[active_node_id] = ResidueType.from_sugar(
                                        self._neu5ac_idx,
                                        active_res.chirality,
                                        active_res.is_furanose,
                                        active_res.is_reducing_end,
                                    )
                                    if 5 in rem_carbons:
                                        rem_carbons.remove(5)
                                    else:
                                        rem_carbons.remove(0)
                            if rem_carbons:
                                mod_node_id = len(monosaccharides)
                                monosaccharides.append(ResidueType.from_sugar(sub.modifier, Chirality.UNKNOWN, False))
                                for carbon in rem_carbons:
                                    edges.append(LinkageEdge(source=mod_node_id, target=active_node_id,
                                                             edge_type=LinkageType(Anomer.UNKNOWN, pending_donor or 0,
                                                                                   carbon)))
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

