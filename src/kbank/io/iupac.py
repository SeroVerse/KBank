from kbank.glycan import Anomer, Chirality, Glycan, LinkageEdge, LinkageType, ResidueType
from kbank.sugars import Sugars
from kbank.io.base import GlycanParser, Substitution, GlycanParseError, TokenType


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
                        edges.append(LinkageEdge(source=mod_node_id, target=new_node_id,
                                                 edge_type=LinkageType(Anomer.UNKNOWN, donor=0, acceptor=carbon)))

                if active_node_id is not None and pending_linkage:
                    edges.append(LinkageEdge(source=new_node_id, target=active_node_id, edge_type=pending_linkage))
                active_node_id = new_node_id
                pending_linkage = None

        if stack:
            raise GlycanParseError("Unmatched ']' in sequence (stack not empty)")

        return Glycan(canonical=sequence, nodes=tuple(monosaccharides), edges=tuple(edges))

