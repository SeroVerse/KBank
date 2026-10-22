import re

from kbank.glycan import Anomer, Chirality, Glycan, LinkageEdge, LinkageType, ResidueType
from kbank.io.base import GlycanParseError, GlycanParser, Substitution, TokenType
from kbank.sugars import Sugars


class IUPACParser(GlycanParser):
    _RE_POLYMER_DONOR = re.compile(rb"-?\(([ab?]?[0-9?]+)(?:->|-)\s*$")
    _RE_POLYMER_ACCEPTOR = re.compile(rb"^\s*(?:->|-)([0-9?]+)\)-?")

    @staticmethod
    def _parse_linkage(link: bytes) -> LinkageType:
        try:
            return LinkageType.from_bytes(link)
        except ValueError as e:
            raise GlycanParseError(str(e)) from e

    def _parse_monosaccharide_with_anomer(self, sugar: bytes) -> tuple[Anomer, ResidueType, list[Substitution]]:
        sugar = sugar.strip(b"-")
        chirality = Chirality.UNKNOWN
        anomer = Anomer.UNKNOWN

        # Handle stereodescriptor/anomeric prefixes (e.g. b-D-, a-L-, D-, L-, b-, a-)
        if sugar.startswith((b"D-", b"d-")):
            chirality = Chirality.D
            sugar = sugar[2:]
        elif sugar.startswith((b"L-", b"l-")):
            chirality = Chirality.L
            sugar = sugar[2:]
        elif len(sugar) > 3 and sugar[0] in (b"a"[0], b"b"[0], b"?"[0]) and sugar[1] == b"-"[0]:
            if sugar[0] == b"a"[0]:
                anomer = Anomer.ALPHA
            elif sugar[0] == b"b"[0]:
                anomer = Anomer.BETA
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
        is_furanose = base_str.endswith(b"f")
        if chirality == Chirality.UNKNOWN:
            chirality = Chirality(Sugars.natural_chirality[sugar_idx])
        base_mono = ResidueType.from_sugar(sugar_idx, chirality, is_furanose)

        remainder = sugar[mono_match.end() :]
        modifications = []
        if remainder:
            idx = 0
            carbons = []
            while idx < len(remainder):
                c_byte = remainder[idx]
                if b"1"[0] <= c_byte <= b"9"[0] or c_byte == b"?"[0]:
                    carbons.append(c_byte - 48 if b"1"[0] <= c_byte <= b"9"[0] else 0)
                    idx += 1
                elif c_byte == b","[0]:
                    idx += 1
                else:
                    mod_match = self._RE_MOD.match(remainder[idx:])
                    if not mod_match:
                        raise GlycanParseError(f"Unknown mod at idx {idx} in {remainder}")
                    matched_mod = self.mod_map[mod_match.group(1)]
                    modifications.append(
                        Substitution(
                            modifier=matched_mod,
                            carbons=tuple(carbons) if carbons else (0,),
                        )
                    )
                    idx += mod_match.end()
                    carbons = []
        return anomer, base_mono, modifications

    def _parse_monosaccharide(self, sugar: bytes) -> tuple[ResidueType, list[Substitution]]:
        _, base_mono, modifications = self._parse_monosaccharide_with_anomer(sugar)
        return base_mono, modifications

    def parse(self, sequence: bytes, strict: bool = False) -> Glycan:
        if not isinstance(sequence, (bytes, bytearray)):
            raise TypeError(f"sequence must be bytes, got {type(sequence).__name__}")
        sequence = bytes(sequence)
        clean_seq = sequence.strip()

        polymer_donor: int | None = None
        polymer_anomer: Anomer = Anomer.UNKNOWN
        donor_match = self._RE_POLYMER_DONOR.search(clean_seq)
        if donor_match:
            d_tok = donor_match.group(1)
            if d_tok[0:1] == b"a":
                polymer_anomer = Anomer.ALPHA
                d_tok = d_tok[1:]
            elif d_tok[0:1] == b"b":
                polymer_anomer = Anomer.BETA
                d_tok = d_tok[1:]
            elif d_tok[0:1] == b"?":
                d_tok = d_tok[1:]
            polymer_donor = (d_tok[0] - 48) if (d_tok and 49 <= d_tok[0] <= 57) else 0
            clean_seq = clean_seq[: donor_match.start()].strip()

        polymer_acceptor: int | None = None
        acc_match = self._RE_POLYMER_ACCEPTOR.search(clean_seq)
        if acc_match:
            a_tok = acc_match.group(1)
            polymer_acceptor = (a_tok[0] - 48) if (a_tok and 49 <= a_tok[0] <= 57) else 0
            clean_seq = clean_seq[acc_match.end() :].strip()

        monosaccharides = []
        edges = []
        stack = []
        active_node_id: int | None = None
        pending_linkage: LinkageType | None = None
        root_anomer: Anomer = Anomer.UNKNOWN

        for token in self._lex(clean_seq):
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
                anomer, monosaccharide, subs = self._parse_monosaccharide_with_anomer(token.value)
                new_node_id = len(monosaccharides)
                if new_node_id == 0:
                    monosaccharide = monosaccharide.with_reducing_end()
                    root_anomer = anomer
                monosaccharides.append(monosaccharide)
                for sub in subs:
                    mod_node_id = len(monosaccharides)
                    monosaccharides.append(ResidueType.from_sugar(sub.modifier, Chirality.UNKNOWN, False))
                    for carbon in sub.carbons:
                        edges.append(
                            LinkageEdge(
                                source=mod_node_id,
                                target=new_node_id,
                                edge_type=LinkageType(Anomer.UNKNOWN, donor=0, acceptor=carbon),
                            )
                        )

                if active_node_id is not None and pending_linkage is not None:
                    eff_link = pending_linkage
                    if eff_link.anomer == Anomer.UNKNOWN and anomer != Anomer.UNKNOWN:
                        eff_link = LinkageType(anomer, donor=eff_link.donor, acceptor=eff_link.acceptor)
                    edges.append(LinkageEdge(source=new_node_id, target=active_node_id, edge_type=eff_link))
                active_node_id = new_node_id
                pending_linkage = None

        if stack:
            raise GlycanParseError("Unmatched ']' in sequence (stack not empty)")

        if polymer_donor is not None and polymer_acceptor is not None and active_node_id is not None:
            eff_poly_anom = polymer_anomer if polymer_anomer != Anomer.UNKNOWN else root_anomer
            edges.append(
                LinkageEdge(
                    source=0,
                    target=active_node_id,
                    edge_type=LinkageType(eff_poly_anom, donor=polymer_donor, acceptor=polymer_acceptor),
                )
            )

        return Glycan(canonical=sequence, nodes=tuple(monosaccharides), edges=tuple(edges))
