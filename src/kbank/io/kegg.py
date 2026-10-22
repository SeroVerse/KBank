"""KEGG Glycan KCF (`NODE`/`EDGE` and `COMPOSITION`) flatfile parser."""

from __future__ import annotations

import re

from kbank.glycan import Anomer, Chirality, Glycan, LinkageEdge, LinkageType, ResidueType
from kbank.io.base import GlycanParseError, GlycanParser
from kbank.sugars import Sugars


class KEGGParser(GlycanParser):
    """Parser for KEGG Glycan KCF flatfiles (`NODE`/`EDGE` connection tables and `COMPOSITION` blocks)."""

    _RE_ENTRY = re.compile(r"^ENTRY\s+(G\d{5})", re.MULTILINE)
    _RE_COMP_TOKENS = re.compile(r"\(([^)]+)\)")
    _RE_DIGITS = re.compile(r"\d")

    def resolve_residue(self, token: str, strict: bool = False) -> ResidueType | None:
        """Resolve a KEGG KCF monosaccharide or modifier label into a `ResidueType`."""
        clean = token.strip().replace("(", "").replace(")", "")
        if not clean:
            return None

        chi = Chirality.UNKNOWN
        if clean.startswith(("D-", "d-")):
            chi = Chirality.D
            clean = clean[2:]
        elif clean.startswith(("L-", "l-")):
            chi = Chirality.L
            clean = clean[2:]
        elif len(clean) > 3 and clean[0] in ("a", "b") and clean[1] == "-":
            clean = clean[2:]
            if clean.startswith(("D-", "d-")):
                chi = Chirality.D
                clean = clean[2:]
            elif clean.startswith(("L-", "l-")):
                chi = Chirality.L
                clean = clean[2:]

        b_clean = clean.encode("utf-8")
        is_fur = b_clean.endswith(b"f")
        s_idx = self.mono_map.get(b_clean)
        if s_idx is None and b_clean.endswith((b"p", b"f")):
            s_idx = self.mono_map.get(b_clean[:-1])
        if s_idx is None:
            if strict:
                raise GlycanParseError(f"Unknown KEGG sugar token: {token!r}")
            return None
        if chi == Chirality.UNKNOWN:
            chi = Chirality(Sugars.natural_chirality[s_idx])
        return ResidueType.from_sugar(s_idx, chi, is_fur)

    def parse(
        self,
        sequence: bytes | str,
        strict: bool = False,
        *,
        record_id: bytes | str = b"",
        allow_composition: bool = False,
    ) -> Glycan:
        """Parse a KEGG Glycan flatfile (`NODE`/`EDGE` or optional `COMPOSITION`) into a `Glycan` graph."""
        text = sequence.decode("utf-8", errors="replace") if isinstance(sequence, (bytes, bytearray)) else str(sequence)
        rec_b = record_id.encode("utf-8") if isinstance(record_id, str) else bytes(record_id)
        if not rec_b:
            m_entry = self._RE_ENTRY.search(text)
            if m_entry:
                rec_b = m_entry.group(1).encode("utf-8")

        in_node = False
        in_edge = False
        raw_nodes: list[tuple[int, str]] = []
        raw_edges: list[tuple[int, int, str]] = []
        composition = ""

        for line in text.splitlines():
            if line.startswith("COMPOSITION"):
                composition = line[len("COMPOSITION") :].strip()
                in_node = False
                in_edge = False
            elif line.startswith("NODE"):
                in_node = True
                in_edge = False
            elif line.startswith("EDGE"):
                in_edge = True
                in_node = False
            elif line and not line[0].isspace():
                in_node = False
                in_edge = False
            elif in_node:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    raw_nodes.append((int(parts[0]), parts[1]))
            elif in_edge:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].isdigit():
                    donor_part = parts[1]
                    target_part = parts[2]
                    s_str, d_link = donor_part.split(":", 1) if ":" in donor_part else (donor_part, "")
                    t_str, a_link = target_part.split(":", 1) if ":" in target_part else (target_part, "")
                    if s_str.isdigit() and t_str.isdigit():
                        raw_edges.append((int(s_str), int(t_str), f"{d_link}:{a_link}"))

        if raw_nodes:
            id_map: dict[int, int] = {}
            nodes: list[ResidueType] = []
            for kegg_idx, label in raw_nodes:
                res = self.resolve_residue(label, strict=strict)
                if res is not None:
                    if len(nodes) == 0:
                        res = res.with_reducing_end()
                    id_map[kegg_idx] = len(nodes)
                    nodes.append(res)
            if not nodes:
                raise GlycanParseError("KEGG NODE block contained no resolvable monosaccharides.")

            edges: list[LinkageEdge] = []
            for s_kegg, t_kegg, link_str in raw_edges:
                if s_kegg in id_map and t_kegg in id_map:
                    d_part, a_part = link_str.split(":", 1) if ":" in link_str else (link_str, "")
                    anom = Anomer.ALPHA if "a" in d_part else (Anomer.BETA if "b" in d_part else Anomer.UNKNOWN)
                    d_nums = [int(d) for d in self._RE_DIGITS.findall(d_part) if 1 <= int(d) <= 9]
                    a_nums = [int(d) for d in self._RE_DIGITS.findall(a_part) if 1 <= int(d) <= 9]
                    donor_c = d_nums[0] if d_nums else 1
                    acc_c = a_nums[0] if a_nums else (d_nums[1] if len(d_nums) >= 2 else 0)
                    edges.append(
                        LinkageEdge(
                            source=id_map[s_kegg],
                            target=id_map[t_kegg],
                            edge_type=LinkageType(anom, donor=donor_c, acceptor=acc_c),
                        )
                    )
            if not edges and not allow_composition:
                raise GlycanParseError("KEGG entry has NODE definitions but 0 connected glycosidic edges.")
            return Glycan(canonical=rec_b or b"KEGG", nodes=tuple(nodes), edges=tuple(edges))

        if composition and allow_composition:
            tokens = self._RE_COMP_TOKENS.findall(composition)
            nodes = []
            for tok in tokens:
                res = self.resolve_residue(tok, strict=strict)
                if res is not None:
                    if len(nodes) == 0:
                        res = res.with_reducing_end()
                    nodes.append(res)
            if nodes:
                return Glycan(canonical=composition.encode("utf-8"), nodes=tuple(nodes), edges=())

        raise GlycanParseError("KEGG flatfile contains no connected NODE/EDGE structure.")


def parse_kegg_glycan_flatfile(
    text: str | bytes,
    record_id: str | bytes = b"",
    *,
    allow_composition: bool = False,
) -> Glycan | None:
    """Parse a KEGG Glycan flatfile into a `Glycan` graph, returning `None` if unparseable or edgeless."""
    parser = KEGGParser()
    try:
        return parser.parse(text, strict=False, record_id=record_id, allow_composition=allow_composition)
    except GlycanParseError, ValueError, KeyError, IndexError:
        return None
