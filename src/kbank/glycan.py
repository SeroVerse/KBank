"""Module for representing a glycan and glycan batches."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any, Self, overload

import numpy as np
from scipy.sparse import csr_array, diags
from scipy.stats import wasserstein_distance_nd
from serogen.core.graph.models import TAG_EDGE, BitField, DirectedEdge, GraphBatch, GraphModel, IntNode

from kbank.sugars import Sugars


# Enums ----------------------------------------------------------------------------------------------------------------
class Chirality(IntEnum):
    UNKNOWN = 0
    D = 1
    L = 2


class Anomer(IntFlag):
    """Anomeric configuration of a glycosidic linkage."""

    UNKNOWN = 0
    ALPHA = 1
    BETA = 2


# Single instances -----------------------------------------------------------------------------------------------------
class ResidueType(IntNode):
    """A unified 32-bit integer encoding for a monosaccharide residue node.

    Bit 0-7: Sugar ID (0..255)
    Bit 8-9: Chirality (0=Unknown, 1=D, 2=L)
    Bit 10: Ring form (0=Pyranose, 1=Furanose)
    Bit 11: Reducing end flag (1=True, 0=False)
    Bit 12: Main-chain backbone flag (1=Backbone, 0=Side-chain branch or modifier)
    Bit 31: Tag bit (0 = primary node, 1 = intermediate edge node)
    """

    SUGAR = BitField(shift=0, width=8)
    CHIRALITY = BitField(shift=8, width=2)
    FURANOSE = BitField(shift=10, width=1)
    REDUCING_END = BitField(shift=11, width=1)
    BACKBONE = BitField(shift=12, width=1)
    TAG_EDGE = BitField(shift=31, width=1)

    TOPOLOGY_MASK: int = int(REDUCING_END.mask | BACKBONE.mask)

    def __new__(
        cls,
        val_or_sugar: int | bytes | bytearray,
        chirality: Chirality = Chirality.UNKNOWN,
        is_furanose: bool = False,
        is_reducing_end: bool = False,
        is_backbone: bool = False,
    ) -> Self:
        if isinstance(val_or_sugar, (int, np.integer)):
            return super().__new__(cls, int(val_or_sugar))
        sugar_idx = Sugars.index(val_or_sugar)
        packed = cls.pack(
            sugar=sugar_idx,
            chirality=int(chirality),
            is_furanose=is_furanose,
            is_reducing_end=is_reducing_end,
            is_backbone=is_backbone,
        )
        return super().__new__(cls, packed)

    @classmethod
    def pack(
        cls,
        sugar: int = 0,
        chirality: int = 0,
        is_furanose: bool | int = False,
        is_reducing_end: bool | int = False,
        is_backbone: bool | int = False,
        **_kwargs: int,
    ) -> int:
        return (
            cls.SUGAR.pack(sugar)
            | cls.CHIRALITY.pack(chirality)
            | cls.FURANOSE.pack(int(is_furanose))
            | cls.REDUCING_END.pack(int(is_reducing_end))
            | cls.BACKBONE.pack(int(is_backbone))
        )

    @classmethod
    def from_sugar(
        cls,
        sugar_idx: int,
        chirality: Chirality | int = Chirality.UNKNOWN,
        is_furanose: bool = False,
        is_reducing_end: bool = False,
        is_backbone: bool = False,
    ) -> ResidueType:
        chi_val = chirality.value if isinstance(chirality, Chirality) else int(chirality)
        val = cls.pack(
            sugar=sugar_idx,
            chirality=chi_val,
            is_furanose=is_furanose,
            is_reducing_end=is_reducing_end,
            is_backbone=is_backbone,
        )
        return cls(val)

    @property
    def sugar(self) -> int:
        return self.SUGAR

    @property
    def is_modifier(self) -> bool:
        return Sugars.classes[self.sugar] == b"Mod"

    @property
    def chirality(self) -> Chirality:
        return Chirality(self.CHIRALITY)

    @property
    def is_furanose(self) -> bool:
        return bool(self.FURANOSE)

    @property
    def is_reducing_end(self) -> bool:
        return bool(self.REDUCING_END)

    @property
    def is_backbone(self) -> bool:
        """True if this residue lies on the main-chain polymer backbone."""
        return bool(self.BACKBONE)

    @property
    def is_branch(self) -> bool:
        """True if this is a non-modifier side-chain branch monosaccharide."""
        return not bool(self.BACKBONE) and not self.is_modifier

    def with_reducing_end(self, is_reducing: bool = True) -> ResidueType:
        """Returns a copy of this ResidueType with the reducing end bit set/cleared."""
        mask = type(self).REDUCING_END.mask
        if is_reducing:
            return ResidueType(self | mask)
        return ResidueType(self & ~mask)

    def with_backbone(self, is_backbone: bool = True) -> ResidueType:
        """Returns a copy of this ResidueType with the main-chain backbone bit set/cleared."""
        mask = type(self).BACKBONE.mask
        if is_backbone:
            return ResidueType(self | mask)
        return ResidueType(self & ~mask)

    def __repr__(self) -> str:
        s_name = Sugars.names[self.sugar].decode("ascii", errors="replace")
        chi = "D-" if self.chirality == Chirality.D else "L-" if self.chirality == Chirality.L else ""
        form = "f" if self.is_furanose else "p"
        red = "*" if self.is_reducing_end else ""
        return f"{red}{chi}{s_name}{form}"


class LinkageType(IntNode):
    """A unified 32-bit integer encoding for a directed glycosidic linkage.

    Bits 0-1:   Anomeric configuration (Anomer: UNKNOWN=0, ALPHA=1, BETA=2)
    Bits 2-5:   Donor carbon position (Carbon: UNKNOWN=0, C1..C9 -> 1..9)
    Bits 6-9:   Acceptor carbon position (Carbon: UNKNOWN=0, C1..C9 -> 1..9)
    Bit 10:     Main-chain backbone linkage flag (1 = Backbone, 0 = Branch/Modifier)
    Bit 11:     Inter-repeat Wzy polymer linkage flag (1 = Polymer repeat edge, 0 = Intra-unit)
    Bits 12-30: Reserved
    Bit 31:     TAG_EDGE (1 = Edge node in bipartite graphs)
    """

    ANOMER = BitField(shift=0, width=2)
    DONOR = BitField(shift=2, width=4)
    ACCEPTOR = BitField(shift=6, width=4)
    BACKBONE = BitField(shift=10, width=1)
    POLYMER = BitField(shift=11, width=1)
    TAG_EDGE = BitField(shift=31, width=1)

    OFFSET: int = 1 << 31
    TOPOLOGY_MASK: int = int(BACKBONE.mask | POLYMER.mask)
    CHEMICAL_MASK: int = int(ANOMER.mask | DONOR.mask | ACCEPTOR.mask | TAG_EDGE.mask)

    def __new__(
        cls,
        val_or_anomer: int | Anomer | bytes | bytearray,
        donor: int | None = None,
        acceptor: int | None = None,
        *,
        is_backbone: bool = False,
        is_polymer: bool = False,
    ) -> Self:
        """Constructs or parses a bit-packed LinkageType."""
        if isinstance(val_or_anomer, (bytes, bytearray)):
            return cls.from_bytes(val_or_anomer)
        if donor is None:
            val = int(val_or_anomer)
            if not (val & TAG_EDGE):
                val |= TAG_EDGE
            return super().__new__(cls, val)
        anomer = val_or_anomer if isinstance(val_or_anomer, Anomer) else Anomer(int(val_or_anomer))
        d = 0 if donor is None else int(donor)
        acc = 0 if acceptor is None else int(acceptor)
        packed = cls.pack(anomer, d, acc, is_backbone=is_backbone, is_polymer=is_polymer)
        return super().__new__(cls, packed)

    @classmethod
    def pack(
        cls,
        anomer: int | Anomer = Anomer.UNKNOWN,
        donor: int = 0,
        acceptor: int = 0,
        *,
        is_backbone: bool | int = False,
        is_polymer: bool | int = False,
        **_kwargs: int,
    ) -> int:
        """Packs anomer, donor carbon, acceptor carbon, and topology flags into a 32-bit integer with TAG_EDGE set."""
        return (
            cls.ANOMER.pack(int(anomer))
            | cls.DONOR.pack(int(donor))
            | cls.ACCEPTOR.pack(int(acceptor))
            | cls.BACKBONE.pack(int(is_backbone))
            | cls.POLYMER.pack(int(is_polymer))
            | cls.TAG_EDGE.pack(1)
        )

    @classmethod
    def from_bytes(cls, link: bytes | bytearray) -> Self:  # ty: ignore[invalid-method-override]
        """Parses a linkage byte-string like b'b1-4', b'a1-3', b'1-4', b'?1-?' into a LinkageType."""
        clean = bytes(link).replace(b"->", b"-")
        if len(clean) == 3 and clean[1] == 45:  # e.g. b"1-4" -> b"?1-4"
            clean = b"?" + clean
        if len(clean) != 4 or clean[2] != 45:
            raise ValueError(f"Invalid linkage format: {clean.decode('ascii', errors='replace')}")

        anomer_byte = clean[0]
        if anomer_byte in (97, b"a"[0]):
            anomer = Anomer.ALPHA
        elif anomer_byte in (98, b"b"[0]):
            anomer = Anomer.BETA
        else:
            anomer = Anomer.UNKNOWN

        donor = (clean[1] - 48) if 49 <= clean[1] <= 57 else 0
        acceptor = (clean[3] - 48) if 49 <= clean[3] <= 57 else 0
        return cls(cls.pack(anomer, donor, acceptor))

    @property
    def raw_index(self) -> int:
        """Chemical linkage bits without TAG_EDGE or topology flags."""
        return int(self) & int(self.ANOMER.mask | self.DONOR.mask | self.ACCEPTOR.mask)

    @property
    def chemical_int(self) -> int:
        """32-bit integer with `TAG_EDGE` and chemical bits (`ANOMER`, `DONOR`, `ACCEPTOR`), excluding topology flags."""
        return int(self) & self.CHEMICAL_MASK

    @property
    def anomer(self) -> Anomer:
        """Anomeric configuration."""
        return Anomer(self.ANOMER)

    @property
    def donor(self) -> int:
        """Donor carbon (1..9, or 0 if unknown)."""
        return int(self.DONOR)

    @property
    def acceptor(self) -> int:
        """Acceptor carbon (1..9, or 0 if unknown)."""
        return int(self.ACCEPTOR)

    @property
    def is_backbone(self) -> bool:
        """True if this glycosidic linkage connects two main-chain backbone residues."""
        return bool(self.BACKBONE)

    @property
    def is_polymer(self) -> bool:
        """True if this linkage is the inter-repeat Wzy polymerase linkage."""
        return bool(self.POLYMER)

    @property
    def is_branch(self) -> bool:
        """True if this linkage attaches or extends a side-chain branch (`not is_backbone`)."""
        return not bool(self.BACKBONE)

    def with_topology(self, *, is_backbone: bool = False, is_polymer: bool = False) -> LinkageType:
        """Return a copy of this `LinkageType` with updated `BACKBONE` and `POLYMER` bitflags."""
        cleared = int(self) & ~type(self).TOPOLOGY_MASK
        packed = cleared | type(self).BACKBONE.pack(int(is_backbone)) | type(self).POLYMER.pack(int(is_polymer))
        return LinkageType(packed)

    @property
    def name(self) -> bytes:
        """Byte-string representation (e.g. b'b1-4', b'a1-3')."""
        anom = self.ANOMER
        a_b = 97 if anom == Anomer.ALPHA else (98 if anom == Anomer.BETA else 63)
        d_val = self.DONOR
        d_b = (48 + d_val) if 0 < d_val <= 9 else 63
        acc_val = self.ACCEPTOR
        acc_b = (48 + acc_val) if 0 < acc_val <= 9 else 63
        return bytes([a_b, d_b, 45, acc_b])

    def __contains__(self, item: object) -> bool:
        """Checks if an Anomer or carbon number is part of this linkage."""
        if isinstance(item, Anomer):
            return item in self.anomer
        if isinstance(item, int):
            return item == self.donor or item == self.acceptor
        return False

    def __repr__(self) -> str:
        """Returns string representation of the linkage."""
        return f"Link({self.name.decode('ascii')})"


@dataclass(slots=True, frozen=True)
class LinkageEdge(DirectedEdge):
    source: int  # Parent / source index
    target: int  # Child / target index
    edge_type: LinkageType


def _residue_compatible(
    n_sub: ResidueType,
    n_sup: ResidueType,
    *,
    ignore_reducing_end: bool = True,
    allow_wildcards: bool = False,
) -> bool:
    """Return True if `n_sub` is semantically compatible with `n_sup`."""
    if n_sub.sugar != n_sup.sugar or n_sub.is_furanose != n_sup.is_furanose:
        return False
    if not ignore_reducing_end and n_sub.is_reducing_end != n_sup.is_reducing_end:
        return False
    if n_sub.chirality != n_sup.chirality:
        if not allow_wildcards or (n_sub.chirality != Chirality.UNKNOWN and n_sup.chirality != Chirality.UNKNOWN):
            return False
    return True


def _linkage_compatible(
    l_sub: LinkageType,
    l_sup: LinkageType,
    *,
    allow_wildcards: bool = False,
) -> bool:
    """Return True if directed glycosidic linkage `l_sub` is compatible with `l_sup`."""
    if l_sub.chemical_int == l_sup.chemical_int:
        return True
    if not allow_wildcards:
        return False
    if l_sub.anomer != l_sup.anomer and l_sub.anomer != Anomer.UNKNOWN and l_sup.anomer != Anomer.UNKNOWN:
        return False
    if l_sub.donor != l_sup.donor and l_sub.donor != 0 and l_sup.donor != 0:
        return False
    if l_sub.acceptor != l_sup.acceptor and l_sub.acceptor != 0 and l_sup.acceptor != 0:
        return False
    return True


def _find_glycan_subgraph_mapping(
    sub: Glycan,
    sup: Glycan,
    *,
    ignore_reducing_end: bool = True,
    allow_wildcards: bool = False,
    require_bijection: bool = False,
) -> dict[int, int] | None:
    """Return an injective directed edge-preserving node mapping (`sub` -> `sup`), or `None`."""
    n_sub = len(sub.nodes)
    n_sup = len(sup.nodes)
    e_sub = len(sub.edges)
    e_sup = len(sup.edges)

    if require_bijection and (n_sub != n_sup or e_sub != e_sup):
        return None
    if n_sub > n_sup or e_sub > e_sup:
        return None
    if n_sub == 0:
        return {}

    # O(1) bitmask pre-filter: every sugar index in sub must exist in sup
    if (sub.sugar_mask & ~sup.sugar_mask) != 0:
        return None

    if require_bijection:
        if sub.sugar_mask != sup.sugar_mask:
            return None
        if sorted(n.sugar for n in sub.nodes) != sorted(n.sugar for n in sup.nodes):
            return None
        if not allow_wildcards:
            mask = ~ResidueType.TOPOLOGY_MASK if ignore_reducing_end else ~ResidueType.BACKBONE.mask
            if sorted(int(n) & mask for n in sub.nodes) != sorted(int(n) & mask for n in sup.nodes):
                return None
            if sorted(e.edge_type.chemical_int for e in sub.edges) != sorted(
                e.edge_type.chemical_int for e in sup.edges
            ):
                return None

    out_sub: list[list[tuple[int, LinkageType]]] = [[] for _ in range(n_sub)]
    in_sub: list[list[tuple[int, LinkageType]]] = [[] for _ in range(n_sub)]
    for e in sub.edges:
        if 0 <= e.source < n_sub and 0 <= e.target < n_sub:
            out_sub[e.source].append((e.target, e.edge_type))
            in_sub[e.target].append((e.source, e.edge_type))

    out_sup: list[list[tuple[int, LinkageType]]] = [[] for _ in range(n_sup)]
    in_sup: list[list[tuple[int, LinkageType]]] = [[] for _ in range(n_sup)]
    for e in sup.edges:
        if 0 <= e.source < n_sup and 0 <= e.target < n_sup:
            out_sup[e.source].append((e.target, e.edge_type))
            in_sup[e.target].append((e.source, e.edge_type))

    candidates: list[list[int]] = []
    for u in range(n_sub):
        u_node = sub.nodes[u]
        u_out = out_sub[u]
        u_in = in_sub[u]
        u_cands: list[int] = []
        for v in range(n_sup):
            if not _residue_compatible(
                u_node,
                sup.nodes[v],
                ignore_reducing_end=ignore_reducing_end,
                allow_wildcards=allow_wildcards,
            ):
                continue
            v_out = out_sup[v]
            v_in = in_sup[v]
            if require_bijection:
                if len(u_out) != len(v_out) or len(u_in) != len(v_in):
                    continue
            else:
                if len(u_out) > len(v_out) or len(u_in) > len(v_in):
                    continue

            ok = True
            for w, l_sub in u_out:
                w_node = sub.nodes[w]
                if not any(
                    _linkage_compatible(l_sub, l_sup, allow_wildcards=allow_wildcards)
                    and _residue_compatible(
                        w_node,
                        sup.nodes[vw],
                        ignore_reducing_end=ignore_reducing_end,
                        allow_wildcards=allow_wildcards,
                    )
                    for vw, l_sup in v_out
                ):
                    ok = False
                    break
            if not ok:
                continue

            for w, l_sub in u_in:
                w_node = sub.nodes[w]
                if not any(
                    _linkage_compatible(l_sub, l_sup, allow_wildcards=allow_wildcards)
                    and _residue_compatible(
                        w_node,
                        sup.nodes[vw],
                        ignore_reducing_end=ignore_reducing_end,
                        allow_wildcards=allow_wildcards,
                    )
                    for vw, l_sup in v_in
                ):
                    ok = False
                    break
            if ok:
                u_cands.append(v)

        if not u_cands:
            return None
        candidates.append(u_cands)

    order = sorted(
        range(n_sub),
        key=lambda u: (len(candidates[u]), -(len(out_sub[u]) + len(in_sub[u]))),
    )
    mapping: dict[int, int] = {}
    used_sup: set[int] = set()

    def _pair_edges_compatible(
        req_links: list[LinkageType],
        avail_links: list[LinkageType],
    ) -> bool:
        if len(req_links) > len(avail_links):
            return False
        used_idx: set[int] = set()
        for rl in req_links:
            matched = False
            for idx_a, al in enumerate(avail_links):
                if idx_a not in used_idx and _linkage_compatible(rl, al, allow_wildcards=allow_wildcards):
                    used_idx.add(idx_a)
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _backtrack(step: int) -> bool:
        if step == n_sub:
            return True
        u = order[step]
        for v in candidates[u]:
            if v in used_sup:
                continue
            consistent = True
            # Verify directed edges between u and all already-mapped neighbors w
            for w, _ in out_sub[u]:
                if w == u:
                    req = [lt for tgt, lt in out_sub[u] if tgt == u]
                    avail = [lt for tgt, lt in out_sup[v] if tgt == v]
                    if not _pair_edges_compatible(req, avail):
                        consistent = False
                        break
                elif w in mapping:
                    vw = mapping[w]
                    req = [lt for tgt, lt in out_sub[u] if tgt == w]
                    avail = [lt for tgt, lt in out_sup[v] if tgt == vw]
                    if not _pair_edges_compatible(req, avail):
                        consistent = False
                        break
            if not consistent:
                continue

            for w, _ in in_sub[u]:
                if w != u and w in mapping:
                    vw = mapping[w]
                    req = [lt for src, lt in in_sub[u] if src == w]
                    avail = [lt for src, lt in in_sup[v] if src == vw]
                    if not _pair_edges_compatible(req, avail):
                        consistent = False
                        break
            if not consistent:
                continue

            mapping[u] = v
            used_sup.add(v)
            if _backtrack(step + 1):
                return True
            del mapping[u]
            used_sup.remove(v)
        return False

    return dict(mapping) if _backtrack(0) else None


def _match_glycan_subgraph(
    sub: Glycan,
    sup: Glycan,
    *,
    ignore_reducing_end: bool = True,
    allow_wildcards: bool = False,
    require_bijection: bool = False,
) -> bool:
    """Exact directed edge-preserving injective homomorphism (`sub` -> `sup`)."""
    return (
        _find_glycan_subgraph_mapping(
            sub,
            sup,
            ignore_reducing_end=ignore_reducing_end,
            allow_wildcards=allow_wildcards,
            require_bijection=require_bijection,
        )
        is not None
    )


# Main container -------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Glycan(GraphModel):
    """Container for the components of a glycan."""

    canonical: bytes
    nodes: tuple[ResidueType, ...]
    edges: tuple[LinkageEdge, ...]

    def __post_init__(self) -> None:
        """Stamp `ResidueType.BACKBONE` and `LinkageType.BACKBONE` / `POLYMER` bitflags if not yet initialized."""
        n = len(self.nodes)
        if n == 0 or any(node.is_backbone for node in self.nodes):
            return

        cyc = self.cycle_nodes()
        r_idx = self.reducing_end_index
        wzy_edge_idx: int | None = None
        backbone_nodes: set[int] = set()

        if cyc:
            backbone_nodes = set(cyc)
            anchor = r_idx if r_idx in backbone_nodes else (0 if 0 in backbone_nodes else cyc[0])
            for k, e in enumerate(self.edges):
                if (
                    e.source == anchor
                    and e.target in backbone_nodes
                    and not self.nodes[e.source].is_modifier
                    and not self.nodes[e.target].is_modifier
                ):
                    wzy_edge_idx = k
                    break
        else:
            if self.has_polymer_repeat:
                for k, e in enumerate(self.edges):
                    if e.source == 0 and not self.nodes[e.target].is_modifier:
                        wzy_edge_idx = k
                        break
            rev_core: dict[int, list[int]] = {i: [] for i in range(n)}
            for k, e in enumerate(self.edges):
                if k == wzy_edge_idx or e.source == e.target:
                    continue
                if 0 <= e.source < n and 0 <= e.target < n:
                    if not self.nodes[e.source].is_modifier and not self.nodes[e.target].is_modifier:
                        rev_core[e.target].append(e.source)

            depths: dict[int, int] = {}
            parent_to_root: dict[int, int] = {}
            if 0 <= r_idx < n and not self.nodes[r_idx].is_modifier:
                depths[r_idx] = 0
                queue: list[int] = [r_idx]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    d_u = depths[u]
                    for v_donor in rev_core.get(u, []):
                        if v_donor not in depths:
                            depths[v_donor] = d_u + 1
                            parent_to_root[v_donor] = u
                            queue.append(v_donor)

            if depths:
                deepest_u = max(depths.keys(), key=lambda u: (depths[u], -u))
                curr_u: int | None = deepest_u
                while curr_u is not None and curr_u not in backbone_nodes:
                    backbone_nodes.add(curr_u)
                    curr_u = parent_to_root.get(curr_u)
            else:
                for i, node in enumerate(self.nodes):
                    if not node.is_modifier:
                        backbone_nodes.add(i)

        new_nodes = tuple(
            node.with_backbone((i in backbone_nodes) and not node.is_modifier) for i, node in enumerate(self.nodes)
        )
        new_edges = tuple(
            LinkageEdge(
                source=e.source,
                target=e.target,
                edge_type=e.edge_type.with_topology(
                    is_backbone=(k == wzy_edge_idx)
                    or (
                        e.source in backbone_nodes
                        and e.target in backbone_nodes
                        and not self.nodes[e.source].is_modifier
                        and not self.nodes[e.target].is_modifier
                    ),
                    is_polymer=(k == wzy_edge_idx),
                ),
            )
            for k, e in enumerate(self.edges)
        )
        object.__setattr__(self, "nodes", new_nodes)
        object.__setattr__(self, "edges", new_edges)

    def __repr__(self) -> str:
        return self.canonical.__repr__()

    @property
    def reducing_end(self) -> ResidueType | None:
        """Returns the reducing end ResidueType node if marked, or None."""
        for node in self.nodes:
            if node.is_reducing_end:
                return node
        return self.nodes[0] if self.nodes else None

    @property
    def reducing_end_index(self) -> int:
        """Return the 0-based index of the reducing-end node in `self.nodes` (or 0 if unmarked)."""
        for idx, node in enumerate(self.nodes):
            if node.is_reducing_end and not node.is_modifier:
                return idx
        for idx, node in enumerate(self.nodes):
            if not node.is_modifier:
                return idx
        return 0

    def with_reducing_end(self, node_index: int) -> Glycan:
        """Return a copy of this `Glycan` with `node_index` marked as the sole reducing-end residue."""
        if not (0 <= node_index < len(self.nodes)) or self.nodes[node_index].is_modifier:
            return self
        if self.reducing_end_index == node_index and sum(1 for n in self.nodes if n.is_reducing_end) == 1:
            return self
        new_nodes = tuple(n.with_reducing_end(i == node_index).with_backbone(False) for i, n in enumerate(self.nodes))
        new_edges = tuple(
            LinkageEdge(
                source=e.source,
                target=e.target,
                edge_type=e.edge_type.with_topology(is_backbone=False, is_polymer=False),
            )
            for e in self.edges
        )
        return Glycan(canonical=self.canonical, nodes=new_nodes, edges=new_edges)

    @classmethod
    def from_topology(
        cls,
        nodes: Sequence[ResidueType],
        edges: Sequence[LinkageEdge],
        *,
        canonical: bytes | None = None,
    ) -> Glycan:
        """Construct a `Glycan` from `(nodes, edges)` and automatically synthesize its canonical CSDB byte-string."""
        temp = cls(canonical=canonical or b"", nodes=tuple(nodes), edges=tuple(edges))
        if canonical is not None and len(canonical) > 0:
            return temp
        return cls(canonical=temp.to_csdb(), nodes=temp.nodes, edges=temp.edges)

    def to_csdb(self) -> bytes:
        """Serialize this `Glycan` graph into a deterministic canonical CSDB linear byte-string."""
        n = len(self.nodes)
        if n == 0:
            return b""

        r_idx = self.reducing_end_index
        wzy_edge_idx, _, _ = self.polymer_edge_and_assembly_depths(r_idx)
        wzy_edge = self.edges[wzy_edge_idx] if wzy_edge_idx is not None else None
        wzy_tip = wzy_edge.target if wzy_edge is not None else None

        ol_idx = Sugars.name_to_index.get(b"ol", -1)
        has_alditol_root = False
        mod_groups: dict[tuple[int, int], list[LinkageType]] = {}
        rev_core: dict[int, list[tuple[int, LinkageType]]] = {i: [] for i in range(n)}

        for k, e in enumerate(self.edges):
            if k == wzy_edge_idx or e.source == e.target:
                continue
            if not (0 <= e.source < n and 0 <= e.target < n):
                continue
            src_node = self.nodes[e.source]
            tgt_node = self.nodes[e.target]
            if src_node.is_modifier:
                if src_node.sugar == ol_idx and e.target == r_idx:
                    has_alditol_root = True
                else:
                    mod_groups.setdefault((e.source, e.target), []).append(e.edge_type)
            elif not tgt_node.is_modifier:
                rev_core[e.target].append((e.source, e.edge_type))

        # Build cycle-free directed assembly tree rooted at r_idx via BFS
        tree_children: dict[int, list[tuple[int, LinkageType]]] = {i: [] for i in range(n)}
        parent_map: dict[int, int] = {}
        visited_core: set[int] = {r_idx}
        queue: list[int] = [r_idx]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v_donor, ltype in rev_core.get(u, []):
                if v_donor not in visited_core:
                    visited_core.add(v_donor)
                    parent_map[v_donor] = u
                    tree_children[u].append((v_donor, ltype))
                    queue.append(v_donor)

        # Trace path from r_idx to wzy_tip so the polymer repeat tip is always on the unbracketed main chain
        on_wzy_path: set[int] = set()
        if wzy_tip is not None and wzy_tip in visited_core:
            curr_w: int | None = wzy_tip
            while curr_w is not None and curr_w not in on_wzy_path:
                on_wzy_path.add(curr_w)
                curr_w = parent_map.get(curr_w)

        subtree_heights: dict[int, int] = {}

        def _get_height(u_idx: int) -> int:
            if u_idx in subtree_heights:
                return subtree_heights[u_idx]
            ch = tree_children.get(u_idx, [])
            h = 1 if not ch else 1 + max(_get_height(v_c) for v_c, _ in ch)
            subtree_heights[u_idx] = h
            return h

        for i_node in visited_core:
            _get_height(i_node)

        def _format_link(lt: LinkageType) -> bytes:
            d_b = str(lt.donor).encode("ascii") if lt.donor > 0 else b"1"
            a_b = str(lt.acceptor).encode("ascii") if lt.acceptor > 0 else b"?"
            return b"(" + d_b + b"-" + a_b + b")"

        def _format_mod(mod_src: int, ltypes: list[LinkageType]) -> bytes:
            m_name = Sugars.names[self.nodes[mod_src].sugar]
            if len(ltypes) >= 2:
                lt1, lt2 = sorted(ltypes[:2], key=lambda x: (x.acceptor, x.donor))
                d1 = str(lt1.donor if lt1.donor > 0 else 2).encode("ascii")
                a1 = str(lt1.acceptor if lt1.acceptor > 0 else 4).encode("ascii")
                d2 = str(lt2.donor if lt2.donor > 0 else 2).encode("ascii")
                a2 = str(lt2.acceptor if lt2.acceptor > 0 else 6).encode("ascii")
                return b"[" + m_name + b"(" + d1 + b"-" + a1 + b":" + d2 + b"-" + a2 + b")]"
            lt0 = ltypes[0]
            d0 = str(lt0.donor if lt0.donor > 0 else 1).encode("ascii")
            a0 = str(lt0.acceptor).encode("ascii") if lt0.acceptor > 0 else b"?"
            return b"[" + m_name + b"(" + d0 + b"-" + a0 + b")]"

        def _format_residue(u_idx: int, anom: Anomer) -> bytes:
            res = self.nodes[u_idx]
            anom_b = b"a" if anom == Anomer.ALPHA else (b"b" if anom == Anomer.BETA else b"?")
            chi_b = b"D" if res.chirality == Chirality.D else (b"L" if res.chirality == Chirality.L else b"")
            s_name = Sugars.names[res.sugar]
            if res.is_furanose:
                ring_b = b"" if s_name.endswith(b"f") else b"f"
            elif s_name in (
                b"Glc",
                b"Gal",
                b"Man",
                b"Rha",
                b"Fuc",
                b"Rib",
                b"Xyl",
                b"Ara",
                b"Lyx",
                b"Tal",
                b"Alt",
                b"All",
                b"Gul",
                b"Ido",
                b"Qui",
                b"Kdo",
                b"Ko",
                b"Abe",
                b"Par",
                b"Tyv",
                b"Asc",
                b"Col",
                b"Yer",
                b"LDmanHep",
                b"DDmanHep",
                b"Hep",
            ):
                ring_b = b"p"
            else:
                ring_b = b""
            ol_b = b"-ol" if (u_idx == r_idx and has_alditol_root) else b""
            return anom_b + chi_b + s_name + ring_b + ol_b

        mods_by_target: dict[int, list[tuple[int, list[LinkageType]]]] = {i: [] for i in range(n)}
        for (m_src, m_tgt), lts in mod_groups.items():
            mods_by_target[m_tgt].append((m_src, lts))
        for m_tgt in mods_by_target:
            mods_by_target[m_tgt].sort(
                key=lambda item: (
                    min((lt.acceptor for lt in item[1]), default=0),
                    int(self.nodes[item[0]].sugar),
                )
            )

        def _serialize_subtree(u_idx: int, out_anomer: Anomer, *, is_main_chain: bool) -> bytes:
            ch = tree_children.get(u_idx, [])
            main_child: tuple[int, LinkageType] | None = None
            branch_children: list[tuple[int, LinkageType]] = []

            if ch:
                if is_main_chain:
                    candidates = sorted(
                        ch,
                        key=lambda item: (
                            1 if item[0] in on_wzy_path else 0,
                            1 if self.nodes[item[0]].is_backbone else 0,
                            1 if item[1].is_backbone else 0,
                            _get_height(item[0]),
                            -item[1].acceptor,
                        ),
                        reverse=True,
                    )
                    top_cand = candidates[0]
                    is_main_cand = (
                        (top_cand[0] in on_wzy_path)
                        if on_wzy_path
                        else (self.nodes[top_cand[0]].is_backbone or top_cand[1].is_backbone or len(candidates) == 1)
                    )
                    if is_main_cand:
                        main_child = top_cand
                        branch_children = candidates[1:]
                    else:
                        branch_children = candidates
                else:
                    candidates = sorted(
                        ch,
                        key=lambda item: (_get_height(item[0]), -item[1].acceptor, -int(self.nodes[item[0]])),
                        reverse=True,
                    )
                    main_child = candidates[0]
                    branch_children = candidates[1:]

            branch_children.sort(key=lambda item: (item[1].acceptor, item[1].donor, int(self.nodes[item[0]])))

            main_prefix = (
                _serialize_subtree(main_child[0], main_child[1].anomer, is_main_chain=is_main_chain)
                + _format_link(main_child[1])
                if main_child is not None
                else b""
            )
            br_str = b"".join(
                b"[" + _serialize_subtree(v_br, lt_br.anomer, is_main_chain=False) + _format_link(lt_br) + b"]"
                for v_br, lt_br in branch_children
            )
            mod_str = b"".join(_format_mod(m_src, m_lts) for m_src, m_lts in mods_by_target.get(u_idx, []))
            return main_prefix + br_str + mod_str + _format_residue(u_idx, out_anomer)

        if wzy_edge is not None and wzy_tip in on_wzy_path:
            wzy_lt = wzy_edge.edge_type
            p_don = str(wzy_lt.donor if wzy_lt.donor > 0 else 1).encode("ascii")
            p_acc = str(wzy_lt.acceptor).encode("ascii") if wzy_lt.acceptor > 0 else b"?"
            body = _serialize_subtree(r_idx, wzy_lt.anomer, is_main_chain=True)
            return b"-" + p_acc + b")" + body + b"(" + p_don + b"-"

        return _serialize_subtree(r_idx, Anomer.UNKNOWN, is_main_chain=True)

    def backbone_indices(self) -> tuple[int, ...]:
        """Return the 0-based indices of all core monosaccharide nodes on the main-chain backbone."""
        return tuple(i for i, n in enumerate(self.nodes) if n.is_backbone)

    def branch_indices(self) -> tuple[int, ...]:
        """Return the 0-based indices of all non-modifier side-chain branch monosaccharide nodes."""
        return tuple(i for i, n in enumerate(self.nodes) if n.is_branch)

    def polymer_edge_index(self) -> int | None:
        """Return the 0-based index in `self.edges` of the inter-repeat `Wzy` polymer linkage, or `None`."""
        for k, e in enumerate(self.edges):
            if e.edge_type.is_polymer:
                return k
        return None

    def branch_point_indices(self) -> tuple[int, ...]:
        """Return the 0-based indices of core monosaccharide residues that accept a side-chain branch linkage."""
        n = len(self.nodes)
        core_in_deg: dict[int, int] = {i: 0 for i in range(n) if not self.nodes[i].is_modifier}
        has_branch_donor: set[int] = set()
        for e in self.edges:
            if 0 <= e.source < n and 0 <= e.target < n:
                s_n = self.nodes[e.source]
                t_n = self.nodes[e.target]
                if not s_n.is_modifier and not t_n.is_modifier:
                    core_in_deg[e.target] = core_in_deg.get(e.target, 0) + 1
                    if e.edge_type.is_branch or s_n.is_branch:
                        has_branch_donor.add(e.target)
        return tuple(
            i
            for i in range(n)
            if not self.nodes[i].is_modifier and (i in has_branch_donor or core_in_deg.get(i, 0) >= 2)
        )

    def terminal_leaf_indices(self) -> tuple[int, ...]:
        """Return the 0-based indices of terminal non-reducing leaf monosaccharides (zero incoming core edges)."""
        n = len(self.nodes)
        has_incoming_core: set[int] = set()
        for e in self.edges:
            if e.edge_type.is_polymer:
                continue
            if 0 <= e.source < n and 0 <= e.target < n:
                if not self.nodes[e.source].is_modifier and not self.nodes[e.target].is_modifier:
                    has_incoming_core.add(e.target)
        return tuple(
            i
            for i, node in enumerate(self.nodes)
            if not node.is_modifier and i not in has_incoming_core and (node.is_branch or not self.has_polymer_repeat)
        )

    def cycle_nodes(self) -> tuple[int, ...]:
        """Return the ordered tuple of core monosaccharide node indices forming the polymer repeat cycle."""
        n = len(self.nodes)
        if n == 0:
            return ()
        out_core: dict[int, int] = {}
        for e in self.edges:
            if 0 <= e.source < n and 0 <= e.target < n:
                if not self.nodes[e.source].is_modifier and not self.nodes[e.target].is_modifier:
                    out_core[e.source] = e.target
        for start in range(n):
            if self.nodes[start].is_modifier or start not in out_core:
                continue
            visited: list[int] = []
            seen: dict[int, int] = {}
            curr: int | None = start
            while curr is not None and curr in out_core:
                if curr in seen:
                    return tuple(visited[seen[curr] :])
                seen[curr] = len(visited)
                visited.append(curr)
                curr = out_core.get(curr)
        return ()

    def polymer_edge_and_assembly_depths(
        self,
        reducing_end_idx: int | None = None,
    ) -> tuple[int | None, dict[int, int], dict[int, int]]:
        """Identify the inter-repeat `Wzy` polymer edge and biosynthetic assembly depths from the reducing end.

        In a cyclic repeating unit where `r_0` is the biological reducing-end residue transferred to Und-P by
        `INITIATOR`, `r_0`'s anomeric C1 is occupied by Und-PP during intra-unit GT assembly and forms the
        inter-repeat linkage via `Wzy` (`edge.source == r_0`). Removing that `Wzy` edge yields a directed
        assembly tree rooted at `r_0` where each intra-unit GT linkage `k` has assembly step `depth(target) + 1`.
        """
        n = len(self.nodes)
        if n == 0:
            return None, {}, {}
        default_r_idx = self.reducing_end_index
        r_idx = reducing_end_idx if reducing_end_idx is not None else default_r_idx

        wzy_edge_idx: int | None = None
        if r_idx == default_r_idx:
            wzy_edge_idx = self.polymer_edge_index()

        if wzy_edge_idx is None:
            cyc = set(self.backbone_indices()) or set(self.cycle_nodes())
            if cyc and self.has_polymer_repeat:
                anchor = r_idx if r_idx in cyc else (0 if 0 in cyc else min(cyc))
                for k, e in enumerate(self.edges):
                    if (
                        e.source == anchor
                        and e.target in cyc
                        and not self.nodes[e.source].is_modifier
                        and not self.nodes[e.target].is_modifier
                    ):
                        wzy_edge_idx = k
                        break
            elif self.has_polymer_repeat:
                for k, e in enumerate(self.edges):
                    if e.source == 0 and not self.nodes[e.target].is_modifier:
                        wzy_edge_idx = k
                        break

        rev_adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
        for k, e in enumerate(self.edges):
            if k == wzy_edge_idx:
                continue
            if 0 <= e.source < n and 0 <= e.target < n:
                if not self.nodes[e.source].is_modifier and not self.nodes[e.target].is_modifier:
                    rev_adj[e.target].append((e.source, k))

        node_depths: dict[int, int] = {r_idx: 0}
        edge_steps: dict[int, int] = {}
        queue: list[int] = [r_idx]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            d_u = node_depths[u]
            for v_donor, k_edge in rev_adj.get(u, []):
                edge_steps[k_edge] = d_u + 1
                if v_donor not in node_depths:
                    node_depths[v_donor] = d_u + 1
                    queue.append(v_donor)

        for k, e in enumerate(self.edges):
            if k == wzy_edge_idx or k in edge_steps:
                continue
            if not self.nodes[e.source].is_modifier and not self.nodes[e.target].is_modifier:
                d_t = node_depths.get(e.target, 1)
                edge_steps[k] = d_t + 1
                node_depths.setdefault(e.source, d_t + 1)

        return wzy_edge_idx, node_depths, edge_steps

    @property
    def is_alditol(self) -> bool:
        """Returns True if the reducing end is an alditol (has an 'ol' modifier)."""
        ol_idx = Sugars.name_to_index.get(b"ol", -1)
        for edge in self.edges:
            if edge.target == 0 and (self.nodes[edge.source] & 0xFF) == ol_idx:
                return True
        return False

    @property
    def num_modifiers(self) -> int:
        """Number of non-monosaccharide modifier nodes (`OAc`, `Pyr`, `Gro`, `P`, etc.) in this glycan."""
        return sum(1 for n in self.nodes if n.is_modifier)

    @property
    def sugar_mask(self) -> int:
        """Bitmask of all `Sugars` indices present in `self.nodes`."""
        mask = 0
        for n in self.nodes:
            mask |= 1 << n.sugar
        return mask

    @property
    def core_sugar_mask(self) -> int:
        """Bitmask of all non-modifier core monosaccharide `Sugars` indices in `self.nodes`."""
        mask = 0
        for n in self.nodes:
            if not n.is_modifier:
                mask |= 1 << n.sugar
        return mask

    @property
    def has_polymer_repeat(self) -> bool:
        """True if this glycan has a polymer wrap-around edge originating from the root node (`source == 0` or `is_polymer`)."""
        return any(e.edge_type.is_polymer or e.source == 0 for e in self.edges)

    def without_alditol(self) -> Glycan:
        """Return a copy of this `Glycan` with any reducing-end `-ol` alditol modifier node removed."""
        if not self.is_alditol:
            return self
        ol_idx = Sugars.name_to_index.get(b"ol", -1)
        drop_nodes = {e.source for e in self.edges if e.target == 0 and (self.nodes[e.source] & 0xFF) == ol_idx}
        if not drop_nodes:
            return self
        remap: dict[int, int] = {}
        new_nodes: list[ResidueType] = []
        for idx, n in enumerate(self.nodes):
            if idx not in drop_nodes:
                remap[idx] = len(new_nodes)
                new_nodes.append(n)
        new_edges = tuple(
            LinkageEdge(source=remap[e.source], target=remap[e.target], edge_type=e.edge_type)
            for e in self.edges
            if e.source in remap and e.target in remap
        )
        return Glycan(canonical=self.canonical, nodes=tuple(new_nodes), edges=new_edges)

    def without_modifiers(self) -> Glycan:
        """Return the core monosaccharide subgraph of this `Glycan` with all modifier nodes stripped."""
        remap: dict[int, int] = {}
        new_nodes: list[ResidueType] = []
        for idx, n in enumerate(self.nodes):
            if not n.is_modifier:
                remap[idx] = len(new_nodes)
                new_nodes.append(n)
        if len(new_nodes) == len(self.nodes):
            return self
        new_edges = tuple(
            LinkageEdge(source=remap[e.source], target=remap[e.target], edge_type=e.edge_type)
            for e in self.edges
            if e.source in remap and e.target in remap
        )
        return Glycan(canonical=self.canonical, nodes=tuple(new_nodes), edges=new_edges)

    def without_polymer_edge(self) -> Glycan:
        """Return a copy of this `Glycan` without the polymer wrap-around edge (`source == 0` or `is_polymer`)."""
        if not self.has_polymer_repeat:
            return self
        new_edges = tuple(
            LinkageEdge(source=e.source, target=e.target, edge_type=e.edge_type.with_topology(is_polymer=False))
            for e in self.edges
            if not e.edge_type.is_polymer and e.source != 0
        )
        return Glycan(canonical=self.canonical, nodes=self.nodes, edges=new_edges)

    def is_isomorphic(
        self,
        other: Glycan,
        *,
        ignore_reducing_end: bool = True,
        ignore_modifiers: bool = False,
        allow_open_repeat: bool = True,
        allow_wildcards: bool = False,
    ) -> bool:
        """Return True if `self` and `other` are topologically isomorphic directed glycan graphs.

        Supports cyclic-shift invariance on polymer repeat units (`ignore_reducing_end=True`),
        core-backbone comparison (`ignore_modifiers=True`), and matching closed CSDB repeat
        cycles against open-chain IUPAC repeat units (`allow_open_repeat=True`).
        """
        g1 = self.without_modifiers() if ignore_modifiers else self
        g2 = other.without_modifiers() if ignore_modifiers else other

        if _match_glycan_subgraph(
            g1,
            g2,
            ignore_reducing_end=ignore_reducing_end,
            allow_wildcards=allow_wildcards,
            require_bijection=True,
        ):
            return True

        if allow_open_repeat and len(g1.nodes) == len(g2.nodes):
            if g1.has_polymer_repeat and not g2.has_polymer_repeat:
                return _match_glycan_subgraph(
                    g1.without_polymer_edge(),
                    g2,
                    ignore_reducing_end=ignore_reducing_end,
                    allow_wildcards=allow_wildcards,
                    require_bijection=True,
                )
            if g2.has_polymer_repeat and not g1.has_polymer_repeat:
                return _match_glycan_subgraph(
                    g1,
                    g2.without_polymer_edge(),
                    ignore_reducing_end=ignore_reducing_end,
                    allow_wildcards=allow_wildcards,
                    require_bijection=True,
                )
        return False

    def wwl_distance(self, other: Glycan, iterations: int = 2) -> float:
        """Compute the Wasserstein Weisfeiler-Lehman (WWL) graph distance between `self` and `other`."""
        if len(self.nodes) == 0 and len(other.nodes) == 0:
            return 0.0
        if len(self.nodes) == 0 or len(other.nodes) == 0:
            return 1.0
        batch = Glycans.from_list([self, other])
        return float(batch.wwl_distance(iterations=iterations)[0, 1])

    def distance(self, other: Glycan, iterations: int = 2) -> float:
        """Alias for `wwl_distance(other, iterations=iterations)`."""
        return self.wwl_distance(other, iterations=iterations)

    def is_subgraph_of(
        self,
        other: Glycan,
        *,
        ignore_reducing_end: bool = True,
        ignore_alditol: bool = True,
        allow_wildcards: bool = True,
    ) -> bool:
        """Return True if `self` is a topological subgraph (fragment or un-decorated variant) of `other`."""
        sub = self.without_alditol() if (ignore_alditol and self.is_alditol and not other.is_alditol) else self
        return _match_glycan_subgraph(
            sub,
            other,
            ignore_reducing_end=ignore_reducing_end,
            allow_wildcards=allow_wildcards,
            require_bijection=False,
        )

    def is_oligomer_of(
        self,
        repeating_unit: Glycan,
        *,
        allow_wildcards: bool = True,
    ) -> bool:
        """Return True if `self` is a multi-repeat oligomer (`2x`/`3x` dimer/trimer) of `repeating_unit`."""
        sub = self.without_alditol()
        core_self = sub.without_modifiers()
        core_rep = repeating_unit.without_modifiers()
        if len(core_self.nodes) <= len(core_rep.nodes) or len(core_rep.nodes) < 2:
            return False

        # 1. The open 1x repeating unit core must be a subgraph of the oligomer core
        if not _match_glycan_subgraph(
            core_rep.without_polymer_edge(),
            core_self,
            ignore_reducing_end=True,
            allow_wildcards=allow_wildcards,
            require_bijection=False,
        ):
            return False

        # 2. Every residue and directed (donor_residue, linkage, acceptor_residue) transition in self
        #    must exist in repeating_unit
        for n_s in core_self.nodes:
            if not any(
                _residue_compatible(n_s, n_r, ignore_reducing_end=True, allow_wildcards=allow_wildcards)
                for n_r in core_rep.nodes
            ):
                return False

        for e_s in core_self.edges:
            s_src = core_self.nodes[e_s.source]
            s_tgt = core_self.nodes[e_s.target]
            if not any(
                _linkage_compatible(e_s.edge_type, e_r.edge_type, allow_wildcards=allow_wildcards)
                and _residue_compatible(
                    s_src, core_rep.nodes[e_r.source], ignore_reducing_end=True, allow_wildcards=allow_wildcards
                )
                and _residue_compatible(
                    s_tgt, core_rep.nodes[e_r.target], ignore_reducing_end=True, allow_wildcards=allow_wildcards
                )
                for e_r in core_rep.edges
            ):
                return False

        return True

    def graft_modifiers_from(
        self,
        donor: Glycan,
        *,
        allow_wildcards: bool = True,
    ) -> tuple[Glycan, tuple[bytes, ...]]:
        """Graft missing modifier nodes (`OAc`, `Pyr`, `OMe`, `Fo`, etc.) from `donor` onto `self`.

        Aligns the core monosaccharide subgraph of `donor` onto `self` and transfers any
        non-alditol modifier substituents (preserving their linkage carbons) that are absent
        on the corresponding core residue in `self`.
        """
        if donor.num_modifiers == 0:
            return self, ()

        ol_idx = Sugars.name_to_index.get(b"ol", -1)
        donor_core = donor.without_alditol().without_modifiers()
        self_core = self.without_modifiers()

        core_map = _find_glycan_subgraph_mapping(
            donor_core,
            self_core,
            ignore_reducing_end=True,
            allow_wildcards=allow_wildcards,
            require_bijection=False,
        )
        if core_map is None and donor_core.has_polymer_repeat:
            core_map = _find_glycan_subgraph_mapping(
                donor_core.without_polymer_edge(),
                self_core,
                ignore_reducing_end=True,
                allow_wildcards=allow_wildcards,
                require_bijection=False,
            )
        if core_map is None:
            return self, ()

        donor_orig_to_core: dict[int, int] = {}
        c_idx = 0
        for idx, n in enumerate(donor.nodes):
            if not n.is_modifier:
                donor_orig_to_core[idx] = c_idx
                c_idx += 1

        self_core_to_orig: list[int] = [idx for idx, n in enumerate(self.nodes) if not n.is_modifier]

        new_nodes = list(self.nodes)
        new_edges = list(self.edges)
        grafted_names: list[bytes] = []

        for m_idx, m_res in enumerate(donor.nodes):
            if not m_res.is_modifier or m_res.sugar == ol_idx:
                continue
            mod_edges = [e for e in donor.edges if e.source == m_idx and not donor.nodes[e.target].is_modifier]
            if not mod_edges:
                continue
            if not all(e.target in donor_orig_to_core and donor_orig_to_core[e.target] in core_map for e in mod_edges):
                continue

            mapped_targets = [self_core_to_orig[core_map[donor_orig_to_core[e.target]]] for e in mod_edges]
            primary_t = mapped_targets[0]
            req_carbons = {e.edge_type.acceptor for e in mod_edges}

            # Check if primary_t in self already carries this modifier at an overlapping or wildcard carbon
            already_has = False
            for ex_e in new_edges:
                if ex_e.target == primary_t and new_nodes[ex_e.source].sugar == m_res.sugar:
                    ex_acc = ex_e.edge_type.acceptor
                    if ex_acc == 0 or 0 in req_carbons or ex_acc in req_carbons:
                        already_has = True
                        break
            if already_has:
                continue

            new_mod_idx = len(new_nodes)
            new_nodes.append(m_res)
            for me, tgt_self in zip(mod_edges, mapped_targets, strict=False):
                new_edges.append(LinkageEdge(source=new_mod_idx, target=tgt_self, edge_type=me.edge_type))
            grafted_names.append(Sugars.names[m_res.sugar])

        if not grafted_names:
            return self, ()
        return Glycan(canonical=self.canonical, nodes=tuple(new_nodes), edges=tuple(new_edges)), tuple(grafted_names)


@dataclass(slots=True, frozen=True)
class Glycans(GraphBatch[Glycan, "Glycans"]):
    """
    Structure-of-Arrays (SoA) batch container for glycan datasets.
    """

    node_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    node_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_sources: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_targets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    canonicals: np.ndarray[tuple[int], np.dtype[np.bytes_]]

    def __post_init__(self) -> None:
        if not isinstance(self.canonicals, np.ndarray):
            arr = np.array(self.canonicals, dtype="S") if len(self.canonicals) > 0 else np.empty(0, dtype="S1")
            object.__setattr__(self, "canonicals", arr)

    def __len__(self) -> int:
        return len(self.node_offsets) - 1

    @classmethod
    def empty(cls) -> Self:
        """Creates an empty 0-length Glycans batch."""
        return cls(
            node_types=np.empty(0, dtype=np.uint32),
            node_offsets=np.zeros(1, dtype=np.uint32),
            edge_sources=np.empty(0, dtype=np.uint32),
            edge_targets=np.empty(0, dtype=np.uint32),
            edge_types=np.empty(0, dtype=np.uint32),
            edge_offsets=np.zeros(1, dtype=np.uint32),
            canonicals=np.empty(0, dtype="S1"),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenates multiple Glycans batches into a single contiguous Glycans batch."""
        batch_list: list[Self] = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty()
        if len(batch_list) == 1:
            return batch_list[0]

        nodes, n_offs, sources, targets, types, e_offs = cls._concat_soa_base(batch_list)
        canonicals = np.concatenate([b.canonicals for b in batch_list])

        return cls(
            node_types=nodes,
            node_offsets=n_offs,
            edge_sources=sources,
            edge_targets=targets,
            edge_types=types,
            edge_offsets=e_offs,
            canonicals=canonicals,
        )

    @classmethod
    def from_list(cls, glycans: Sequence[Glycan]) -> Self:
        """Compiles a sequence of individual Glycans into SoA batch format."""
        num_glycans = len(glycans)
        if num_glycans == 0:
            return cls.empty()

        node_offsets = np.empty(num_glycans + 1, dtype=np.uint32)
        node_offsets[0] = 0
        edge_offsets = np.empty(num_glycans + 1, dtype=np.uint32)
        edge_offsets[0] = 0

        tot_nodes = sum(len(g.nodes) for g in glycans)
        tot_edges = sum(len(g.edges) for g in glycans)

        node_types = np.empty(tot_nodes, dtype=np.uint32)
        edge_sources = np.empty(tot_edges, dtype=np.uint32)
        edge_targets = np.empty(tot_edges, dtype=np.uint32)
        edge_types = np.empty(tot_edges, dtype=np.uint32)
        canonicals: list[bytes] = []

        curr_node = 0
        curr_edge = 0

        for i, g in enumerate(glycans):
            n_nodes = len(g.nodes)
            n_edges = len(g.edges)

            if n_nodes > 0:
                node_types[curr_node : curr_node + n_nodes] = [int(n) for n in g.nodes]

            if n_edges > 0:
                for j, b in enumerate(g.edges):
                    edge_sources[curr_edge + j] = b.source + curr_node
                    edge_targets[curr_edge + j] = b.target + curr_node
                    edge_types[curr_edge + j] = int(b.edge_type)

            curr_node += n_nodes
            curr_edge += n_edges
            node_offsets[i + 1] = curr_node
            edge_offsets[i + 1] = curr_edge
            canonicals.append(g.canonical)

        return cls(
            node_types=node_types,
            node_offsets=node_offsets,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_types=edge_types,
            edge_offsets=edge_offsets,
            canonicals=np.array(canonicals, dtype="S"),
        )

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> Glycan: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Glycan | Self: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Glycan | Self:
        """
        Extracts a single Glycan by integer index, or a sub-batch by slice / boolean mask / integer index array.
        """
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("Glycans index out of range")

            m_start, m_end = self.node_offsets[idx], self.node_offsets[idx + 1]
            e_start, e_end = self.edge_offsets[idx], self.edge_offsets[idx + 1]

            nodes = tuple(ResidueType(x) for x in self.node_types[m_start:m_end])
            edges = tuple(
                LinkageEdge(
                    source=int(self.edge_sources[i]) - int(m_start),
                    target=int(self.edge_targets[i]) - int(m_start),
                    edge_type=LinkageType(self.edge_types[i]),
                )
                for i in range(e_start, e_end)
            )
            return Glycan(canonical=bytes(self.canonicals[idx]), nodes=nodes, edges=edges)

        if isinstance(item, slice):
            kept_indices = np.arange(len(self), dtype=np.int32)[item]
        elif isinstance(item, (np.ndarray, Sequence)):
            arr = np.asarray(item)
            if arr.dtype == bool:
                if len(arr) != len(self):
                    raise ValueError(f"Mask length {len(arr)} does not match batch length {len(self)}.")
                kept_indices = np.nonzero(arr)[0].astype(np.int32)
            elif np.issubdtype(arr.dtype, np.integer) or len(arr) == 0:
                kept_indices = self._normalize_indices(arr)
            else:
                raise TypeError(f"Unsupported array dtype for Glycans indexing: {arr.dtype}")
        else:
            raise TypeError(
                f"Only integers, slices, and 1D boolean/integer arrays are supported for indexing (got {type(item)})."
            )

        if len(kept_indices) == 0:
            return self.empty()

        new_canonicals = self.canonicals[kept_indices]
        new_nodes, new_node_offsets, new_sources, new_targets, new_types, new_edge_offsets = self._slice_soa_arrays(
            kept_indices
        )
        return self.__class__(
            node_types=new_nodes,
            node_offsets=new_node_offsets,
            edge_sources=new_sources,
            edge_targets=new_targets,
            edge_types=new_types,
            edge_offsets=new_edge_offsets,
            canonicals=new_canonicals,
        )

    @property
    def is_alditol(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Returns a boolean array indicating if each glycan's reducing end is an alditol."""
        N = len(self)
        result = np.zeros(N, dtype=bool)
        if N == 0 or len(self.edge_types) == 0:
            return result

        ol_idx = Sugars.name_to_index.get(b"ol", -1)
        is_ol_node = (self.node_types & 0xFF) == ol_idx

        glycan_indices = np.repeat(np.arange(N), np.diff(self.edge_offsets))
        global_sources = self.edge_sources
        targets_root = self.edge_targets == self.node_offsets[glycan_indices]

        valid_edges = targets_root & is_ol_node[global_sources]
        result[glycan_indices[valid_edges]] = True
        return result

    def wwl_distance(self, iterations: int = 2) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Computes the Wasserstein Weisfeiler-Lehman (WWL) continuous distance matrix.
        Uses the node features from Sugars and topologically smooths them
        over the core edges, then computes the Earth Mover's Distance between
        the resulting point clouds.

        Constructs an unweighted, symmetrized adjacency matrix using :class:`scipy.sparse.csr_array`,
        normalizes by inverse degree using :func:`scipy.sparse.diags`, and computes pairwise
        Wasserstein distances between node feature distributions using
        :func:`scipy.stats.wasserstein_distance_nd`.

        Args:
            iterations: Number of Weisfeiler-Lehman neighborhood smoothing iterations.

        Returns:
            A symmetric (N x N) distance matrix (`np.ndarray[tuple[int, int], np.dtype[np.float64]]`).
        """
        N = len(self.node_types)
        if N == 0:
            return np.zeros((len(self), len(self)), dtype=np.float64)

        # 1. Get initial features
        sugar_indices = self.node_types & 0xFF
        features = Sugars.to_feature_matrix()[sugar_indices]

        # 2. Build undirected batch adjacency matrix for WWL smoothing
        adj_indices = np.concatenate([self.edge_sources, self.edge_targets])
        adj_indptr = np.concatenate([self.edge_targets, self.edge_sources])

        adj = csr_array((np.ones_like(adj_indices), (adj_indices, adj_indptr)), shape=(N, N))

        deg = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
        deg[deg == 0] = 1.0  # avoid div by zero
        deg_inv = diags(1.0 / deg)
        adj_norm = deg_inv @ adj

        # 3. Iteratively smooth features over core
        embeddings = [features]
        current_h = features
        for _ in range(iterations):
            current_h = 0.5 * current_h + 0.5 * (adj_norm @ current_h)
            embeddings.append(current_h)

        final_embeddings = np.concatenate(embeddings, axis=1)

        # 4. Compute pairwise WWL distances
        n_glycans = len(self)
        dist_matrix = np.zeros((n_glycans, n_glycans), dtype=np.float64)

        for i in range(n_glycans):
            start_i = self.node_offsets[i]
            end_i = self.node_offsets[i + 1]
            X_i = final_embeddings[start_i:end_i]
            for j in range(i + 1, n_glycans):
                start_j = self.node_offsets[j]
                end_j = self.node_offsets[j + 1]
                X_j = final_embeddings[start_j:end_j]

                d = wasserstein_distance_nd(X_i, X_j)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix
