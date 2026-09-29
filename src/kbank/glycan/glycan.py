"""Module for representing a glycan and glycan batches."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any, Self

import numpy as np
from kaptive.core.collections import BatchedContainer
from scipy.sparse import csr_array, diags
from scipy.stats import wasserstein_distance_nd

from kbank.core.graph import BitField, DirectedEdge, GraphBatch, GraphModel, IntNode, TAG_EDGE
from kbank.glycan.sugars import Sugars


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
    Bit 31: Tag bit (0 = primary node, 1 = intermediate edge node)
    """
    SUGAR = BitField(shift=0, width=8)
    CHIRALITY = BitField(shift=8, width=2)
    FURANOSE = BitField(shift=10, width=1)
    REDUCING_END = BitField(shift=11, width=1)
    TAG_EDGE = BitField(shift=31, width=1)

    def __new__(cls, val_or_sugar: int | bytes | bytearray, chirality: Chirality = Chirality.UNKNOWN, is_furanose: bool = False, is_reducing_end: bool = False):
        if isinstance(val_or_sugar, (int, np.integer)):
            return super().__new__(cls, int(val_or_sugar))
        sugar_idx = Sugars.index(val_or_sugar)
        packed = cls.pack(
            sugar=sugar_idx,
            chirality=int(chirality),
            is_furanose=is_furanose,
            is_reducing_end=is_reducing_end,
        )
        return super().__new__(cls, packed)

    @classmethod
    def pack(cls, sugar: int, chirality: int = 0, is_furanose: bool = False, is_reducing_end: bool = False) -> int:
        return (
            cls.SUGAR.pack(sugar)
            | cls.CHIRALITY.pack(chirality)
            | cls.FURANOSE.pack(int(is_furanose))
            | cls.REDUCING_END.pack(int(is_reducing_end))
        )

    @classmethod
    def from_sugar(
        cls,
        sugar_idx: int,
        chirality: Chirality | int = Chirality.UNKNOWN,
        is_furanose: bool = False,
        is_reducing_end: bool = False,
    ) -> ResidueType:
        chi_val = chirality.value if isinstance(chirality, Chirality) else int(chirality)
        val = cls.pack(
            sugar=sugar_idx,
            chirality=chi_val,
            is_furanose=is_furanose,
            is_reducing_end=is_reducing_end,
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

    def with_reducing_end(self, is_reducing: bool = True) -> ResidueType:
        """Returns a copy of this ResidueType with the reducing end bit set/cleared."""
        mask = type(self).REDUCING_END.mask
        if is_reducing:
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
    Bits 10-30: Reserved
    Bit 31:     TAG_EDGE (1 = Edge node in bipartite graphs)
    """

    ANOMER = BitField(shift=0, width=2)
    DONOR = BitField(shift=2, width=4)
    ACCEPTOR = BitField(shift=6, width=4)
    TAG_EDGE = BitField(shift=31, width=1)

    OFFSET: int = TAG_EDGE

    def __new__(
        cls,
        val_or_anomer: int | Anomer | bytes | bytearray,
        donor: int | None = None,
        acceptor: int | None = None,
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
        packed = cls.pack(anomer, d, acc)
        return super().__new__(cls, packed)

    @classmethod
    def pack(
        cls,
        anomer: int | Anomer = Anomer.UNKNOWN,
        donor: int = 0,
        acceptor: int = 0,
    ) -> int:
        """Packs anomer, donor carbon, and acceptor carbon into a 32-bit integer with TAG_EDGE set."""
        return (
            cls.ANOMER.pack(int(anomer))
            | cls.DONOR.pack(int(donor))
            | cls.ACCEPTOR.pack(int(acceptor))
            | cls.TAG_EDGE.pack(1)
        )

    @classmethod
    def from_bytes(cls, link: bytes | bytearray) -> Self:
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
        """Linkage bits without TAG_EDGE."""
        return self & ~TAG_EDGE

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
    def name(self) -> bytes:
        """Byte-string representation (e.g. b'b1-4', b'a1-3')."""
        a_str = b"a" if self.anomer == Anomer.ALPHA else b"b" if self.anomer == Anomer.BETA else b"?"
        d_str = (b"%d" % self.donor) if self.donor != 0 else b"?"
        acc_str = (b"%d" % self.acceptor) if self.acceptor != 0 else b"?"
        return a_str + d_str + b"-" + acc_str

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


# Main container -------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Glycan(GraphModel):
    """Container for the components of a glycan."""
    canonical: bytes
    nodes: tuple[ResidueType, ...]
    edges: tuple[LinkageEdge, ...]

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
    def is_alditol(self) -> bool:
        """Returns True if the reducing end is an alditol (has an 'ol' modifier)."""
        ol_idx = Sugars.name_to_index.get(b'ol', -1)
        for edge in self.edges:
            if edge.target == 0 and (self.nodes[edge.source] & 0xFF) == ol_idx:
                return True
        return False


@dataclass(slots=True, frozen=True)
class Glycans(BatchedContainer[Glycan, "Glycans"], GraphBatch):
    """
    Structure-of-Arrays (SoA) batch container for glycan datasets.
    """
    node_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    node_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_sources: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_targets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    canonicals: tuple[bytes, ...]

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
            canonicals=(),
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
        canonicals = tuple(c for b in batch_list for c in b.canonicals)

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
    def from_list(cls, glycans: list[Glycan]) -> Self:
        """Compiles a list of individual Glycans into the SoA format."""
        num_glycans = len(glycans)
        node_offsets = np.zeros(num_glycans + 1, dtype=np.uint32)
        edge_offsets = np.zeros(num_glycans + 1, dtype=np.uint32)
        
        all_monos = []
        all_sources = []
        all_targets = []
        all_links = []
        canonicals = []
        
        mono_offset = 0
        for i, g in enumerate(glycans):
            all_monos.extend(g.nodes)
            
            for b in g.edges:
                all_sources.append(b.source + mono_offset)
                all_targets.append(b.target + mono_offset)
                all_links.append(b.edge_type)
                
            mono_offset += len(g.nodes)
            node_offsets[i + 1] = mono_offset
            edge_offsets[i + 1] = len(all_links)
            canonicals.append(g.canonical)
            
        return cls(
            node_types=np.array(all_monos, dtype=np.uint32),
            node_offsets=node_offsets,
            edge_sources=np.array(all_sources, dtype=np.uint32),
            edge_targets=np.array(all_targets, dtype=np.uint32),
            edge_types=np.array(all_links, dtype=np.uint32),
            edge_offsets=edge_offsets,
            canonicals=tuple(canonicals)
        )

    def __getitem__(self, item: Any) -> Glycan | Self:
        """
        Extracts a single Glycan by integer index, or a sub-batch by slice / boolean mask.
        """
        if isinstance(item, int):
            idx = item
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
                    edge_type=LinkageType(self.edge_types[i])
                ) for i in range(e_start, e_end)
            )
            return Glycan(canonical=self.canonicals[idx], nodes=nodes, edges=edges)

        if isinstance(item, slice):
            kept_indices = np.arange(len(self))[item]
        elif isinstance(item, np.ndarray) and item.dtype == bool:
            if len(item) != len(self):
                raise ValueError(f"Mask length {len(item)} does not match batch length {len(self)}.")
            kept_indices = np.where(item)[0]
        else:
            raise TypeError(f"Only integers, slices, and 1D boolean numpy arrays are supported for indexing (got {type(item)}).")

        new_canonicals = tuple(self.canonicals[i] for i in kept_indices)
        new_nodes, new_node_offsets, new_sources, new_targets, new_types, new_edge_offsets = self._slice_soa_arrays(kept_indices)
        return self.__class__(
            node_types=new_nodes,
            node_offsets=new_node_offsets,
            edge_sources=new_sources,
            edge_targets=new_targets,
            edge_types=new_types,
            edge_offsets=new_edge_offsets,
            canonicals=new_canonicals
        )

    @property
    def is_alditol(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Returns a boolean array indicating if each glycan's reducing end is an alditol."""
        N = len(self)
        result = np.zeros(N, dtype=bool)
        if N == 0 or len(self.edge_types) == 0:
            return result
            
        ol_idx = Sugars.name_to_index.get(b'ol', -1)
        is_ol_node = (self.node_types & 0xFF) == ol_idx
        
        glycan_indices = np.repeat(np.arange(N), np.diff(self.edge_offsets))
        global_sources = self.edge_sources
        targets_root = (self.edge_targets == self.node_offsets[glycan_indices])
        
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
        features = Sugars().to_feature_matrix()[sugar_indices]

        # 2. Build undirected batch adjacency matrix for WWL smoothing
        adj_indices = np.concatenate([self.edge_sources, self.edge_targets])
        adj_indptr = np.concatenate([self.edge_targets, self.edge_sources])

        adj = csr_array(
            (np.ones_like(adj_indices), (adj_indices, adj_indptr)),
            shape=(N, N)
        )

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
