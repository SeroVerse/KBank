"""Bacterial Biosynthetic Gene Cluster (BGC) core models, registries, and SoA containers."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import threading
from typing import Any, Self

import numpy as np
from kaptive.core.collections import BatchedContainer
from kaptive.core.interval import Intervals, Strand

from kbank.core.graph import TAG_EDGE, BitField, DirectedEdge, GraphBatch, GraphModel, IntNode
from kbank.genes.roles import GeneRoles, TargetType


# Classes --------------------------------------------------------------------------------------------------------------
class GeneClusterRegistry:
    """Thread-safe registry mapping (role_idx, gene_name) to an incremental cluster integer.

    Assigns each unique gene name within a role an incremental cluster integer (0, 1, 2, ...).
    """
    __slots__ = ("_lock", "_role_to_gene", "_role_to_id")

    def __init__(self) -> None:
        """Initializes an empty GeneClusterRegistry."""
        self._role_to_gene: dict[int, dict[bytes, int]] = {}
        self._role_to_id: dict[int, dict[int, bytes]] = {}
        self._lock: threading.Lock = threading.Lock()

    def get_or_create(self, role: int, gene: bytes | str) -> int:
        """Retrieves existing cluster ID or registers a new incremental integer for the given role and gene name."""
        gene_bytes = gene.encode("ascii") if isinstance(gene, str) else bytes(gene)
        gene_clean = gene_bytes.strip().lower()
        with self._lock:
            mapping = self._role_to_gene.setdefault(role, {})
            if gene_clean in mapping:
                return mapping[gene_clean]
            new_id = len(mapping)
            if new_id > 32767:
                raise ValueError(f"Cluster ID {new_id} exceeds maximum 15-bit capacity (32,767) for role {role}")
            mapping[gene_clean] = new_id
            self._role_to_id.setdefault(role, {})[new_id] = gene_clean
            return new_id

    def get(self, role: int, gene: bytes | str, default: int | None = None) -> int | None:
        """Looks up the assigned cluster ID for a role and gene name, if registered."""
        gene_bytes = gene.encode("ascii") if isinstance(gene, str) else bytes(gene)
        gene_clean = gene_bytes.strip().lower()
        with self._lock:
            mapping = self._role_to_gene.get(role)
            if mapping is None:
                return default
            return mapping.get(gene_clean, default)

    def get_gene_name(self, role: int, cluster: int) -> bytes | None:
        """Reverse lookup: returns the unique gene name bytes assigned to a role and cluster ID."""
        with self._lock:
            rev_map = self._role_to_id.get(role)
            if rev_map is None:
                return None
            return rev_map.get(cluster)

    def node_for(self, role: int, gene: bytes | str) -> GeneNode:
        """Returns a bit-packed GeneNode for the specified role and gene symbol."""
        cid = self.get_or_create(role, gene)
        return GeneNode(GeneNode.pack(role=role, cluster=cid))

    def gene_name_for_node(self, node: GeneNode) -> bytes | None:
        """Returns the gene name bytes associated with a GeneNode's role and cluster."""
        return self.get_gene_name(role=node.role, cluster=node.cluster)

    def cluster_count(self, role: int) -> int:
        """Returns the number of unique clusters/alleles registered under the specified role."""
        with self._lock:
            mapping = self._role_to_gene.get(role)
            return len(mapping) if mapping is not None else 0

    @property
    def total_clusters(self) -> int:
        """Total number of unique gene alleles registered across all roles."""
        with self._lock:
            return sum(len(m) for m in self._role_to_gene.values())

    def __len__(self) -> int:
        """Number of total clusters registered."""
        return self.total_clusters

    def clear(self) -> None:
        """Resets all registered mappings."""
        with self._lock:
            self._role_to_gene.clear()
            self._role_to_id.clear()

    def to_dict(self) -> dict[tuple[int, bytes], int]:
        """Exports a copy of all registered mappings as a (role_idx, gene_name) -> cluster_id dict."""
        with self._lock:
            out: dict[tuple[int, bytes], int] = {}
            for r, g_map in self._role_to_gene.items():
                for g, cid in g_map.items():
                    out[(r, g)] = cid
            return out


class GeneNode(IntNode):
    """Unified bit-packed integer for Gene nodes in BGC graphs.

    Bits 0-7: Role index (0..255 into GeneRoles)
    Bits 8-11: Locus type (0 = Primary locus, 1..15 = Auxiliary loci 1..15)
    Bits 12-15: Reserved (0)
    Bits 16-30: Cluster ID (0..32,767)
    Bit 31: Tag bit (0 = Node, 1 = Edge).
    """

    ROLE = BitField(shift=0, width=8)
    LOCUS_TYPE = BitField(shift=8, width=4)
    RESERVED = BitField(shift=12, width=4)
    CLUSTER = BitField(shift=16, width=15)
    TAG_EDGE = BitField(shift=31, width=1)

    @classmethod
    def pack(cls, role: int, cluster: int = 0, locus_type: int = 0) -> int:
        """Packs role, cluster, and locus_type into a single 32-bit integer."""
        return cls.ROLE.pack(role) | cls.LOCUS_TYPE.pack(locus_type) | cls.CLUSTER.pack(cluster)

    @classmethod
    def from_role(
        cls,
        role: int | bytes | str | GeneNode,
        cluster: int | None = None,
        registry: GeneClusterRegistry | None = None,
        roles_db: GeneRoles | None = None,
        description: bytes | str | None = None,
        locus_type: int = 0,
    ) -> Self:
        """Constructs a GeneNode from a role integer, string, bytes, or GeneNode."""
        if isinstance(role, cls):
            c = cluster if cluster is not None else role.cluster
            lt = locus_type if locus_type != 0 else role.locus_type
            return cls(cls.pack(role.role, cluster=c, locus_type=lt))
        if isinstance(role, (bytes, bytearray, str)):
            sym_bytes = role.encode("ascii") if isinstance(role, str) else bytes(role)
            gene_symbol = sym_bytes
            db = roles_db or GeneRoles()
            if sym_bytes in db.name_to_index:
                role_idx = db.name_to_index[sym_bytes]
            else:
                desc_bytes = description.encode("utf-8") if isinstance(description, str) else (
                    bytes(description) if description is not None else None
                )
                role_idx = db.label(symbol=sym_bytes, description=desc_bytes)
        else:
            role_idx = int(role)
            gene_symbol = b""

        if cluster is not None:
            cid = int(cluster)
        elif registry is not None:
            cid = registry.get_or_create(role_idx, gene_symbol)
        else:
            cid = 0

        return cls(cls.pack(role_idx, cluster=cid, locus_type=locus_type))

    @property
    def role(self) -> int:
        """Role index into GeneRoles."""
        return self.ROLE

    @property
    def role_name(self) -> bytes:
        """Name of the role as bytes."""
        return GeneRoles.names[self.role]

    @property
    def target_type(self) -> TargetType:
        """TargetType enum of the role."""
        return GeneRoles.target_types[self.role]

    @property
    def locus_type(self) -> int:
        """Locus type index (0 for primary, >0 for auxiliary)."""
        return self.LOCUS_TYPE

    @property
    def is_auxiliary(self) -> bool:
        """True if node belongs to an auxiliary locus."""
        return self.locus_type > 0

    @property
    def cluster(self) -> int:
        """Cluster ID within this role."""
        return self.CLUSTER

    def __repr__(self) -> str:
        """String representation of GeneNode."""
        role_name = GeneRoles.names[self.role]
        if self.is_auxiliary:
            return (
                b"GeneNode(%b, cluster=%d, locus_type=%d)"
                % (role_name, self.cluster, self.locus_type)
            ).decode("ascii", errors="replace")
        return (b"GeneNode(%b, cluster=%d)" % (role_name, self.cluster)).decode("ascii", errors="replace")


class IntergenicType(int):
    """Unified integer encoding for an intergenic edge in BGC graphs.

    Bit 31 is set (TAG_EDGE) to distinguish from primary GeneNodes.
    """

    SAME_STRAND: IntergenicType
    DEFAULT: IntergenicType

    def __repr__(self) -> str:
        """String representation of IntergenicType."""
        code = self & ~TAG_EDGE
        return f"IntergenicType(code={code})"


IntergenicType.SAME_STRAND = IntergenicType(TAG_EDGE | 1)
IntergenicType.DEFAULT = IntergenicType.SAME_STRAND


@dataclass(slots=True, frozen=True)
class IntergenicEdge(DirectedEdge):
    """Directed syntenic edge between genes in a BGC."""
    source: int
    target: int
    edge_type: IntergenicType = IntergenicType.DEFAULT


@dataclass(slots=True, frozen=True)
class GeneCluster(GraphModel):
    """Graph model representing a single bacterial biosynthetic gene cluster."""
    nodes: tuple[GeneNode, ...]
    edges: tuple[IntergenicEdge, ...]
    gene_names: tuple[str, ...] | None = None
    strands: tuple[int, ...] | None = None
    lengths: tuple[int, ...] | None = None
    registry: GeneClusterRegistry | None = None

    @property
    def primary_nodes(self) -> tuple[GeneNode, ...]:
        """All nodes belonging to the primary locus (locus_type == 0)."""
        return tuple(n for n in self.nodes if not n.is_auxiliary)

    @property
    def auxiliary_nodes(self) -> tuple[GeneNode, ...]:
        """All nodes belonging to auxiliary/satellite loci (locus_type > 0)."""
        return tuple(n for n in self.nodes if n.is_auxiliary)

    @property
    def has_auxiliary(self) -> bool:
        """Indicates whether this cluster contains any auxiliary locus genes."""
        return any(n.is_auxiliary for n in self.nodes)

    @property
    def has_mobile_elements(self) -> bool:
        """Indicates whether this cluster contains any mobile genetic elements (e.g. transposases, integrases)."""
        roles_db = GeneRoles()
        return any(roles_db.categories[n.role] == b"MobileElement" for n in self.nodes)

    def heal_mobile_elements(
        self,
        bypassed_roles: Sequence[int | bytes | str] | None = None,
        *,
        remove_nodes: bool = True,
        roles_db: GeneRoles | None = None,
    ) -> GeneCluster:
        """Heals and bridges an operon interrupted by mobile genetic elements.

        Bypasses mobile elements by connecting non-mobile upstream genes directly to non-mobile downstream genes.

        Args:
            bypassed_roles: Roles to bypass. If None, defaults to all roles with category b'MobileElement'
                           (such as TRANSPOSASE and INTEGRASE).
            remove_nodes: If True, removes the mobile element nodes from the cluster and re-indexes
                          edges. If False, keeps all nodes and adds bridging shortcut edges.
            roles_db: Optional GeneRoles instance. Defaults to GeneRoles().

        Returns:
            A new GeneCluster instance with healed syntenic continuity.
        """
        db = roles_db or GeneRoles()
        if bypassed_roles is None:
            bypassed_set = set(np.where(db.categories == b"MobileElement")[0])
        else:
            bypassed_set = set()
            for r in bypassed_roles:
                if isinstance(r, int):
                    bypassed_set.add(r)
                elif isinstance(r, (bytes, str)):
                    r_bytes = r.encode() if isinstance(r, str) else r
                    if r_bytes in db.name_to_index:
                        bypassed_set.add(db.name_to_index[r_bytes])
                    elif r_bytes in db.alias_map:
                        bypassed_set.add(db.name_to_index[db.alias_map[r_bytes]])

        N = len(self.nodes)
        if N == 0:
            return self

        is_mobile = [n.role in bypassed_set for n in self.nodes]
        if not any(is_mobile):
            return self

        # Build adjacency core
        adj: dict[int, list[int]] = {}
        for e in self.edges:
            adj.setdefault(e.source, []).append(e.target)

        if remove_nodes:
            kept = [i for i in range(N) if not is_mobile[i]]
            if not kept:
                return GeneCluster(nodes=(), edges=())

            old_to_new = {old: new for new, old in enumerate(kept)}
            new_edges_set: set[tuple[int, int]] = set()
            new_edges: list[IntergenicEdge] = []

            for u in kept:
                visited: set[int] = set()
                queue = list(adj.get(u, []))
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    if not is_mobile[curr]:
                        edge_pair = (old_to_new[u], old_to_new[curr])
                        if edge_pair not in new_edges_set:
                            new_edges_set.add(edge_pair)
                            new_edges.append(IntergenicEdge(source=edge_pair[0], target=edge_pair[1]))
                    else:
                        queue.extend(adj.get(curr, []))

            new_nodes = tuple(self.nodes[i] for i in kept)
            new_names = tuple(self.gene_names[i] for i in kept) if self.gene_names is not None else None
            new_strands = tuple(self.strands[i] for i in kept) if self.strands is not None else None
            new_lengths = tuple(self.lengths[i] for i in kept) if self.lengths is not None else None
            return GeneCluster(
                nodes=new_nodes,
                edges=tuple(new_edges),
                gene_names=new_names,
                strands=new_strands,
                lengths=new_lengths,
            )
        else:
            # Keep nodes, add shortcut edges
            existing_edges_set = {(e.source, e.target) for e in self.edges}
            extended_edges = list(self.edges)

            for u in range(N):
                if is_mobile[u]:
                    continue
                visited: set[int] = set()
                queue = list(adj.get(u, []))
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    if not is_mobile[curr]:
                        edge_pair = (u, curr)
                        if edge_pair not in existing_edges_set:
                            existing_edges_set.add(edge_pair)
                            extended_edges.append(IntergenicEdge(source=u, target=curr))
                    else:
                        queue.extend(adj.get(curr, []))

            return GeneCluster(
                nodes=self.nodes,
                edges=tuple(extended_edges),
                gene_names=self.gene_names,
                strands=self.strands,
                lengths=self.lengths,
            )

    @classmethod
    def create_intergenic_edges(
        cls,
        strands: Intervals | Sequence[Any],
        edge_type: IntergenicType = IntergenicType.DEFAULT,
    ) -> tuple[IntergenicEdge, ...]:
        """Creates directed syntenic edges between adjacent genes.

        Transcriptional flow: adjacent genes on the forward strand flow i -> i+1;
        adjacent genes on the reverse strand flow i+1 -> i. Genes with orientation
        switches or unstranded orientation remain disconnected.

        Args:
            strands: Sequence of strand indicators (or an Intervals collection) in chromosomal order.
            edge_type: The IntergenicType to assign to each created edge.

        Returns:
            A tuple of IntergenicEdge instances.
        """
        raw = strands.strands if isinstance(strands, Intervals) else strands
        N = len(raw)
        if N < 2:
            return ()

        s = [Strand(x) for x in raw]
        edges: list[IntergenicEdge] = []
        for i in range(N - 1):
            if s[i] == s[i + 1] and s[i] != Strand.UNSTRANDED:
                src, tgt = (i, i + 1) if s[i] == Strand.FORWARD else (i + 1, i)
                edges.append(IntergenicEdge(source=src, target=tgt, edge_type=edge_type))
        return tuple(edges)

    @classmethod
    def from_genes(
        cls,
        roles: Sequence[int | bytes | str | GeneNode] | None = None,
        strands: Intervals | Sequence[Any] | None = None,
        clusters: Sequence[int] | None = None,
        *,
        descriptions: Sequence[bytes | str] | None = None,
        lengths: Sequence[int] | None = None,
        registry: GeneClusterRegistry | None = None,
        roles_db: GeneRoles | None = None,
        locus_type: int = 0,
    ) -> Self:
        """Constructs a GeneCluster from sequences of gene symbols/roles.

        When registry is provided (or when clusters is None), assigns each unique gene name an incremental
        cluster integer per role.
        """
        items = roles if roles is not None else ()
        N = len(items)
        if N == 0:
            return cls(nodes=(), edges=())

        reg = registry if registry is not None else (GeneClusterRegistry() if clusters is None else None)
        db = roles_db or GeneRoles()

        nodes_list: list[GeneNode] = []
        for i, item in enumerate(items):
            desc = descriptions[i] if descriptions is not None and i < len(descriptions) else None
            cid = clusters[i] if clusters is not None else None
            node = GeneNode.from_role(
                item,
                cluster=cid,
                registry=reg,
                roles_db=db,
                description=desc,
                locus_type=locus_type,
            )
            nodes_list.append(node)

        nodes = tuple(nodes_list)
        if strands is not None:
            edges = cls.create_intergenic_edges(strands)
        else:
            edges = tuple(
                IntergenicEdge(source=i, target=i + 1)
                for i in range(N - 1)
            )

        # Extract metadata tuples: prioritize gene symbols for gene_names
        names_tuple = None
        if roles is not None:
            names_tuple = tuple(
                g.decode("utf-8", errors="replace") if isinstance(g, (bytes, bytearray)) else (
                    g.role_name.decode("utf-8", errors="replace") if isinstance(g, GeneNode) else str(g)
                )
                for g in roles
            )
        elif descriptions is not None:
            names_tuple = tuple(
                d.decode("utf-8", errors="replace") if isinstance(d, (bytes, bytearray)) else str(d)
                for d in descriptions
            )

        strands_tuple = None
        if strands is not None:
            raw_s = strands.strands if isinstance(strands, Intervals) else strands
            strands_tuple = tuple(int(x) for x in raw_s)

        lengths_tuple = tuple(int(length) for length in lengths) if lengths is not None else None

        return cls(
            nodes=nodes,
            edges=edges,
            gene_names=names_tuple,
            strands=strands_tuple,
            lengths=lengths_tuple,
            registry=reg,
        )

    @classmethod
    def from_loci(
        cls,
        primary_genes: Sequence[int | bytes | str | GeneNode],
        auxiliary_loci: Sequence[Sequence[int | bytes | str | GeneNode]] | None = None,
        *,
        primary_descriptions: Sequence[bytes | str] | None = None,
        auxiliary_descriptions: Sequence[Sequence[bytes | str] | None] | None = None,
        primary_strands: Intervals | Sequence[Any] | None = None,
        auxiliary_strands: Sequence[Intervals | Sequence[Any] | None] | None = None,
        primary_clusters: Sequence[int] | None = None,
        auxiliary_clusters: Sequence[Sequence[int] | None] | None = None,
        registry: GeneClusterRegistry | None = None,
        roles_db: GeneRoles | None = None,
    ) -> Self:
        """Constructs a multipartite GeneCluster with a primary locus and one or more auxiliary loci.

        Syntenic edges are constructed strictly WITHIN each physical locus, preventing spurious
        syntenic transitions across unlinked genomic segments.

        Nodes in primary_genes receive locus_type=0.
        Nodes in auxiliary_loci receive sequential locus_type=1, 2, ...
        """
        reg = registry if registry is not None else GeneClusterRegistry()
        db = roles_db or GeneRoles()

        all_nodes: list[GeneNode] = []
        all_edges: list[IntergenicEdge] = []

        # 1. Primary locus (locus_type = 0)
        c_prim = cls.from_genes(
            genes=primary_genes,
            descriptions=primary_descriptions,
            strands=primary_strands,
            clusters=primary_clusters,
            registry=reg,
            roles_db=db,
            locus_type=0,
        )
        all_nodes.extend(c_prim.nodes)
        all_edges.extend(c_prim.edges)

        # 2. Auxiliary loci (locus_type = 1, 2, ...)
        if auxiliary_loci:
            for k, aux_genes in enumerate(auxiliary_loci, start=1):
                aux_descs = (
                    auxiliary_descriptions[k - 1]
                    if auxiliary_descriptions is not None and k - 1 < len(auxiliary_descriptions)
                    else None
                )
                aux_str = (
                    auxiliary_strands[k - 1]
                    if auxiliary_strands is not None and k - 1 < len(auxiliary_strands)
                    else None
                )
                aux_clusts = (
                    auxiliary_clusters[k - 1]
                    if auxiliary_clusters is not None and k - 1 < len(auxiliary_clusters)
                    else None
                )

                node_offset = len(all_nodes)
                c_aux = cls.from_genes(
                    genes=aux_genes,
                    descriptions=aux_descs,
                    strands=aux_str,
                    clusters=aux_clusts,
                    registry=reg,
                    roles_db=db,
                    locus_type=k,
                )
                all_nodes.extend(c_aux.nodes)
                for edge in c_aux.edges:
                    all_edges.append(IntergenicEdge(
                        source=edge.source + node_offset,
                        target=edge.target + node_offset,
                        edge_type=edge.edge_type,
                    ))

        return cls(nodes=tuple(all_nodes), edges=tuple(all_edges), registry=reg)


@dataclass(frozen=True, slots=True)
class GeneClusters(BatchedContainer[GeneCluster, "GeneClusters"], GraphBatch):
    """A highly vectorized, Structure-of-Arrays (SoA) batch of GeneClusters."""
    node_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    node_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_sources: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_targets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]

    def __len__(self) -> int:
        """Returns the number of gene clusters in the batch."""
        return len(self.node_offsets) - 1

    @property
    def is_auxiliary(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Boolean mask indicating which nodes in node_types are auxiliary."""
        return GeneNode.LOCUS_TYPE.extract_array(self.node_types) > 0

    @classmethod
    def empty(cls) -> Self:
        """Creates an empty 0-length GeneClusters batch."""
        return cls(
            node_types=np.empty(0, dtype=np.uint32),
            node_offsets=np.zeros(1, dtype=np.uint32),
            edge_sources=np.empty(0, dtype=np.uint32),
            edge_targets=np.empty(0, dtype=np.uint32),
            edge_types=np.empty(0, dtype=np.uint32),
            edge_offsets=np.zeros(1, dtype=np.uint32),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenates multiple GeneClusters batches into a single contiguous batch."""
        batch_list: list[Self] = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty()
        if len(batch_list) == 1:
            return batch_list[0]

        nodes, n_offs, sources, targets, types, e_offs = cls._concat_soa_base(batch_list)
        return cls(
            node_types=nodes,
            node_offsets=n_offs,
            edge_sources=sources,
            edge_targets=targets,
            edge_types=types,
            edge_offsets=e_offs,
        )

    @classmethod
    def from_list(cls, clusters: Sequence[GeneCluster]) -> Self:
        """Compiles a sequence of individual GeneClusters into SoA batch format."""
        num_clusters = len(clusters)
        node_offsets = np.zeros(num_clusters + 1, dtype=np.uint32)
        edge_offsets = np.zeros(num_clusters + 1, dtype=np.uint32)

        all_nodes: list[int] = []
        all_sources: list[int] = []
        all_targets: list[int] = []
        all_types: list[int] = []

        node_offset = 0
        for i, c in enumerate(clusters):
            all_nodes.extend(int(n) for n in c.nodes)
            for e in c.edges:
                all_sources.append(e.source + node_offset)
                all_targets.append(e.target + node_offset)
                all_types.append(int(e.edge_type))
            node_offset += len(c.nodes)
            node_offsets[i + 1] = node_offset
            edge_offsets[i + 1] = len(all_types)

        return cls(
            node_types=np.array(all_nodes, dtype=np.uint32),
            node_offsets=node_offsets,
            edge_sources=np.array(all_sources, dtype=np.uint32),
            edge_targets=np.array(all_targets, dtype=np.uint32),
            edge_types=np.array(all_types, dtype=np.uint32),
            edge_offsets=edge_offsets,
        )

    @classmethod
    def from_genes_list(
        cls,
        genes_list: Sequence[Sequence[int | bytes | str | GeneNode]],
        descriptions_list: Sequence[Sequence[bytes | str] | None] | None = None,
        strands_list: Sequence[Intervals | Sequence[Any] | None] | None = None,
        registry: GeneClusterRegistry | None = None,
        roles_db: GeneRoles | None = None,
    ) -> tuple[Self, GeneClusterRegistry]:
        """Compiles multiple gene clusters into an SoA batch.

        Automatic gene role classification and unique allele cluster integer assignment.

        Returns:
            Tuple of (GeneClusters batch, GeneClusterRegistry mapping).
        """
        reg = registry if registry is not None else GeneClusterRegistry()
        db = roles_db or GeneRoles()
        clusters: list[GeneCluster] = []

        for i, genes in enumerate(genes_list):
            descs = descriptions_list[i] if descriptions_list is not None and i < len(descriptions_list) else None
            strands = strands_list[i] if strands_list is not None and i < len(strands_list) else None
            c = GeneCluster.from_genes(
                genes=genes,
                descriptions=descs,
                strands=strands,
                registry=reg,
                roles_db=db,
            )
            clusters.append(c)

        batch = cls.from_list(clusters)
        return batch, reg

    def __getitem__(
        self,
        item: int | np.integer | slice | np.ndarray | Sequence[int],
    ) -> GeneCluster | Self:
        """Extracts a single GeneCluster by integer index, or a sub-batch by slice / boolean mask."""
        if isinstance(item, int):
            idx = item
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("GeneClusters index out of range")

            m_start, m_end = self.node_offsets[idx], self.node_offsets[idx + 1]
            e_start, e_end = self.edge_offsets[idx], self.edge_offsets[idx + 1]

            nodes = tuple(GeneNode(x) for x in self.node_types[m_start:m_end])
            edges = tuple(
                IntergenicEdge(
                    source=int(self.edge_sources[i]) - int(m_start),
                    target=int(self.edge_targets[i]) - int(m_start),
                    edge_type=IntergenicType(self.edge_types[i])
                )
                for i in range(e_start, e_end)
            )
            return GeneCluster(nodes=nodes, edges=edges)

        if isinstance(item, slice):
            kept_indices = np.arange(len(self))[item]
        elif isinstance(item, np.ndarray) and item.dtype == bool:
            if len(item) != len(self):
                raise ValueError(f"Mask length {len(item)} does not match batch length {len(self)}.")
            kept_indices = np.where(item)[0]
        else:
            kept_indices = np.asarray(item, dtype=np.int64)

        (
            new_nodes,
            new_node_offsets,
            new_sources,
            new_targets,
            new_types,
            new_edge_offsets,
        ) = self._slice_soa_arrays(kept_indices)
        return self.__class__(
            node_types=new_nodes,
            node_offsets=new_node_offsets,
            edge_sources=new_sources,
            edge_targets=new_targets,
            edge_types=new_types,
            edge_offsets=new_edge_offsets,
        )
