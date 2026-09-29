"""
Object-oriented core alignment framework with pluggable cost strategies,
topological embedding smoothers, and interactive bipartite connection visualizers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array, diags

from kbank.core.graph import TAG_EDGE, GraphModel
from kbank.genes.roles import GeneRoles, TargetType
from kbank.glycan.glycan import Glycan, LinkageType
from kbank.glycan.sugars import Sugars
from kbank.loci.loci import GeneCluster, GeneNode


# Configuration Dataclass ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """
    Immutable configuration for bipartite core alignment solvers.
    """
    iterations: int = 2
    topo_weight: float = 1.0
    penalty_unmatched: float = 1e6
    directed: bool = True
    as_bipartite: bool = True
    symmetrize_embedding: bool = False

    @classmethod
    def default(cls) -> Self:
        return cls()

    @classmethod
    def rigid(cls) -> Self:
        """Strict alignment prioritizing rigid semantic constraints over topology."""
        return cls(iterations=1, topo_weight=0.2, penalty_unmatched=1e8)

    @classmethod
    def topological(cls) -> Self:
        """Relaxed alignment prioritizing topological core flow smoothing."""
        return cls(iterations=4, topo_weight=2.5, penalty_unmatched=1e5)


# Embedding Strategies -------------------------------------------------------------------------------------------------
class EmbeddingStrategy(ABC):
    """Abstract base strategy for computing continuous topological node embeddings."""
    @abstractmethod
    def compute(
        self,
        adj: csr_array[np.number, tuple[int, int]],
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        ...


@dataclass(frozen=True, slots=True)
class WeisfeilerLehmanEmbedding(EmbeddingStrategy):
    """
    Computes generalized continuous Weisfeiler-Lehman topological smoothing.
    """
    iterations: int = 2
    smoothing_factor: float = 0.5

    def compute(
        self,
        adj: csr_array[np.number, tuple[int, int]],
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Smooths continuous features across sparse core adjacency.

        Args:
            adj: Sparse CSR adjacency matrix of shape (N, N)
                (:class:`scipy.sparse.csr_array`).
            features: 2D NumPy array of initial node features of shape (N, D)
                (`np.ndarray[tuple[int, int], np.dtype[np.float64]]`).

        Returns:
            Concatenated multi-scale continuous embeddings of shape (N, D * (iterations + 1))
            (`np.ndarray[tuple[int, int], np.dtype[np.float64]]`).
        """
        N = adj.shape[0]
        if N == 0:
            return features

        deg = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
        deg[deg == 0] = 1.0  # avoid division by zero
        deg_inv = diags(1.0 / deg)
        adj_norm = deg_inv @ adj

        embeddings = [features]
        current_h = features
        alpha = self.smoothing_factor
        for _ in range(self.iterations):
            current_h = (1.0 - alpha) * current_h + alpha * (adj_norm @ current_h)
            embeddings.append(current_h)

        return np.concatenate(embeddings, axis=1)


@dataclass(frozen=True, slots=True)
class DegreeProfileEmbedding(EmbeddingStrategy):
    """
    Computes in-degree, out-degree, and total degree profiles for each node.
    """
    def compute(
        self,
        adj: csr_array[np.number, tuple[int, int]],
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Augments node features with in-degree and out-degree computed from sparse adjacency.

        Args:
            adj: Sparse CSR adjacency matrix of shape (N, N)
                (:class:`scipy.sparse.csr_array`).
            features: 2D NumPy array of initial node features of shape (N, D)
                (`np.ndarray[tuple[int, int], np.dtype[np.float64]]`).

        Returns:
            2D NumPy array with degree profiles appended of shape (N, D + 2)
            (`np.ndarray[tuple[int, int], np.dtype[np.float64]]`).
        """
        N = adj.shape[0]
        if N == 0:
            return features
        out_deg = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()[:, None]
        in_deg = np.asarray(adj.sum(axis=0), dtype=np.float64).ravel()[:, None]
        return np.column_stack([features, out_deg, in_deg])


# ----------------------------------------------------------------------------------------------------------------------
# Biochemical Rule Engine ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BiochemicalRule:
    """
    Formal mapping specification between a gene role index and compatible glycan structures.
    """
    role: int
    matches_linkages: bool = False
    target_type: TargetType = TargetType.ANY
    target_modifiers: tuple[bytes, ...] = ()
    allowed_names: frozenset[bytes] = field(default_factory=frozenset)
    allowed_classes: frozenset[bytes] = field(default_factory=frozenset)
    disallowed_classes: frozenset[bytes] = field(default_factory=frozenset)
    base_cost: float = 0.0
    description: bytes = b""

    @property
    def role_name(self) -> bytes:
        if 0 <= self.role < len(GeneRoles):
            return GeneRoles.names[self.role]
        return b""

    def is_compatible(
        self,
        label_b: int,
        attached_modifiers: tuple[bytes, ...] = (),
    ) -> bool:
        """Evaluates whether a glycan ResidueType or LinkageType label is compatible with this rule."""
        if self.target_type == TargetType.NONE:
            return False

        is_edge = bool(label_b & TAG_EDGE)

        if self.target_type == TargetType.LINKAGE:
            # Must match a linkage (edge node)
            return is_edge

        if is_edge:
            # Non-linkage targets cannot match an edge node
            return False

        # Primary node in glycan: ResidueType
        sugar_idx = label_b & 0xFF
        if sugar_idx >= len(Sugars.names):
            return False

        is_mod = Sugars.classes[sugar_idx] == b"Mod"
        is_reducing = bool(label_b & (1 << 11))

        if self.target_type == TargetType.REDUCING_END:
            # Initiator must match reducing-end sugar
            return is_reducing and not is_mod

        if self.target_type == TargetType.MODIFIED_MONO:
            # Must match a core sugar harboring the required modifier
            if is_mod:
                return False
            if self.target_modifiers:
                return any(mod in attached_modifiers for mod in self.target_modifiers)
            return len(attached_modifiers) > 0

        if self.target_type == TargetType.MONOSACCHARIDE:
            # Precursor synthase / epimerase: must match a core sugar (not a modifier)
            if is_mod:
                return False

        # General sugar name/class constraints
        sugar_name = Sugars.names[sugar_idx]
        sugar_class = Sugars.classes[sugar_idx]

        if sugar_class in self.disallowed_classes:
            return False

        if sugar_name in self.allowed_names:
            return True

        if sugar_class in self.allowed_classes:
            return True

        if not self.allowed_names and not self.allowed_classes and self.disallowed_classes:
            return True

        if self.target_type == TargetType.ANY:
            return True

        return not self.allowed_names and not self.allowed_classes


@dataclass(frozen=True, slots=True)
class BiochemicalRuleRegistry:
    """
    Immutable registry of biochemical rules mapping gene role indices to glycan elements.
    Provides inspection, custom rule overrides, and fast matrix evaluation.
    """
    rules: dict[int, BiochemicalRule]

    @classmethod
    def default(cls) -> Self:
        rules = {}
        for i in range(len(GeneRoles)):
            rules[i] = BiochemicalRule(
                role=i,
                matches_linkages=bool(GeneRoles.matches_linkages[i]),
                target_type=GeneRoles.target_types[i],
                target_modifiers=GeneRoles.target_modifiers[i],
                allowed_names=frozenset(GeneRoles.allowed_names[i]),
                allowed_classes=frozenset(GeneRoles.allowed_classes[i]),
                disallowed_classes=frozenset(GeneRoles.disallowed_classes[i]),
                description=GeneRoles.descriptions[i],
            )
        return cls(rules=rules)

    def get_rule(self, role: int | bytes) -> BiochemicalRule:
        if isinstance(role, (bytes, bytearray)):
            idx = GeneRoles.name_to_index.get(bytes(role).upper())
            if idx is None:
                return BiochemicalRule(
                    role=-1,
                    matches_linkages=False,
                    target_type=TargetType.ANY,
                    target_modifiers=(),
                    allowed_names=frozenset(),
                    allowed_classes=frozenset(),
                    disallowed_classes=frozenset(),
                    description=b"Unregistered or unknown gene role."
                )
            return self.rules[idx]
        return self.rules.get(role, BiochemicalRule(
            role=role,
            matches_linkages=False,
            target_type=TargetType.ANY,
            target_modifiers=(),
            allowed_names=frozenset(),
            allowed_classes=frozenset(),
            disallowed_classes=frozenset(),
            description=b"Unregistered or unknown gene role index."
        ))


# Semantic Cost Strategies ---------------------------------------------------------------------------------------------
class CostStrategy(ABC):
    """Abstract base strategy for computing semantic node compatibility cost matrices."""
    @abstractmethod
    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        ...


@dataclass(frozen=True, slots=True)
class GeneGlycanCostStrategy(CostStrategy):
    """
    Biochemical rule engine mapping Gene roles (from BGCs) to ResidueTypes & Linkages (from Glycans).
    Uses a formal BiochemicalRuleRegistry to determine compatibility.
    """
    rules: BiochemicalRuleRegistry = field(default_factory=BiochemicalRuleRegistry.default)

    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        N, M = len(labels_a), len(labels_b)
        C = np.full((N, M), np.inf, dtype=np.float64)

        num_nodes = len(getattr(graph_b, "nodes", ()))
        nodes = getattr(graph_b, "nodes", ())
        edges = getattr(graph_b, "edges", ())
        attached_getter = getattr(graph_b, "attached_modifiers_by_node", None) if graph_b is not None else None
        attached_mods = attached_getter() if callable(attached_getter) else {}

        for i in range(N):
            val_a = int(labels_a[i])
            is_edge_a = bool(val_a & TAG_EDGE)

            if is_edge_a:
                # Intergenic edges in BGC are syntenic spacers, not biochemical targets
                continue

            role_idx = val_a & 0xFF  # Bit 0-7: GeneRoles index
            rule = self.rules.get_rule(role_idx)
            role_name = rule.role_name
            if rule.target_type == TargetType.NONE:
                continue

            for j in range(M):
                val_b = int(labels_b[j])
                is_edge_b = bool(val_b & TAG_EDGE)
                mods = attached_mods.get(j, ())

                if rule.target_type == TargetType.LINKAGE:
                    if is_edge_b and edges:
                        edge_idx = j - num_nodes
                        if 0 <= edge_idx < len(edges):
                            edge = edges[edge_idx]
                            src_node = nodes[edge.source]
                            tgt_node = nodes[edge.target]
                            is_mod_edge = getattr(src_node, "is_modifier", False) or getattr(tgt_node, "is_modifier", False)
                            if is_mod_edge:
                                continue
                            is_reducing = getattr(src_node, "is_reducing_end", False)
                            if role_name == b"POLYMERASE":
                                C[i, j] = 0.0 if is_reducing else 2.5
                            elif role_name == b"GLYCOSYLTRANSFERASE":
                                C[i, j] = 0.0 if not is_reducing else 2.0
                            else:
                                C[i, j] = rule.base_cost
                    elif is_edge_b:
                        C[i, j] = rule.base_cost

                elif rule.target_type == TargetType.REDUCING_END:
                    if not is_edge_b and j < num_nodes:
                        is_red = getattr(nodes[j], "is_reducing_end", False)
                        is_mod = getattr(nodes[j], "is_modifier", False)
                        if is_red and not is_mod:
                            C[i, j] = 0.0

                elif rule.target_type == TargetType.MODIFIED_MONO:
                    if not is_edge_b and j < num_nodes:
                        if getattr(nodes[j], "is_modifier", False):
                            sname = Sugars.names[nodes[j].sugar]
                            if rule.target_modifiers and any(mod in sname or sname.startswith(mod) for mod in rule.target_modifiers):
                                C[i, j] = 0.0
                            elif not rule.target_modifiers:
                                C[i, j] = 0.0
                        elif rule.is_compatible(val_b, attached_modifiers=mods):
                            C[i, j] = rule.base_cost

                elif rule.target_type == TargetType.MONOSACCHARIDE:
                    if not is_edge_b and j < num_nodes:
                        if rule.is_compatible(val_b, attached_modifiers=mods) and (rule.allowed_names or rule.allowed_classes):
                            C[i, j] = rule.base_cost

                elif rule.target_type != TargetType.NONE:
                    if rule.is_compatible(val_b, attached_modifiers=mods):
                        C[i, j] = rule.base_cost

        return C


@dataclass(frozen=True, slots=True)
class ExactMatchCostStrategy(CostStrategy):
    """
    Isomorphic cost model: 0.0 for identical node types, infinite penalty for mismatches.
    """
    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        N, M = len(labels_a), len(labels_b)
        C = np.full((N, M), np.inf, dtype=np.float64)
        for i in range(N):
            for j in range(M):
                if labels_a[i] == labels_b[j]:
                    C[i, j] = 0.0
        return C


@dataclass(frozen=True, slots=True)
class CallableCostStrategy(CostStrategy):
    """
    Adapter wrapping custom user-defined cost functions.
    """
    fn: Callable[..., np.ndarray[tuple[int, int], np.dtype[np.float64]]]

    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        return np.asarray(self.fn(labels_a, labels_b), dtype=np.float64)



# Rich Alignment Result Object -----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Alignment:
    """
    Rich outcome of bipartite core alignment containing matched node pairs,
    individual costs, total alignment cost, and unmatched nodes.
    """
    matches: dict[int, int]
    costs: dict[int, float]
    total_cost: float
    graph_a: GraphModel
    graph_b: GraphModel
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]

    def to_dict(self) -> dict[int, tuple[int, float]]:
        """Returns {node_a: (node_b, cost)} dictionary."""
        return {a: (b, self.costs[a]) for a, b in self.matches.items()}

    def __iter__(self):
        """Allows unpacking: mapping, cost = align_graphs(...)"""
        yield self.to_dict()
        yield self.total_cost


# Alignment Solver Classes ---------------------------------------------------------------------------------------------
class GraphAligner:
    """
    Configurable, object-oriented bipartite core alignment solver.
    """
    def __init__(
        self,
        cost_strategy: CostStrategy | None = None,
        embedding_strategy: EmbeddingStrategy | None = None,
        config: AlignmentConfig | None = None,
    ):
        self.cost_strategy = cost_strategy or GeneGlycanCostStrategy()
        self.embedding_strategy = embedding_strategy or WeisfeilerLehmanEmbedding()
        self.config = config or AlignmentConfig.default()

    def align(self, graph_a: GraphModel, graph_b: GraphModel) -> Alignment:
        """
        Executes bipartite core alignment between two GraphModels.

        Constructs a bipartite cost matrix combining semantic domain compatibility
        and topological embedding distance, then computes the minimum weight bipartite
        matching using the Hungarian algorithm via :func:`scipy.optimize.linear_sum_assignment`.

        Args:
            graph_a: First core model to align (e.g. GeneCluster).
            graph_b: Second core model to align (e.g. Glycan).

        Returns:
            An :class:`Alignment` containing matched node index pairs,
            individual match costs, total alignment cost, and unmatched nodes.
        """
        is_sym = self.config.symmetrize_embedding or not self.config.directed
        adj_a, labels_a = graph_a.as_graph(as_bipartite=self.config.as_bipartite, symmetrize=is_sym)
        adj_b, labels_b = graph_b.as_graph(as_bipartite=self.config.as_bipartite, symmetrize=is_sym)

        N, M = len(labels_a), len(labels_b)
        if N == 0 or M == 0:
            return Alignment(
                matches={},
                costs={},
                total_cost=0.0,
                graph_a=graph_a,
                graph_b=graph_b,
                unmatched_a=tuple(range(N)),
                unmatched_b=tuple(range(M))
            )

        # 1. Base Semantic Compatibility Cost Matrix
        C_base = self.cost_strategy.compute_cost(
            labels_a,
            labels_b,
            penalty=self.config.penalty_unmatched,
            graph_a=graph_a,
            graph_b=graph_b,
        )

        # 2. Compute Topological Embeddings
        deg_a = np.asarray(adj_a.sum(axis=1), dtype=np.float64).ravel()[:, None]
        deg_b = np.asarray(adj_b.sum(axis=1), dtype=np.float64).ravel()[:, None]

        emb_a = self.embedding_strategy.compute(adj_a, deg_a)
        emb_b = self.embedding_strategy.compute(adj_b, deg_b)

        # 3. Topological Euclidean Distance Matrix
        C_topo = np.linalg.norm(emb_a[:, None, :] - emb_b[None, :, :], axis=2) * self.config.topo_weight

        # 4. Total Cost & Safe Penalty Substitution
        C_total = C_base + C_topo
        C_safe = np.where(np.isinf(C_total), self.config.penalty_unmatched, C_total)

        # 5. Optimal Bipartite Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(C_safe)

        matches: dict[int, int] = {}
        costs: dict[int, float] = {}
        total_cost = 0.0

        for r, c in zip(row_ind, col_ind):
            cost_val = float(C_total[r, c])
            if cost_val < self.config.penalty_unmatched and not np.isinf(cost_val):
                matches[int(r)] = int(c)
                costs[int(r)] = cost_val
                total_cost += cost_val

        unmatched_a = tuple(i for i in range(N) if i not in matches)
        unmatched_b = tuple(j for j in range(M) if j not in matches.values())

        return Alignment(
            matches=matches,
            costs=costs,
            total_cost=total_cost,
            graph_a=graph_a,
            graph_b=graph_b,
            unmatched_a=unmatched_a,
            unmatched_b=unmatched_b
        )


class GeneGlycanAligner(GraphAligner):
    """Specialized aligner pre-configured for BGC-to-Glycan biosynthetic pathway matching."""
    def __init__(
        self,
        rules: BiochemicalRuleRegistry | None = None,
        config: AlignmentConfig | None = None,
    ):
        super().__init__(
            cost_strategy=GeneGlycanCostStrategy(rules=rules or BiochemicalRuleRegistry.default()),
            embedding_strategy=WeisfeilerLehmanEmbedding(),
            config=config or AlignmentConfig(as_bipartite=True, directed=False, symmetrize_embedding=True)
        )


class IsomorphicGraphAligner(GraphAligner):
    """Specialized aligner pre-configured for same-domain structural isomorphism matching."""
    def __init__(self, config: AlignmentConfig | None = None):
        super().__init__(
            cost_strategy=ExactMatchCostStrategy(),
            embedding_strategy=WeisfeilerLehmanEmbedding(iterations=3),
            config=config or AlignmentConfig(as_bipartite=False, directed=True)
        )


# ----------------------------------------------------------------------------------------------------------------------
# Top-Level Facade Functions
# ----------------------------------------------------------------------------------------------------------------------

def align_graphs(
    graph_a: GraphModel,
    graph_b: GraphModel,
    cost_fn: Callable[..., np.ndarray[tuple[int, int], np.dtype[np.float64]]] | CostStrategy | None = None,
    config: AlignmentConfig | None = None,
    **kwargs: Any,
) -> Alignment:
    """
    Aligns two GraphModels using optimal bipartite core matching.
    Returns an Alignment result which can be unpacked as (matches_dict, total_cost)
    or used directly as a rich object with .matches, .costs.
    """
    if cost_fn is None:
        cost_strategy = GeneGlycanCostStrategy()
    elif isinstance(cost_fn, CostStrategy):
        cost_strategy = cost_fn
    elif callable(cost_fn):
        cost_strategy = CallableCostStrategy(cost_fn)
    else:
        raise TypeError(f"Invalid cost_fn type: {type(cost_fn)}")

    aligner = GraphAligner(cost_strategy=cost_strategy, config=config)
    return aligner.align(graph_a, graph_b)


def infer_gt_linkage_assignments(
    bgc: GeneCluster,
    glycan: Glycan,
    aligner: GraphAligner | None = None,
) -> dict[GeneNode, LinkageType]:
    """
    Infers specific glycosyltransferase (and polymerase) enzyme-to-linkage catalytic assignments
    via optimal bipartite topological core alignment.

    Args:
        bgc: Biosynthetic gene cluster containing GeneNodes (roles + clusters).
        glycan: Target glycan structure containing residues and linkages.
        aligner: Optional GraphAligner instance. If None, uses GeneGlycanAligner.

    Returns:
        Dictionary mapping each matched GeneNode (with role and cluster) to its assigned LinkageType.
    """
    if aligner is None:
        aligner = GeneGlycanAligner()

    alignment = aligner.align(bgc, glycan)
    num_glycan_nodes = len(glycan.nodes)

    assignments: dict[GeneNode, LinkageType] = {}
    for i_a, i_b in alignment.matches.items():
        if i_a < len(bgc.nodes):
            gene = bgc.nodes[i_a]
            if gene.target_type == TargetType.LINKAGE:
                link_edge_idx = i_b - num_glycan_nodes
                if 0 <= link_edge_idx < len(glycan.edges):
                    assignments[gene] = glycan.edges[link_edge_idx].edge_type
    return assignments
