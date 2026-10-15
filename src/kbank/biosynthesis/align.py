"""Cross-domain geno-glycomic alignment between gene clusters and glycan graphs.

Provides `TargetType`, `BiochemicalRule`, `BiochemicalRuleRegistry`, `GeneGlycanCostStrategy`,
`GeneGlycanAligner`, and `infer_gt_linkage_assignments` for mapping biosynthetic gene clusters
(`GeneCluster` / `GeneClusters` from `serogen`) onto bipartite carbohydrate structures
(`Glycan` / `Glycans` from `kbank`).
"""
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from serogen.core.graph import (
    TAG_EDGE,
    Alignment,
    AlignmentConfig,
    CostStrategy,
    EmbeddingStrategy,
    GraphAligner,
    GraphModel,
    WeisfeilerLehmanEmbedding,
)
from serogen.roles import GeneRole
from serogen.loci import GeneCluster, GeneNode

from kbank.glycan import Glycan, LinkageType, ResidueType
from kbank.sugars import Sugars


class TargetType(StrEnum):
    """Structural target category in a bipartite glycan graph for a biosynthetic gene role."""

    NONE = "NONE"
    REDUCING_END = "REDUCING_END"
    LINKAGE = "LINKAGE"
    MODIFIED_MONO = "MODIFIED_MONO"
    MONOSACCHARIDE = "MONOSACCHARIDE"
    ANY = "ANY"


@dataclass(slots=True, frozen=True)
class BiosyntheticRule:
    """Declarative rule defining what structural elements a `GeneRole` can synthesize."""

    role_idx: int
    role_name: str
    target_type: TargetType
    compatible_sugars: frozenset[int] = frozenset()
    target_modifiers: frozenset[int] = frozenset()
    matches_linkages: bool = False
    is_polymerase: bool = False


@dataclass(slots=True, frozen=True)
class _RoleGlycanSpec:
    """Static glycan compatibility specification for a `GeneRole`."""

    target_type: TargetType = TargetType.NONE
    matches_linkages: bool = False
    target_modifiers: tuple[bytes, ...] = ()
    allowed_names: tuple[bytes, ...] = ()
    allowed_classes: tuple[bytes, ...] = ()
    disallowed_classes: tuple[bytes, ...] = ()


_ROLE_GLYCAN_SPECS: dict[GeneRole, _RoleGlycanSpec] = {
    GeneRole.FLIPPASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.POLYMERASE: _RoleGlycanSpec(
        TargetType.LINKAGE,
        matches_linkages=True,
    ),
    GeneRole.COPOLYMERASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.INITIATOR: _RoleGlycanSpec(
        TargetType.REDUCING_END,
        disallowed_classes=(b"Mod",),
    ),
    GeneRole.KINASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.PHOSPHATASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.ABC_PERMEASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.ABC_ATPASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.TRANSMEMBRANE_SYNTHASE: _RoleGlycanSpec(
        TargetType.LINKAGE,
        matches_linkages=True,
        disallowed_classes=(b"Mod",),
    ),
    GeneRole.LIGASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.TRANSLOCON: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.CAPSULE_ANCHOR: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.GLYCOSYLTRANSFERASE: _RoleGlycanSpec(
        TargetType.LINKAGE,
        matches_linkages=True,
        disallowed_classes=(b"Mod",),
    ),
    GeneRole.PYRUVYLTRANSFERASE: _RoleGlycanSpec(
        TargetType.MODIFIED_MONO,
        target_modifiers=(b"Pyr",),
        allowed_names=(b"Pyr",),
    ),
    GeneRole.METHYLTRANSFERASE: _RoleGlycanSpec(
        TargetType.MODIFIED_MONO,
        target_modifiers=(b"OMe",),
        allowed_names=(b"OMe",),
    ),
    GeneRole.ACETYLTRANSFERASE: _RoleGlycanSpec(
        TargetType.MODIFIED_MONO,
        target_modifiers=(
            b"OAc",
            b"3HOBut",
            b"3HOMyr",
            b"3HOLau",
            b"3HODco",
            b"3HOSte",
            b"3HOCap",
            b"Lau",
            b"Myr",
            b"Ste",
            b"Lipid",
        ),
        allowed_names=(
            b"OAc",
            b"3HOBut",
            b"3HOMyr",
            b"3HOLau",
            b"3HODco",
            b"3HOSte",
            b"3HOCap",
            b"Lau",
            b"Myr",
            b"Ste",
            b"Lipid",
        ),
    ),
    GeneRole.PHOSPHOTRANSFERASE: _RoleGlycanSpec(
        TargetType.MODIFIED_MONO,
        target_modifiers=(b"P", b"PEtn", b"ChoP", b"GroP"),
        allowed_names=(b"P", b"PEtn", b"ChoP", b"GroP"),
    ),
    GeneRole.FORMYLTRANSFERASE: _RoleGlycanSpec(
        TargetType.MODIFIED_MONO,
        target_modifiers=(b"Fo",),
        allowed_names=(b"Fo",),
    ),
    GeneRole.ALANINE_TRANSFERASE: _RoleGlycanSpec(
        TargetType.MODIFIED_MONO,
        target_modifiers=(b"DAla", b"LAla", b"Ala"),
        allowed_names=(b"DAla", b"LAla", b"Ala"),
        allowed_classes=(b"HexA", b"HexNA"),
    ),
    GeneRole.DEACETYLASE: _RoleGlycanSpec(
        TargetType.MODIFIED_MONO,
        target_modifiers=(
            b"GlcN",
            b"GalN",
            b"QuiN",
            b"QuiN4N",
            b"FucN",
            b"FucN4N",
            b"RhaN",
            b"ManN",
            b"GalNA",
            b"GlcNA",
            b"ManNA",
            b"GlcN3NA",
        ),
        allowed_names=(
            b"GlcN",
            b"GalN",
            b"QuiN",
            b"QuiN4N",
            b"FucN",
            b"FucN4N",
            b"RhaN",
            b"ManN",
            b"GalNA",
            b"GlcNA",
            b"ManNA",
            b"GlcN3NA",
        ),
    ),
    GeneRole.GLYCOSYLHYDROLASE: _RoleGlycanSpec(TargetType.MODIFIED_MONO),
    GeneRole.AMINOTRANSFERASE: _RoleGlycanSpec(
        TargetType.MONOSACCHARIDE,
        allowed_names=(
            b"GlcN",
            b"GalN",
            b"QuiN",
            b"QuiN4N",
            b"FucN",
            b"FucN4N",
            b"Pse",
            b"Leg",
            b"Bac",
            b"Aci",
            b"RhaN",
            b"RhaN3N",
        ),
        allowed_classes=(b"HexN", b"dHexN"),
    ),
    GeneRole.ALDOLASE: _RoleGlycanSpec(
        TargetType.MONOSACCHARIDE,
        allowed_names=(b"Neu5Ac", b"Pse", b"Leg", b"Aci", b"Kdo"),
        allowed_classes=(b"HigherSugar", b"gPse", b"gLeg", b"g4eLeg", b"gAci", b"gNeu", b"NeuAc"),
    ),
    GeneRole.EPIMERASE: _RoleGlycanSpec(
        TargetType.MONOSACCHARIDE,
        allowed_classes=(b"Hex", b"HexNAc", b"DeoxyHex", b"DeoxyHexNAc", b"dHex", b"dHexNAc"),
        disallowed_classes=(b"Mod",),
    ),
    GeneRole.OXIDOREDUCTASE: _RoleGlycanSpec(
        TargetType.MONOSACCHARIDE,
        allowed_names=(
            b"Fuc",
            b"Rha",
            b"Qui",
            b"6dGul",
            b"6dAlt",
            b"6dTal",
            b"Oli",
            b"Tyv",
            b"Abe",
            b"Par",
            b"Dig",
            b"Col",
            b"ol",
            b"Gro",
            b"RhaN",
            b"RhaN3N",
        ),
        allowed_classes=(b"DeoxyHex", b"DeoxyHexNAc", b"DiDeoxyHex", b"dHex", b"dHexNAc", b"ddHex", b"dHexN"),
    ),
    GeneRole.NUCLEOTIDYLTRANSFERASE: _RoleGlycanSpec(
        TargetType.MONOSACCHARIDE,
        disallowed_classes=(b"Mod",),
    ),
    GeneRole.NUCLEOTIDASE: _RoleGlycanSpec(
        TargetType.MONOSACCHARIDE,
        allowed_names=(b"Pse", b"Leg", b"Aci", b"Neu5Ac", b"Man"),
        allowed_classes=(b"HigherSugar", b"gPse", b"gLeg", b"g4eLeg", b"gAci", b"gNeu", b"NeuAc"),
    ),
    GeneRole.TRANSPOSASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.INTEGRASE: _RoleGlycanSpec(TargetType.NONE),
    GeneRole.UNKNOWN: _RoleGlycanSpec(
        TargetType.ANY,
        allowed_classes=(
            b"Hex",
            b"HexNAc",
            b"DeoxyHex",
            b"DeoxyHexNAc",
            b"dHex",
            b"dHexNAc",
            b"Pentose",
            b"Pen",
            b"DiDeoxyHex",
            b"ddHex",
            b"Mod",
            b"HigherSugar",
            b"Generic",
        ),
    ),
}


class BiosyntheticRuleRegistry:
    """Builds and caches biochemical compatibility rules from `GeneRole` and `Sugars`."""

    def __init__(self, sugars_db: type[Sugars] | Sugars = Sugars) -> None:
        """Initialize the registry and precompute rules for all `GeneRole` members."""
        self.sugars_db = sugars_db
        self.rules: dict[int, BiosyntheticRule] = {}
        self._build_rules()

    def _build_rules(self) -> None:
        names = self.sugars_db.names
        classes = self.sugars_db.classes
        name_to_idx = self.sugars_db.name_to_index

        for role in GeneRole:
            spec = _ROLE_GLYCAN_SPECS[role]
            mod_indices = frozenset(name_to_idx[m] for m in spec.target_modifiers if m in name_to_idx)

            compat_indices: set[int] = set()
            for s_idx in range(len(names)):
                s_name = names[s_idx]
                s_cls = classes[s_idx]

                if spec.disallowed_classes and s_cls in spec.disallowed_classes:
                    continue

                if spec.allowed_names or spec.allowed_classes:
                    if s_name in spec.allowed_names or s_cls in spec.allowed_classes:
                        compat_indices.add(s_idx)
                elif spec.disallowed_classes:
                    compat_indices.add(s_idx)

            self.rules[int(role)] = BiosyntheticRule(
                role_idx=int(role),
                role_name=role.name,
                target_type=spec.target_type,
                compatible_sugars=frozenset(compat_indices),
                target_modifiers=mod_indices,
                matches_linkages=spec.matches_linkages,
                is_polymerase=(role is GeneRole.POLYMERASE),
            )

    def get_rule(self, role_idx: int | GeneRole) -> BiosyntheticRule:
        """Retrieve the biochemical rule for a given `GeneRole` index."""
        return self.rules.get(int(role_idx), self.rules[int(GeneRole.UNKNOWN)])


class BiosyntheticCostStrategy(CostStrategy):
    """Domain-specific cost strategy mapping `GeneCluster` nodes to `Glycan` bipartite nodes."""

    def __init__(
        self,
        rule_registry: BiosyntheticRuleRegistry | None = None,
    ) -> None:
        """Initialize `GeneGlycanCostStrategy` with a `BiochemicalRuleRegistry`."""
        self.registry = rule_registry or BiosyntheticRuleRegistry()

    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Compute `(N_gene, N_glycan)` pairwise biochemical compatibility cost matrix."""
        is_swapped = isinstance(graph_a, Glycan) and isinstance(graph_b, GeneCluster)
        if is_swapped:
            gene_labels, glycan_labels = labels_b, labels_a
            glycan_graph = graph_a
        else:
            gene_labels, glycan_labels = labels_a, labels_b
            glycan_graph = graph_b if isinstance(graph_b, Glycan) else None

        n_genes = len(gene_labels)
        n_glycan = len(glycan_labels)
        cost = np.full((n_genes, n_glycan), np.inf, dtype=np.float64)

        parent_to_mods: dict[int, set[int]] = {j: set() for j in range(n_glycan)}
        reducing_end_indices: set[int] = set()
        linkage_to_donor_sugar: dict[int, int] = {}

        if glycan_graph is not None:
            n_res = len(glycan_graph.nodes)
            for j, res in enumerate(glycan_graph.nodes):
                if res.is_reducing_end:
                    reducing_end_indices.add(j)
            if not reducing_end_indices and n_res > 0:
                reducing_end_indices.add(0)

            for k, edge in enumerate(glycan_graph.edges):
                src, tgt = edge.source, edge.target
                s_res = glycan_graph.nodes[src]
                t_res = glycan_graph.nodes[tgt]
                if s_res.is_modifier and not t_res.is_modifier:
                    parent_to_mods[tgt].add(s_res.sugar)
                elif t_res.is_modifier and not s_res.is_modifier:
                    parent_to_mods[src].add(t_res.sugar)
                bipartite_edge_idx = n_res + k
                if bipartite_edge_idx < n_glycan:
                    linkage_to_donor_sugar[bipartite_edge_idx] = s_res.sugar
        else:
            for j, raw_g in enumerate(glycan_labels):
                val = int(raw_g)
                if not (val & TAG_EDGE):
                    res = ResidueType(val)
                    if res.is_reducing_end:
                        reducing_end_indices.add(j)
            if not reducing_end_indices and n_glycan > 0 and not (int(glycan_labels[0]) & TAG_EDGE):
                reducing_end_indices.add(0)

        for i, raw_gene in enumerate(gene_labels):
            g_val = int(raw_gene)
            if g_val & TAG_EDGE:
                for j, raw_gly in enumerate(glycan_labels):
                    if int(raw_gly) & TAG_EDGE:
                        cost[i, j] = 0.2
                continue

            gene = GeneNode(g_val)
            rule = self.registry.get_rule(gene.role_idx)

            if rule.target_type == TargetType.NONE:
                continue

            for j, raw_gly in enumerate(glycan_labels):
                gly_val = int(raw_gly)
                is_link = bool(gly_val & TAG_EDGE)

                if is_link:
                    if rule.matches_linkages:
                        if rule.is_polymerase:
                            cost[i, j] = 0.05
                        else:
                            donor_s = linkage_to_donor_sugar.get(j, -1)
                            if donor_s in rule.compatible_sugars or not rule.compatible_sugars:
                                cost[i, j] = 0.10
                            else:
                                cost[i, j] = 0.45
                    elif rule.target_type == TargetType.ANY:
                        cost[i, j] = 0.60
                else:
                    res = ResidueType(gly_val)
                    sugar_idx = res.sugar
                    is_red = j in reducing_end_indices
                    attached_mods = parent_to_mods.get(j, set())

                    if rule.target_type == TargetType.REDUCING_END:
                        if is_red and sugar_idx in rule.compatible_sugars:
                            cost[i, j] = 0.02
                        elif not res.is_modifier and sugar_idx in rule.compatible_sugars:
                            cost[i, j] = 0.40

                    elif rule.target_type == TargetType.MODIFIED_MONO:
                        if res.is_modifier and sugar_idx in rule.target_modifiers:
                            cost[i, j] = 0.05
                        elif not res.is_modifier and (attached_mods & rule.target_modifiers):
                            cost[i, j] = 0.12
                        elif sugar_idx in rule.compatible_sugars:
                            cost[i, j] = 0.15

                    elif rule.target_type == TargetType.MONOSACCHARIDE:
                        if sugar_idx in rule.compatible_sugars:
                            cost[i, j] = 0.10
                        elif not res.is_modifier:
                            cost[i, j] = 0.65

                    elif rule.target_type == TargetType.LINKAGE:
                        if not res.is_modifier and not is_red and sugar_idx in rule.compatible_sugars:
                            cost[i, j] = 0.25

                    elif rule.target_type == TargetType.ANY:
                        cost[i, j] = 0.50

        return cost.T if is_swapped else cost


class BiosyntheticAligner(GraphAligner):
    """Specialized `GraphAligner` pre-configured for matching `GeneCluster` to `Glycan`."""

    def __init__(
        self,
        config: AlignmentConfig | None = None,
        embedding_strategy: EmbeddingStrategy | None = None,
        sugars_db: type[Sugars] | Sugars = Sugars,
    ) -> None:
        """Initialize `GeneGlycanAligner`."""
        cfg = config or AlignmentConfig(iterations=2, topo_weight=0.5, as_bipartite=True)
        registry = BiosyntheticRuleRegistry(sugars_db=sugars_db)
        super().__init__(
            cost_strategy=BiosyntheticCostStrategy(rule_registry=registry),
            embedding_strategy=embedding_strategy or WeisfeilerLehmanEmbedding(iterations=cfg.iterations),
            config=cfg,
        )


def infer_gt_linkage_assignments(
    cluster: GeneCluster,
    glycan: Glycan,
    alignment: Alignment | None = None,
) -> dict[int, tuple[int, LinkageType]]:
    """Explicit helper mapping each glycosyltransferase in a `GeneCluster` to its `Glycan` linkage."""
    if alignment is None:
        aligner = BiosyntheticAligner()
        alignment = aligner.align(cluster, glycan)

    _, cluster_labels = cluster.as_graph(as_bipartite=True)
    _, glycan_labels = glycan.as_graph(as_bipartite=True)

    assignments: dict[int, tuple[int, LinkageType]] = {}
    for g_idx, gly_idx in alignment.matches.items():
        if g_idx >= len(cluster_labels) or gly_idx >= len(glycan_labels):
            continue
        g_val = int(cluster_labels[g_idx])
        gly_val = int(glycan_labels[gly_idx])

        if not (g_val & TAG_EDGE):
            g_node = GeneNode(g_val)
            role = g_node.role
            is_gt_like = role in (
                GeneRole.GLYCOSYLTRANSFERASE,
                GeneRole.POLYMERASE,
                GeneRole.TRANSMEMBRANE_SYNTHASE,
            )
            if is_gt_like and (gly_val & TAG_EDGE):
                assignments[int(g_idx)] = (int(gly_idx), LinkageType(gly_val))

    return assignments

