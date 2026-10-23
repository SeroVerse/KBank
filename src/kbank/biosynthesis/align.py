"""Cross-domain geno-glycomic alignment between gene clusters and glycan graphs.

Provides `TargetType`, `AssignmentEvidence`, `BiosyntheticRule`, `BiosyntheticRuleRegistry`,
`PanGraphConsensusPrior`, `BiosyntheticCostStrategy`, `GeneGlycanAssignment`, `GeneGlycanAssignments`,
`LocusMatchResult`, and `BiosyntheticAssigner` for mapping biosynthetic gene clusters
(`Locus` / `Loci` from `serogen`) onto bipartite carbohydrate structures
(`Glycan` / `Glycans` from `kbank`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Self, overload

import numpy as np
from serogen.core.collections import RaggedArrayContainer, ScalarView
from serogen.core.graph.align import (
    Alignment,
    AlignmentConfig,
    CostStrategy,
    EmbeddingStrategy,
    GraphAligner,
    WeisfeilerLehmanEmbedding,
)
from serogen.core.graph.models import TAG_EDGE, GraphModel
from serogen.genes.cnn import GeneRoleClassifier
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import ArchetypeCategory, LocusArchetype
from serogen.loci.models import GeneNode, Loci, Locus

from kbank.glycan import Anomer, Chirality, Glycan, LinkageEdge, LinkageType, ResidueType
from kbank.sugars import Sugars


class TargetType(StrEnum):
    """Structural target category in a bipartite glycan graph for a biosynthetic gene role."""

    NONE = "NONE"
    REDUCING_END = "REDUCING_END"
    LINKAGE = "LINKAGE"
    MODIFIED_MONO = "MODIFIED_MONO"
    MONOSACCHARIDE = "MONOSACCHARIDE"
    ANY = "ANY"


_TARGET_TYPES: tuple[TargetType, ...] = tuple(TargetType)
_TARGET_TYPE_TO_INT: dict[TargetType, int] = {t: idx for idx, t in enumerate(_TARGET_TYPES)}


class AssignmentEvidence(IntEnum):
    """Provenance and evidence level for a gene-to-glycan component assignment."""

    RULE_INFERRED = 0
    GRAPH_ALIGNED = 1
    LITERATURE = 2
    LOCUS_COMPLETED = 3
    PAN_GRAPH_CONSENSUS = 4


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
        allowed_classes=(b"Hex", b"HexNAc", b"HexA", b"HexNA", b"DeoxyHex", b"DeoxyHexNAc", b"dHex", b"dHexNAc"),
        disallowed_classes=(b"Mod",),
    ),
    GeneRole.OXIDOREDUCTASE: _RoleGlycanSpec(
        TargetType.MONOSACCHARIDE,
        allowed_names=(
            b"GlcA",
            b"GalA",
            b"ManA",
            b"GulA",
            b"IdoA",
            b"GlcNA",
            b"GalNA",
            b"ManNA",
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
        allowed_classes=(
            b"HexA",
            b"HexNA",
            b"DeoxyHex",
            b"DeoxyHexNAc",
            b"DiDeoxyHex",
            b"dHex",
            b"dHexNAc",
            b"ddHex",
            b"dHexN",
        ),
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


def _edge_signatures(
    glycan: Glycan,
    edge_idx: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Compute structural signatures and catalytic mechanism (`1=RETAINING, 2=INVERTING`) for `edge_idx`."""
    e = glycan.edges[edge_idx]
    s_node = glycan.nodes[e.source]
    t_node = glycan.nodes[e.target]
    s_sug = int(s_node.sugar)
    s_chi = int(s_node.chirality)
    s_fur = int(s_node.is_furanose)
    anom = int(e.edge_type.anomer)
    don = int(e.edge_type.donor)
    acc = int(e.edge_type.acceptor)
    t_sug = int(t_node.sugar)
    is_bb = int(e.edge_type.is_backbone)
    is_term_leaf = int(
        s_node.is_branch
        and not any(
            oe.target == e.source and not oe.edge_type.is_polymer and not glycan.nodes[oe.source].is_modifier
            for oe in glycan.edges
        )
    )
    acc_is_branch_pt = int(
        sum(1 for oe in glycan.edges if oe.target == e.target and not glycan.nodes[oe.source].is_modifier) >= 2
    )

    sig_full = (s_sug, s_chi, s_fur, anom, don, acc, t_sug, is_bb, is_term_leaf, acc_is_branch_pt)
    sig_donor_link = (s_sug, s_chi, s_fur, anom, acc)
    sig_donor_anom = (s_sug, s_chi, anom)

    if e.edge_type.anomer == Anomer.UNKNOWN:
        mech = 0
    else:
        s_name = Sugars.names[s_sug]
        is_beta_ndp_donor = s_node.chirality == Chirality.L or s_name in (b"Rha", b"Fuc", b"6dTal", b"Col", b"Asc")
        if is_beta_ndp_donor:
            mech = 1 if e.edge_type.anomer == Anomer.BETA else 2
        else:
            mech = 1 if e.edge_type.anomer == Anomer.ALPHA else 2

    return sig_full, sig_donor_link, sig_donor_anom, mech


class PanGraphConsensusPrior:
    """Archetype- and cross-species pan-graph consensus model for `Init`, `Wzy`, and `GT` -> linkage resolution.

    Pools node-level homology (`GeneNode.cluster_id`) and edge-level syntenic transitions
    (`Loci.as_pan_graph()` and `Glycans.as_pan_graph()`) across paired records to resolve:
    1. `INITIATOR` homology `cluster_id` -> biological Und-PP initiating monosaccharide (`WbaP` -> `D-Galp`,
       `WcaJ` -> `D-Glcp`, `WecA` -> `D-GlcNAc`/`D-GalNAc`, `Itr`/`PglC` -> `D-QuiNAc`/`D-FucNAc`/`D-Gal`).
    2. `GLYCOSYLTRANSFERASE` & `POLYMERASE` homology `cluster_id` -> `(donor, anomer, acceptor_carbon, acceptor)`
       and retaining vs. inverting stereochemical mechanism.
    3. Consecutive syntenic GT transitions `(cluster_a -> cluster_b)` in the pan-locus network paired with
       consecutive glycosidic linkages `(e_1 -> e_2)` in the pan-glycan network.
    """

    def __init__(self) -> None:
        """Initialize an empty `PanGraphConsensusPrior`."""
        self.init_cluster_sugar: dict[int, dict[int, float]] = {}
        self.cluster_record_counts: dict[int, int] = {}
        self.cluster_sig_full: dict[int, dict[tuple[int, ...], float]] = {}
        self.cluster_sig_dl: dict[int, dict[tuple[int, ...], float]] = {}
        self.cluster_sig_da: dict[int, dict[tuple[int, ...], float]] = {}
        self.cluster_mech: dict[int, dict[int, float]] = {}
        self.syntenic_pair_dl: dict[tuple[int, int], dict[tuple[tuple[int, ...], tuple[int, ...]], float]] = {}
        self.init_adj_gt_step1: dict[tuple[int, int], dict[tuple[int, ...], float]] = {}
        self.precursor_gt_donor: dict[tuple[int, int], dict[int, float]] = {}
        self.precursor_cluster_sugar: dict[int, dict[int, float]] = {}
        self.syntenic_pair_record_counts: dict[tuple[int, int], int] = {}
        self.pan_locus_edge_counts: dict[tuple[int, int], int] = {}
        self.pan_glycan_node_counts: dict[int, int] = {}
        self.pan_glycan_bigram_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        self.modifier_cluster_spec: dict[int, dict[tuple[int, int, int], float]] = {}
        self.fine_to_med: dict[int, int] = {}
        self.rescued_loci: list[Locus] = []
        self.classifier_suite: Any | None = None

    def preferred_init_sugar(
        self,
        cluster_id: int,
        candidate_sugars: set[int],
        aa_seq: bytes | None = None,
    ) -> int | None:
        """Return the highest-scoring initiating sugar index in `candidate_sugars` for `cluster_id`, or `None`."""
        if (
            cluster_id > 0
            and cluster_id in self.init_cluster_sugar
            and self.cluster_record_counts.get(cluster_id, 0) >= 2
        ):
            s_map = self.init_cluster_sugar[cluster_id]
            best_s: int | None = None
            best_val = 0.0
            for s_idx in candidate_sugars:
                val = s_map.get(s_idx, 0.0)
                if val > best_val:
                    best_val = val
                    best_s = s_idx
            if best_s is not None and best_val >= 0.5:
                return best_s
        if self.classifier_suite is not None:
            clf_s, clf_p = self.classifier_suite.preferred_init_sugar(cluster_id, candidate_sugars, aa_seq=aa_seq)
            if clf_s is not None and clf_p >= 0.40:
                return clf_s
        if cluster_id > 0 and cluster_id in self.init_cluster_sugar:
            s_map = self.init_cluster_sugar[cluster_id]
            best_s = None
            best_val = 0.0
            for s_idx in candidate_sugars:
                val = s_map.get(s_idx, 0.0)
                if val > best_val:
                    best_val = val
                    best_s = s_idx
            return best_s if best_val >= 0.5 else None
        return None

    def preferred_precursor_sugar(self, cluster_id: int, candidate_sugars: set[int]) -> int | None:
        """Return the multi-record consensus sugar synthesized by precursor `cluster_id` among `candidate_sugars`."""
        for c_tier in (cluster_id, self.fine_to_med.get(cluster_id, 0)):
            if c_tier <= 0 or self.cluster_record_counts.get(c_tier, 0) < 2:
                continue
            s_map = self.precursor_cluster_sugar.get(c_tier, {})
            if not s_map:
                continue
            cand_w = {s_idx: s_map.get(s_idx, 0.0) for s_idx in candidate_sugars if s_map.get(s_idx, 0.0) > 0.0}
            if not cand_w:
                continue
            tot_cand = sum(cand_w.values())
            best_s, best_w = max(cand_w.items(), key=lambda kv: kv[1])
            if best_w >= 1.5 and best_w >= 0.55 * tot_cand:
                return best_s
        return None

    def has_init_consensus(self, cluster_id: int, sugar_idx: int, aa_seq: bytes | None = None) -> bool:
        """Return True if `cluster_id` has multi-record pan-graph or classifier consensus for initiating on `sugar_idx`."""
        if cluster_id > 0 and self.cluster_record_counts.get(cluster_id, 0) >= 2:
            s_map = self.init_cluster_sugar.get(cluster_id, {})
            tot = sum(s_map.values())
            if tot > 0 and s_map.get(sugar_idx, 0.0) >= 0.45 * tot:
                return True
        if self.classifier_suite is not None:
            clf_s, clf_p = self.classifier_suite.preferred_init_sugar(cluster_id, {sugar_idx}, aa_seq=aa_seq)
            if clf_s == sugar_idx and clf_p >= 0.65:
                return True
        return False

    def gt_edge_bonus(
        self,
        cluster_id: int,
        glycan: Glycan,
        edge_idx: int,
        adj_clusters: Sequence[tuple[int, int]] = (),
        adj_init_clusters: Sequence[int] = (),
        adj_precursor_clusters: Sequence[int] = (),
        adj_precursor_sugars: set[int] | frozenset[int] = frozenset(),
        aa_seq: bytes | None = None,
    ) -> tuple[float, bool]:
        """Compute cross-archetype pan-graph homology & synteny cost bonus and whether pan-consensus holds."""
        sig_full, sig_dl, sig_da, mech = _edge_signatures(glycan, edge_idx)
        bonus = 0.0
        has_consensus = False

        red_idx = glycan.reducing_end_index
        wzy_k, _, _ = glycan.polymer_edge_and_assembly_depths(red_idx)
        all_core = [
            k
            for k, e in enumerate(glycan.edges)
            if not glycan.nodes[e.source].is_modifier and not glycan.nodes[e.target].is_modifier
        ]
        intra_core = [k for k in all_core if k != wzy_k]
        cand_edges = intra_core if (intra_core and edge_idx != wzy_k) else all_core

        for c_tier, tier_scale in ((cluster_id, 1.0), (self.fine_to_med.get(cluster_id, 0), 0.72)):
            if c_tier <= 0 or self.cluster_record_counts.get(c_tier, 0) < 2:
                continue

            full_map = self.cluster_sig_full.get(c_tier, {})
            dl_map = self.cluster_sig_dl.get(c_tier, {})
            da_map = self.cluster_sig_da.get(c_tier, {})

            gly_full_w = {
                _edge_signatures(glycan, k)[0]: full_map.get(_edge_signatures(glycan, k)[0], 0.0) for k in cand_edges
            }
            gly_dl_w = {
                _edge_signatures(glycan, k)[1]: dl_map.get(_edge_signatures(glycan, k)[1], 0.0) for k in cand_edges
            }
            gly_da_w = {
                _edge_signatures(glycan, k)[2]: da_map.get(_edge_signatures(glycan, k)[2], 0.0) for k in cand_edges
            }

            w_full = gly_full_w.get(sig_full, 0.0)
            max_gly_full = max(gly_full_w.values(), default=0.0)
            other_full = max((v for k_s, v in gly_full_w.items() if k_s != sig_full), default=0.0)

            w_dl = gly_dl_w.get(sig_dl, 0.0)
            max_gly_dl = max(gly_dl_w.values(), default=0.0)
            other_dl = max((v for k_s, v in gly_dl_w.items() if k_s != sig_dl), default=0.0)

            w_da = gly_da_w.get(sig_da, 0.0)
            max_gly_da = max(gly_da_w.values(), default=0.0)
            other_da = max((v for k_s, v in gly_da_w.items() if k_s != sig_da), default=0.0)

            if w_full >= 0.60 and w_full == max_gly_full and w_full >= other_full * 1.20 and w_da >= max_gly_da * 0.65:
                bonus -= 0.13 * tier_scale
                has_consensus = True
            elif max_gly_full >= 0.85 and w_full < max_gly_full * 0.50:
                bonus += 0.09 * tier_scale

            if w_dl >= 0.60 and w_dl == max_gly_dl and w_dl >= other_dl * 1.20 and w_da >= max_gly_da * 0.65:
                bonus -= 0.11 * tier_scale
                has_consensus = True
            elif max_gly_dl >= 0.85 and w_dl < max_gly_dl * 0.50:
                bonus += 0.11 * tier_scale

            if w_da >= 0.65 and w_da == max_gly_da and w_da >= other_da * 1.20:
                bonus -= 0.08 * tier_scale
                has_consensus = True
            elif max_gly_da >= 0.90 and w_da < max_gly_da * 0.45:
                bonus += 0.16 * tier_scale

            mech_map = self.cluster_mech.get(c_tier, {})
            tot_mech = sum(mech_map.values())
            if mech > 0 and tot_mech >= 1.2:
                frac_mech = mech_map.get(mech, 0.0) / tot_mech
                if frac_mech >= 0.68:
                    bonus -= 0.03 * tier_scale * frac_mech
                elif frac_mech <= 0.18:
                    bonus += 0.12 * tier_scale

        e_curr = glycan.edges[edge_idx]
        donor_s = glycan.nodes[e_curr.source].sugar

        # Signal 1: (INITIATOR_cluster -> GT_cluster) step-1 anchoring onto the biological reducing end
        if e_curr.target == red_idx and cluster_id > 0:
            step1_edges = [k for k in cand_edges if glycan.edges[k].target == red_idx]
            for c_init in adj_init_clusters:
                for c_tier in (cluster_id, self.fine_to_med.get(cluster_id, 0)):
                    if c_tier > 0 and self.cluster_record_counts.get(c_tier, 0) >= 2:
                        step1_w = self.init_adj_gt_step1.get((c_init, c_tier), {}).get(sig_dl, 0.0)
                        if step1_w >= 0.85:
                            bonus -= 0.07
                            has_consensus = True
                            break
                if self.cluster_record_counts.get(c_init, 0) >= 2 and len(step1_edges) == 1:
                    bonus -= 0.055
                    has_consensus = True
                    break

        # Signal 2: (Precursor_Operon -> GT_cluster) syntenic donor-sugar coupling
        if cluster_id > 0:
            for c_prec in adj_precursor_clusters:
                for c_tier in (cluster_id, self.fine_to_med.get(cluster_id, 0)):
                    if c_tier <= 0 or self.cluster_record_counts.get(c_tier, 0) < 2:
                        continue
                    prec_map = self.precursor_gt_donor.get((c_prec, c_tier), {})
                    if prec_map:
                        max_prec_w = max(prec_map.values(), default=0.0)
                        w_prec = prec_map.get(donor_s, 0.0)
                        if w_prec >= 0.85 and w_prec >= max_prec_w * 0.80:
                            bonus -= 0.06
                            has_consensus = True
                            break

        if adj_precursor_sugars and donor_s in adj_precursor_sugars:
            bonus -= 0.055
            matching_cand_donors = {
                glycan.nodes[glycan.edges[k].source].sugar
                for k in cand_edges
                if glycan.nodes[glycan.edges[k].source].sugar in adj_precursor_sugars
            }
            matching_cand_sigs = {
                _edge_signatures(glycan, k)[1]
                for k in cand_edges
                if glycan.nodes[glycan.edges[k].source].sugar == donor_s
            }
            if len(matching_cand_donors) == 1 and len(matching_cand_sigs) == 1:
                has_consensus = True

        # Signal 3: Syntenic (GT_A -> GT_B) x (Linkage_1 -> Linkage_2) bigrams weighted by Pan-Glycan network
        if cluster_id > 0:
            for c_other, dir_flag in adj_clusters:
                if c_other <= 0:
                    continue
                for c_o_tier, c_s_tier in (
                    (c_other, cluster_id),
                    (self.fine_to_med.get(c_other, 0), self.fine_to_med.get(cluster_id, 0)),
                ):
                    if c_o_tier <= 0 or c_s_tier <= 0:
                        continue
                    pair_key = (c_o_tier, c_s_tier) if dir_flag < 0 else (c_s_tier, c_o_tier)
                    if self.syntenic_pair_record_counts.get(pair_key, 0) < 2:
                        continue
                    pair_map = self.syntenic_pair_dl.get(pair_key, {})
                    if not pair_map:
                        continue
                    for k_other in cand_edges:
                        if k_other == edge_idx:
                            continue
                        e_other = glycan.edges[k_other]
                        if e_curr.target == e_other.source or e_curr.source == e_other.target:
                            _, sig_dl_other, _, _ = _edge_signatures(glycan, k_other)
                            sig_pair = (sig_dl_other, sig_dl) if dir_flag < 0 else (sig_dl, sig_dl_other)
                            sup = pair_map.get(sig_pair, 0.0)
                            if sup >= 0.85:
                                bonus -= 0.06
                                has_consensus = True
                                break

                # Cross-network anchor: if neighbor GT c_other has multi-record consensus on an adjacent edge k_other
                # and (dl_other, dl_curr) is a recurring disaccharide motif in the global Pan-Glycan network
                if not has_consensus and self.cluster_record_counts.get(c_other, 0) >= 2:
                    o_dl_map = self.cluster_sig_dl.get(c_other, {})
                    if o_dl_map:
                        cand_o_w = {
                            _edge_signatures(glycan, k)[1]: o_dl_map.get(_edge_signatures(glycan, k)[1], 0.0)
                            for k in cand_edges
                        }
                        max_o_w = max(cand_o_w.values(), default=0.0)
                        if max_o_w >= 0.80:
                            for k_other in cand_edges:
                                if k_other == edge_idx:
                                    continue
                                _, dl_other, _, _ = _edge_signatures(glycan, k_other)
                                if cand_o_w.get(dl_other, 0.0) < max_o_w:
                                    continue
                                e_other = glycan.edges[k_other]
                                if e_curr.target == e_other.source or e_curr.source == e_other.target:
                                    sig_pair = (dl_other, sig_dl) if dir_flag < 0 else (sig_dl, dl_other)
                                    bg_cnt = self.pan_glycan_bigram_counts.get(sig_pair, 0)
                                    if bg_cnt >= 2:
                                        bonus -= 0.045
                                        adj_sigs = {
                                            _edge_signatures(glycan, k_c)[1]
                                            for k_c in cand_edges
                                            if k_c != k_other
                                            and (
                                                glycan.edges[k_c].target == e_other.source
                                                or glycan.edges[k_c].source == e_other.target
                                            )
                                        }
                                        if len(adj_sigs) == 1:
                                            has_consensus = True
                                        break

        # Hybrid fallback: for singleton/unresolved GT clusters or unclustered genes (cluster_id <= 0),
        # query the factorized 1D-CNN BiosyntheticClassifierSuite (donor, linkage, mechanism)
        if self.classifier_suite is not None and (
            not has_consensus or self.cluster_record_counts.get(cluster_id, 0) < 2
        ):
            clf_b, clf_cons = self.classifier_suite.gt_edge_score(
                cluster_id,
                glycan,
                edge_idx,
                aa_seq=aa_seq,
            )
            scale_clf = 1.0 if self.cluster_record_counts.get(cluster_id, 0) < 2 else 0.55
            bonus += scale_clf * clf_b
            if clf_cons:
                has_consensus = True

        return float(np.clip(bonus, -0.32, 0.45)), has_consensus

    @classmethod
    def fit(
        cls,
        glycans: Sequence[Glycan],
        loci: Loci | Sequence[Locus],
        archetypes: Sequence[LocusArchetype | int] | np.ndarray,
        is_repeating: Sequence[bool] | np.ndarray,
        *,
        assigner: BiosyntheticAssigner | None = None,
        em_iterations: int = 5,
        classifier_suite: Any | None = None,
    ) -> tuple[Self, list[Glycan]]:
        """Fit a `PanGraphConsensusPrior` across all paired records and rotate glycans to biological reducing ends."""
        from collections import Counter, defaultdict

        from scipy.optimize import linear_sum_assignment

        from kbank.biosynthesis.classifier import BiosyntheticClassifierSuite
        from kbank.glycan import Glycans

        prior = cls()
        suite = classifier_suite if classifier_suite is not None else BiosyntheticClassifierSuite.pretrained()
        if suite is not None and suite.is_fitted:
            prior.classifier_suite = suite
        n = len(glycans)
        if n == 0:
            return prior, list(glycans)

        loci_batch = loci if isinstance(loci, Loci) else Loci.from_list(list(loci))
        glycans_batch = Glycans.from_list(list(glycans))

        # 1. Extract pan-locus and pan-glycan consensus transition topologies via .as_pan_graph()
        if len(loci_batch.node_types) > 0:
            role_cluster_mask = int(GeneNode.mask("ROLE_ID", "CLUSTER_ID"))
            pan_l_adj, pan_l_nodes, _ = loci_batch.as_pan_graph(
                node_mask=role_cluster_mask,
                canonicalize_strands=False,
            )
            coo_l = pan_l_adj.tocoo()
            for r_i, c_i, cnt in zip(coo_l.row, coo_l.col, coo_l.data, strict=False):
                c_src = int(GeneNode(int(pan_l_nodes[int(r_i)])).cluster_id)
                c_tgt = int(GeneNode(int(pan_l_nodes[int(c_i)])).cluster_id)
                if c_src > 0 and c_tgt > 0:
                    prior.pan_locus_edge_counts[(c_src, c_tgt)] = prior.pan_locus_edge_counts.get(
                        (c_src, c_tgt), 0
                    ) + int(cnt)

        if len(glycans_batch.node_types) > 0:
            _, pan_g_nodes, pan_g_counts = glycans_batch.as_pan_graph()
            for g_n, g_c in zip(pan_g_nodes, pan_g_counts, strict=False):
                prior.pan_glycan_node_counts[int(g_n)] = int(g_c)

        # Build Medium/Coarse superfamily cluster hierarchy when protein sequences are available (by_role=True)
        cluster_rec_sets: dict[int, set[int]] = defaultdict(set)
        if (
            len(loci_batch.node_types) > 0
            and len(loci_batch.genes) == len(loci_batch.node_types)
            and bool(np.any(loci_batch.cluster_ids > 0))
        ):
            med_loci = loci_batch.cluster_genes(
                mode="mcl",
                min_identity=20.0,
                min_bitscore=35.0,
                inflation=1.6,
                by_role=True,
            )
            for fine_c, med_c in zip(loci_batch.cluster_ids, med_loci.cluster_ids, strict=False):
                fc = int(fine_c)
                mc = int(med_c)
                if fc > 0 and mc > 0:
                    prior.fine_to_med[fc] = mc + 100000

        # 2. Pass 0: Accumulate INITIATOR & Precursor cluster -> sugar specificity & rotate biological reducing ends
        canon_init_names = {
            b"GlcNAc",
            b"GalNAc",
            b"QuiNAc",
            b"FucNAc",
            b"FucN4N",
            b"QuiN4N",
            b"Bac",
            b"Gal",
            b"Glc",
        }
        canon_init_ids = {Sugars.name_to_index[nm] for nm in canon_init_names if nm in Sugars.name_to_index}
        housekeeping_names = {b"Glc", b"Gal", b"GlcNAc"}
        housekeeping_ids = {Sugars.name_to_index[nm] for nm in housekeeping_names if nm in Sugars.name_to_index}
        precursor_roles = {
            GeneRole.OXIDOREDUCTASE,
            GeneRole.EPIMERASE,
            GeneRole.AMINOTRANSFERASE,
            GeneRole.ALDOLASE,
            GeneRole.NUCLEOTIDASE,
            GeneRole.NUCLEOTIDYLTRANSFERASE,
        }
        helper = assigner if assigner is not None else BiosyntheticAssigner()
        init_sugar_acc: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        prec_sugar_acc: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        mod_spec_acc: dict[int, dict[tuple[int, int, int], float]] = defaultdict(lambda: defaultdict(float))
        mod_roles_set = {
            GeneRole.ACETYLTRANSFERASE,
            GeneRole.PYRUVYLTRANSFERASE,
            GeneRole.METHYLTRANSFERASE,
            GeneRole.PHOSPHOTRANSFERASE,
            GeneRole.FORMYLTRANSFERASE,
            GeneRole.ALANINE_TRANSFERASE,
        }

        raw_locus_list = [loci_batch[i] for i in range(n)]
        for i in range(n):
            loc = raw_locus_list[i]
            gly = glycans[i]
            if len(loc.nodes) == 0 or len(gly.nodes) == 0:
                continue
            for g_node in loc.nodes:
                cid = int(g_node.cluster_id)
                if cid > 0:
                    cluster_rec_sets[cid].add(i)
                    mc = prior.fine_to_med.get(cid, 0)
                    if mc > 0:
                        cluster_rec_sets[mc].add(i)

            n_ac_mods, _ = helper._find_n_acetyl_nodes(gly)
            for g_node in loc.nodes:
                if g_node.role in mod_roles_set and g_node.cluster_id > 0:
                    cid = int(g_node.cluster_id)
                    m_targets = helper.registry.get_rule(g_node.role).target_modifiers
                    for e_mod in gly.edges:
                        if 0 <= e_mod.source < len(gly.nodes) and 0 <= e_mod.target < len(gly.nodes):
                            s_m = gly.nodes[e_mod.source]
                            t_m = gly.nodes[e_mod.target]
                            if (
                                s_m.is_modifier
                                and e_mod.source not in n_ac_mods
                                and s_m.sugar in m_targets
                                and not t_m.is_modifier
                            ):
                                m_key = (
                                    int(s_m.sugar),
                                    int(t_m.sugar),
                                    int(e_mod.edge_type.chemical_int),
                                )
                                for t_cid in (cid, prior.fine_to_med.get(cid, 0)):
                                    if t_cid > 0:
                                        mod_spec_acc[t_cid][m_key] += 1.0

            core_sugars = {nd.sugar for nd in gly.nodes if not nd.is_modifier}
            for g_node in loc.nodes:
                if g_node.role in precursor_roles and g_node.cluster_id > 0:
                    cid = int(g_node.cluster_id)
                    compat = helper.registry.get_rule(g_node.role).compatible_sugars
                    cand_s = (core_sugars & compat) if compat else core_sugars
                    spec_s = (cand_s - housekeeping_ids) or cand_s
                    if spec_s:
                        w_p = 2.0 if len(spec_s) == 1 else (1.0 / len(spec_s))
                        for t_cid in (cid, prior.fine_to_med.get(cid, 0)):
                            if t_cid > 0:
                                for s_idx in spec_s:
                                    prec_sugar_acc[t_cid][int(s_idx)] += w_p

            cyc = gly.cycle_nodes()
            cand_nodes = cyc if cyc else tuple(idx for idx, nd in enumerate(gly.nodes) if not nd.is_modifier)
            cyc_sugars = {gly.nodes[idx].sugar for idx in cand_nodes}
            canon_in_cyc = cyc_sugars & canon_init_ids
            target_sugars = canon_in_cyc if canon_in_cyc else cyc_sugars
            if not target_sugars:
                continue

            w_init = 3.0 if len(target_sugars) == 1 else (1.0 / len(target_sugars))
            for g_node in loc.nodes:
                if g_node.role == GeneRole.INITIATOR and g_node.cluster_id > 0:
                    for s_idx in target_sugars:
                        init_sugar_acc[int(g_node.cluster_id)][int(s_idx)] += w_init

        prior.cluster_record_counts = {c: len(recs) for c, recs in cluster_rec_sets.items()}
        prior.init_cluster_sugar = {c: dict(cnts) for c, cnts in init_sugar_acc.items()}
        prior.precursor_cluster_sugar = {c: dict(cnts) for c, cnts in prec_sugar_acc.items()}
        prior.modifier_cluster_spec = {c: dict(cnts) for c, cnts in mod_spec_acc.items()}

        rotated_glycans: list[Glycan] = []
        locus_list: list[Locus] = []
        bigram_acc: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()

        for i in range(n):
            gly = glycans[i]
            loc = raw_locus_list[i]
            arch = LocusArchetype(int(archetypes[i]))
            rep = bool(is_repeating[i])
            if len(gly.nodes) > 0 and rep:
                gly = helper.resolve_biological_reducing_end(
                    gly,
                    loc if len(loc.nodes) > 0 else None,
                    archetype=arch,
                    pan_prior=prior,
                )
            if len(loc.nodes) > 0 and len(gly.nodes) > 0:
                loc = helper.rescue_cassette_roles(
                    loc,
                    gly,
                    archetype=arch,
                    is_repeating=rep,
                    pan_cluster_counts=prior.cluster_record_counts,
                )
                for g_node in loc.nodes:
                    cid = int(g_node.cluster_id)
                    if cid > 0:
                        cluster_rec_sets[cid].add(i)
                        mc = prior.fine_to_med.get(cid, 0)
                        if mc > 0:
                            cluster_rec_sets[mc].add(i)

            rotated_glycans.append(gly)
            locus_list.append(loc)

            # Accumulate global Pan-Glycan consecutive linkage bigrams across ALL glycans (paired & unpaired)
            core_e = [
                k
                for k, e in enumerate(gly.edges)
                if not gly.nodes[e.source].is_modifier and not gly.nodes[e.target].is_modifier
            ]
            for k1 in core_e:
                e1 = gly.edges[k1]
                _, dl1, _, _ = _edge_signatures(gly, k1)
                for k2 in core_e:
                    if k1 == k2:
                        continue
                    e2 = gly.edges[k2]
                    if e1.source == e2.target or e1.target == e2.source:
                        _, dl2, _, _ = _edge_signatures(gly, k2)
                        bigram_acc[(dl1, dl2)] += 1

        prior.cluster_record_counts = {c: len(recs) for c, recs in cluster_rec_sets.items()}
        prior.pan_glycan_bigram_counts = dict(bigram_acc)
        prior.rescued_loci = locus_list
        if prior.classifier_suite is not None:
            prior.classifier_suite.precompute_for_loci(locus_list)

        # 3. Iterative EM across all paired records to refine GT cluster_id -> linkage & syntenic pair priors
        paired_indices = [i for i in range(n) if len(locus_list[i].nodes) > 0 and len(rotated_glycans[i].nodes) > 0]
        if not paired_indices:
            return prior, rotated_glycans

        cost_strat = BiosyntheticCostStrategy(rule_registry=helper.registry, pan_prior=prior)

        for _em_round in range(max(1, em_iterations)):
            sig_full_acc: dict[int, dict[tuple[int, ...], float]] = defaultdict(lambda: defaultdict(float))
            sig_dl_acc: dict[int, dict[tuple[int, ...], float]] = defaultdict(lambda: defaultdict(float))
            sig_da_acc: dict[int, dict[tuple[int, ...], float]] = defaultdict(lambda: defaultdict(float))
            mech_acc: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
            syn_pair_acc: dict[tuple[int, int], dict[tuple[tuple[int, ...], tuple[int, ...]], float]] = defaultdict(
                lambda: defaultdict(float)
            )
            syn_pair_recs: dict[tuple[int, int], set[int]] = defaultdict(set)
            init_step1_acc: dict[tuple[int, int], dict[tuple[int, ...], float]] = defaultdict(
                lambda: defaultdict(float)
            )
            prec_donor_acc: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))

            for i in paired_indices:
                loc = locus_list[i]
                gly = rotated_glycans[i]
                n_res = len(gly.nodes)
                n_edges = len(gly.edges)
                if n_edges == 0:
                    continue

                link_genes = [
                    g_idx
                    for g_idx, g_node in enumerate(loc.nodes)
                    if g_node.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.POLYMERASE)
                ]
                core_edges = [
                    k
                    for k, e in enumerate(gly.edges)
                    if not gly.nodes[e.source].is_modifier and not gly.nodes[e.target].is_modifier
                ]
                if not link_genes or not core_edges:
                    continue

                _, loc_labels = loc.as_graph(as_bipartite=True)
                _, gly_labels = gly.as_graph(as_bipartite=True)
                c_mat = cost_strat.compute_cost(loc_labels, gly_labels, graph_a=loc, graph_b=gly)

                bip_cols = [n_res + k for k in core_edges]
                raw_sub = c_mat[np.ix_(link_genes, bip_cols)]
                sub_c = np.where(np.isfinite(raw_sub) & (raw_sub < 1e4), raw_sub, 10.0)
                r_ind, c_ind = linear_sum_assignment(sub_c)

                red_idx = gly.reducing_end_index
                init_genes_in_loc = [
                    (idx_g, int(nd.cluster_id))
                    for idx_g, nd in enumerate(loc.nodes)
                    if nd.role == GeneRole.INITIATOR and nd.cluster_id > 0
                ]
                prec_genes_in_loc = [
                    (idx_g, int(nd.cluster_id))
                    for idx_g, nd in enumerate(loc.nodes)
                    if nd.role in precursor_roles and nd.cluster_id > 0
                ]

                gt_gene_to_edge: dict[int, tuple[int, float]] = {}
                n_matched = len(r_ind)
                for idx_m in range(n_matched):
                    r_m = int(r_ind[idx_m])
                    c_m = int(c_ind[idx_m])
                    if sub_c[r_m, c_m] >= 9.0:
                        continue
                    g_idx = link_genes[r_m]
                    k_loc = core_edges[c_m]
                    c_id = int(loc.nodes[g_idx].cluster_id)
                    if c_id <= 0:
                        continue

                    c_mm = float(sub_c[r_m, c_m])
                    sig_m = _edge_signatures(gly, k_loc)[1]
                    alt_sum = 0.0
                    for idx_o in range(n_matched):
                        if idx_o == idx_m:
                            continue
                        r_o = int(r_ind[idx_o])
                        c_o = int(c_ind[idx_o])
                        sig_o = _edge_signatures(gly, core_edges[c_o])[1]
                        if sig_m and sig_m == sig_o:
                            continue
                        c_oo = float(sub_c[r_o, c_o])
                        delta_sw = max(0.0, (float(sub_c[r_m, c_o]) + float(sub_c[r_o, c_m])) - (c_mm + c_oo))
                        alt_sum += float(np.exp(-min(delta_sw / 0.12, 30.0)))

                    w_assign = 1.0 / (1.0 + alt_sum)
                    gt_gene_to_edge[g_idx] = (k_loc, w_assign)

                    s_full, s_dl, s_da, mech = _edge_signatures(gly, k_loc)
                    for target_cid in (c_id, prior.fine_to_med.get(c_id, 0)):
                        if target_cid <= 0:
                            continue
                        sig_full_acc[target_cid][s_full] += w_assign
                        sig_dl_acc[target_cid][s_dl] += w_assign
                        sig_da_acc[target_cid][s_da] += w_assign
                        if mech > 0:
                            mech_acc[target_cid][mech] += w_assign

                    e_matched = gly.edges[k_loc]
                    donor_s = gly.nodes[e_matched.source].sugar
                    if e_matched.target == red_idx:
                        for idx_init, c_init in init_genes_in_loc:
                            if abs(g_idx - idx_init) <= 2:
                                for target_cid in (c_id, prior.fine_to_med.get(c_id, 0)):
                                    if target_cid > 0:
                                        init_step1_acc[(c_init, target_cid)][s_dl] += w_assign

                    for idx_prec, c_prec in prec_genes_in_loc:
                        if abs(g_idx - idx_prec) <= 3:
                            for target_cid in (c_id, prior.fine_to_med.get(c_id, 0)):
                                if target_cid > 0:
                                    prec_donor_acc[(c_prec, target_cid)][donor_s] += w_assign

                # Accumulate syntenic transition pairs among GT/Polymerase genes weighted by Pan-Glycan bigrams
                gt_indices = sorted(gt_gene_to_edge.keys())
                for pos_a in range(len(gt_indices) - 1):
                    g_a = gt_indices[pos_a]
                    g_b = gt_indices[pos_a + 1]
                    if g_b - g_a > 3:
                        continue
                    c_a = int(loc.nodes[g_a].cluster_id)
                    c_b = int(loc.nodes[g_b].cluster_id)
                    if c_a <= 0 or c_b <= 0:
                        continue
                    k_a, w_a = gt_gene_to_edge[g_a]
                    k_b, w_b = gt_gene_to_edge[g_b]
                    if k_a == k_b:
                        continue
                    e_a = gly.edges[k_a]
                    e_b = gly.edges[k_b]
                    if e_a.target == e_b.source or e_a.source == e_b.target:
                        _, dl_a, _, _ = _edge_signatures(gly, k_a)
                        _, dl_b, _, _ = _edge_signatures(gly, k_b)
                        bg_cnt = prior.pan_glycan_bigram_counts.get((dl_a, dl_b), 0)
                        bg_mult = 1.0 + 0.18 * float(np.log1p(bg_cnt))
                        for ca_tier, cb_tier in (
                            (c_a, c_b),
                            (prior.fine_to_med.get(c_a, 0), prior.fine_to_med.get(c_b, 0)),
                        ):
                            if ca_tier > 0 and cb_tier > 0:
                                syn_pair_acc[(ca_tier, cb_tier)][(dl_a, dl_b)] += w_a * w_b * bg_mult
                                syn_pair_recs[(ca_tier, cb_tier)].add(i)

            prior.cluster_sig_full = {c: dict(m) for c, m in sig_full_acc.items()}
            prior.cluster_sig_dl = {c: dict(m) for c, m in sig_dl_acc.items()}
            prior.cluster_sig_da = {c: dict(m) for c, m in sig_da_acc.items()}
            prior.cluster_mech = {c: dict(m) for c, m in mech_acc.items()}
            prior.syntenic_pair_dl = {p: dict(m) for p, m in syn_pair_acc.items()}
            prior.syntenic_pair_record_counts = {p: len(rs) for p, rs in syn_pair_recs.items()}
            prior.init_adj_gt_step1 = {p: dict(m) for p, m in init_step1_acc.items()}
            prior.precursor_gt_donor = {p: dict(m) for p, m in prec_donor_acc.items()}

        return prior, rotated_glycans


_LOCUS_PHAGE_PROBS_CACHE: dict[tuple[int, int, bytes], np.ndarray[tuple[int], np.dtype[np.float32]]] = {}


def _cached_locus_phage_probs(locus: Locus | None) -> np.ndarray[tuple[int], np.dtype[np.float32]] | None:
    """Return cached `GeneRoleClassifier.pretrained().predict_source_proba` vector for `locus`."""
    if locus is None or len(locus.nodes) == 0 or len(locus.genes) != len(locus.nodes):
        return None
    trans = locus.genes.translations
    if len(trans) != len(locus.nodes) or len(trans.seqs) == 0:
        return None
    key = (len(locus.nodes), len(trans.seqs), bytes(trans.seqs[:48]))
    cached = _LOCUS_PHAGE_PROBS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        probs = GeneRoleClassifier.pretrained().predict_source_proba(trans)
        if len(_LOCUS_PHAGE_PROBS_CACHE) > 4096:
            _LOCUS_PHAGE_PROBS_CACHE.clear()
        _LOCUS_PHAGE_PROBS_CACHE[key] = probs
        return probs
    except Exception:
        return None


class BiosyntheticCostStrategy(CostStrategy):
    """Domain-specific cost strategy mapping `GeneCluster` nodes to `Glycan` bipartite nodes."""

    def __init__(
        self,
        rule_registry: BiosyntheticRuleRegistry | None = None,
        pan_prior: PanGraphConsensusPrior | None = None,
    ) -> None:
        """Initialize `BiosyntheticCostStrategy` with a `BiosyntheticRuleRegistry` and `PanGraphConsensusPrior`."""
        self.registry = rule_registry or BiosyntheticRuleRegistry()
        self.pan_prior = pan_prior

    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Compute `(N_gene, N_glycan)` pairwise biochemical + pan-graph + assembly-depth cost matrix."""
        is_swapped = isinstance(graph_a, Glycan) and isinstance(graph_b, Locus)
        if is_swapped:
            gene_labels, glycan_labels = labels_b, labels_a
            glycan_graph = graph_a
            gene_graph = graph_b
        else:
            gene_labels, glycan_labels = labels_a, labels_b
            glycan_graph = graph_b if isinstance(graph_b, Glycan) else None
            gene_graph = graph_a if isinstance(graph_a, Locus) else None

        gene_translations = (
            gene_graph.genes.translations if (isinstance(gene_graph, Locus) and len(gene_graph.genes) > 0) else None
        )

        n_genes = len(gene_labels)
        n_glycan = len(glycan_labels)
        phage_probs = _cached_locus_phage_probs(gene_graph if isinstance(gene_graph, Locus) else None)
        cost = np.full((n_genes, n_glycan), np.inf, dtype=np.float64)

        parent_to_mods: dict[int, set[int]] = {j: set() for j in range(n_glycan)}
        reducing_end_indices: set[int] = set()
        linkage_to_donor_sugar: dict[int, int] = {}
        modifier_edge_indices: set[int] = set()
        n_acetyl_mod_nodes: set[int] = set()
        n_acetylated_sugar_nodes: set[int] = set()
        wzy_bipartite_idx: int | None = None
        edge_step_norm: dict[int, float] = {}
        n_res = 0

        if glycan_graph is not None:
            n_acetyl_mod_nodes, n_acetylated_sugar_nodes = BiosyntheticAssigner._find_n_acetyl_nodes(glycan_graph)
            n_res = len(glycan_graph.nodes)
            for j, res in enumerate(glycan_graph.nodes):
                if res.is_reducing_end and not res.is_modifier:
                    reducing_end_indices.add(j)
            if not reducing_end_indices and n_res > 0:
                for j, res in enumerate(glycan_graph.nodes):
                    if not res.is_modifier:
                        reducing_end_indices.add(j)
                        break

            red_idx = min(reducing_end_indices) if reducing_end_indices else 0
            wzy_edge_k, _, raw_edge_steps = glycan_graph.polymer_edge_and_assembly_depths(red_idx)
            if wzy_edge_k is not None:
                wzy_bipartite_idx = n_res + wzy_edge_k
            if raw_edge_steps:
                min_s = min(raw_edge_steps.values())
                max_s = max(raw_edge_steps.values())
                span_s = max(max_s - min_s, 1)
                for k_e, s_val in raw_edge_steps.items():
                    base_norm = (float(s_val - min_s) / float(span_s)) if max_s > min_s else 0.5
                    if 0 <= k_e < len(glycan_graph.edges) and glycan_graph.edges[k_e].edge_type.is_branch:
                        base_norm = min(1.0, base_norm + 0.25)
                    edge_step_norm[n_res + k_e] = base_norm

            for k, edge in enumerate(glycan_graph.edges):
                src, tgt = edge.source, edge.target
                s_res = glycan_graph.nodes[src]
                t_res = glycan_graph.nodes[tgt]
                bipartite_edge_idx = n_res + k
                if s_res.is_modifier and not t_res.is_modifier:
                    if src not in n_acetyl_mod_nodes:
                        parent_to_mods[tgt].add(s_res.sugar)
                    modifier_edge_indices.add(bipartite_edge_idx)
                elif t_res.is_modifier and not s_res.is_modifier:
                    if tgt not in n_acetyl_mod_nodes:
                        parent_to_mods[src].add(t_res.sugar)
                    modifier_edge_indices.add(bipartite_edge_idx)
                elif s_res.is_modifier and t_res.is_modifier:
                    modifier_edge_indices.add(bipartite_edge_idx)
                if bipartite_edge_idx < n_glycan:
                    linkage_to_donor_sugar[bipartite_edge_idx] = s_res.sugar
        else:
            for j, raw_g in enumerate(glycan_labels):
                val = int(raw_g)
                if not (val & TAG_EDGE):
                    res = ResidueType(val)
                    if res.is_reducing_end and not res.is_modifier:
                        reducing_end_indices.add(j)
                elif wzy_bipartite_idx is None and LinkageType(val).is_polymer:
                    wzy_bipartite_idx = j
            if not reducing_end_indices and n_glycan > 0 and not (int(glycan_labels[0]) & TAG_EDGE):
                reducing_end_indices.add(0)

        # Precompute GT order relative to INITIATOR in the locus for biosynthetic step tie-breaking
        precursor_roles = {
            GeneRole.OXIDOREDUCTASE,
            GeneRole.EPIMERASE,
            GeneRole.AMINOTRANSFERASE,
            GeneRole.ALDOLASE,
            GeneRole.NUCLEOTIDASE,
            GeneRole.NUCLEOTIDYLTRANSFERASE,
        }
        has_polymerase_in_locus = False
        init_gene_indices: list[int] = []
        gt_gene_indices: list[int] = []
        prec_gene_indices: list[int] = []
        gene_nodes_by_i: dict[int, GeneNode] = {}
        for i, raw_gene in enumerate(gene_labels):
            g_val = int(raw_gene)
            if g_val & TAG_EDGE:
                continue
            gn = GeneNode(g_val)
            gene_nodes_by_i[i] = gn
            if gn.role == GeneRole.POLYMERASE:
                has_polymerase_in_locus = True
            elif gn.role == GeneRole.INITIATOR:
                init_gene_indices.append(i)
            elif gn.role == GeneRole.GLYCOSYLTRANSFERASE:
                gt_gene_indices.append(i)
            elif gn.role in precursor_roles:
                prec_gene_indices.append(i)

        gt_order_norm: dict[int, float] = {}
        if gt_gene_indices:
            if init_gene_indices:
                anchor_i = init_gene_indices[0]
                ordered_gts = sorted(gt_gene_indices, key=lambda idx_g: (abs(idx_g - anchor_i), idx_g))
            else:
                ordered_gts = list(gt_gene_indices)
            n_gts = len(ordered_gts)
            for rank_idx, idx_g in enumerate(ordered_gts):
                gt_order_norm[idx_g] = (float(rank_idx) / float(n_gts - 1)) if n_gts > 1 else 0.5

        # Defer duplicate consecutive processive repeat edges from 1-to-1 Hungarian matching when intra_gt_edges > n_gts
        deferred_processive_cols: set[int] = set()
        if glycan_graph is not None and gt_gene_indices:
            intra_core_edges = [
                k
                for k, e in enumerate(glycan_graph.edges)
                if (n_res + k) not in modifier_edge_indices
                and ((n_res + k) != wzy_bipartite_idx or not has_polymerase_in_locus)
            ]
            excess_edges = len(intra_core_edges) - len(gt_gene_indices)
            if excess_edges > 0:
                for k1 in intra_core_edges:
                    if excess_edges <= 0:
                        break
                    e1 = glycan_graph.edges[k1]
                    sig1 = _edge_signatures(glycan_graph, k1)[0][:7]
                    for k2 in intra_core_edges:
                        if (
                            k1 == k2
                            or (n_res + k1) in deferred_processive_cols
                            or (n_res + k2) in deferred_processive_cols
                        ):
                            continue
                        e2 = glycan_graph.edges[k2]
                        if e1.source == e2.target and _edge_signatures(glycan_graph, k2)[0][:7] == sig1:
                            deferred_processive_cols.add(n_res + k2)
                            excess_edges -= 1
                            break

        core_sugars_in_glycan: set[int] = (
            {nd.sugar for nd in glycan_graph.nodes if not nd.is_modifier} if glycan_graph is not None else set()
        )
        gt_prec_sugars: dict[int, set[int]] = {idx_g: set() for idx_g in gt_gene_indices}
        if self.pan_prior is not None and gt_gene_indices and prec_gene_indices and core_sugars_in_glycan:
            sugar_to_prec_indices: dict[int, list[int]] = {}
            for p_i in prec_gene_indices:
                p_cid = int(gene_nodes_by_i[p_i].cluster_id)
                pref_s = self.pan_prior.preferred_precursor_sugar(p_cid, core_sugars_in_glycan)
                if pref_s is not None:
                    sugar_to_prec_indices.setdefault(pref_s, []).append(p_i)
            for s_idx, p_list in sugar_to_prec_indices.items():
                dists = {g_i: min(abs(g_i - p_i) for p_i in p_list) for g_i in gt_gene_indices}
                min_d = min(dists.values())
                if min_d <= 3:
                    for g_i, d_val in dists.items():
                        if d_val == min_d:
                            gt_prec_sugars[g_i].add(s_idx)

        for i, gn in gene_nodes_by_i.items():
            if gn.role == GeneRole.UNKNOWN:
                continue
            rule = self.registry.get_rule(gn.role_idx)
            if rule.target_type == TargetType.NONE:
                continue

            c_id = int(gn.cluster_id)
            aa_seq_i: bytes | None = (
                bytes(gene_translations[i].seq)
                if (gene_translations is not None and i < len(gene_translations))
                else None
            )
            adj_clusters: list[tuple[int, int]] = []
            adj_init_clusters: list[int] = []
            adj_prec_clusters: list[int] = []
            if self.pan_prior is not None and gn.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.POLYMERASE):
                for prev_i in gt_gene_indices:
                    if prev_i < i and (i - prev_i) <= 4:
                        adj_clusters.append((int(gene_nodes_by_i[prev_i].cluster_id), -1))
                    elif prev_i > i and (prev_i - i) <= 4:
                        adj_clusters.append((int(gene_nodes_by_i[prev_i].cluster_id), +1))
                for init_i in init_gene_indices:
                    if abs(i - init_i) <= 2 and gene_nodes_by_i[init_i].cluster_id > 0:
                        adj_init_clusters.append(int(gene_nodes_by_i[init_i].cluster_id))
                for prec_i in prec_gene_indices:
                    if abs(i - prec_i) <= 3 and gene_nodes_by_i[prec_i].cluster_id > 0:
                        adj_prec_clusters.append(int(gene_nodes_by_i[prec_i].cluster_id))

            for j, raw_gly in enumerate(glycan_labels):
                gly_val = int(raw_gly)
                is_link = bool(gly_val & TAG_EDGE)

                if is_link:
                    if j in modifier_edge_indices:
                        continue
                    if rule.matches_linkages:
                        if rule.is_polymerase:
                            if wzy_bipartite_idx is not None:
                                base_wzy = 0.01 if j == wzy_bipartite_idx else 0.65
                            else:
                                base_wzy = 0.04
                            if (
                                self.pan_prior is not None
                                and (c_id > 0 or aa_seq_i is not None)
                                and glycan_graph is not None
                            ):
                                edge_k = j - n_res
                                if 0 <= edge_k < len(glycan_graph.edges):
                                    wzy_b, _ = self.pan_prior.gt_edge_bonus(
                                        c_id,
                                        glycan_graph,
                                        edge_k,
                                        adj_clusters=adj_clusters,
                                        aa_seq=aa_seq_i,
                                    )
                                    base_wzy += 0.5 * wzy_b
                            cost[i, j] = float(np.clip(base_wzy, 0.003, 0.85))
                        else:
                            # Intra-unit GT (or synthase)
                            if j in deferred_processive_cols:
                                continue
                            if (
                                gn.role == GeneRole.GLYCOSYLTRANSFERASE
                                and wzy_bipartite_idx is not None
                                and j == wzy_bipartite_idx
                                and has_polymerase_in_locus
                            ):
                                cost[i, j] = 0.65
                                continue

                            donor_s = linkage_to_donor_sugar.get(j, -1)
                            base_c = 0.14 if (donor_s in rule.compatible_sugars or not rule.compatible_sugars) else 0.38
                            if gn.role == GeneRole.GLYCOSYLTRANSFERASE and glycan_graph is not None:
                                edge_k = j - n_res
                                if 0 <= edge_k < len(glycan_graph.edges):
                                    if self.pan_prior is not None and (c_id > 0 or aa_seq_i is not None):
                                        pan_bonus, _ = self.pan_prior.gt_edge_bonus(
                                            c_id,
                                            glycan_graph,
                                            edge_k,
                                            adj_clusters=adj_clusters,
                                            adj_init_clusters=adj_init_clusters,
                                            adj_precursor_clusters=adj_prec_clusters,
                                            adj_precursor_sugars=gt_prec_sugars.get(i, set()),
                                            aa_seq=aa_seq_i,
                                        )
                                        base_c += pan_bonus
                                    if phage_probs is not None and i < len(phage_probs):
                                        p_ph = float(phage_probs[i])
                                        if p_ph >= 0.50:
                                            if glycan_graph.edges[edge_k].edge_type.is_branch:
                                                base_c -= 0.055 * p_ph
                                            elif glycan_graph.edges[edge_k].edge_type.is_backbone:
                                                base_c += 0.035 * p_ph
                                    if i in gt_order_norm and j in edge_step_norm:
                                        base_c += 0.045 * abs(gt_order_norm[i] - edge_step_norm[j])
                            cost[i, j] = float(np.clip(base_c, 0.005, 0.85))
                else:
                    res = ResidueType(gly_val)
                    sugar_idx = res.sugar
                    is_red = j in reducing_end_indices
                    attached_mods = parent_to_mods.get(j, set())

                    if rule.target_type == TargetType.REDUCING_END:
                        if is_red and sugar_idx in rule.compatible_sugars:
                            c_red = 0.015
                            if self.pan_prior is not None and self.pan_prior.has_init_consensus(
                                c_id, sugar_idx, aa_seq=aa_seq_i
                            ):
                                c_red = 0.005
                            cost[i, j] = c_red
                        elif not res.is_modifier and sugar_idx in rule.compatible_sugars:
                            cost[i, j] = 0.50

                    elif rule.target_type == TargetType.MODIFIED_MONO:
                        if res.is_modifier:
                            if j not in n_acetyl_mod_nodes and sugar_idx in rule.target_modifiers:
                                mod_c = 0.05
                                if (
                                    gn.role == GeneRole.ACETYLTRANSFERASE
                                    and phage_probs is not None
                                    and i < len(phage_probs)
                                    and float(phage_probs[i]) >= 0.50
                                ):
                                    mod_c = 0.025
                                cost[i, j] = mod_c
                        elif gn.role == GeneRole.DEACETYLASE:
                            if j not in n_acetylated_sugar_nodes and sugar_idx in rule.target_modifiers:
                                cost[i, j] = 0.08
                        elif attached_mods & rule.target_modifiers:
                            cost[i, j] = 0.12

                    elif rule.target_type == TargetType.MONOSACCHARIDE:
                        if not res.is_modifier and sugar_idx in rule.compatible_sugars:
                            cost[i, j] = 0.10

        return cost.T if is_swapped else cost


@dataclass(slots=True, frozen=True)
class GeneGlycanAssignment(ScalarView):
    """Resolved assignment of a gene (or theoretical `GeneRole`) to a structural element in a `Glycan`."""

    gene_index: int
    role: GeneRole
    bipartite_glycan_index: int
    target_type: TargetType
    residue: ResidueType | None = None
    linkage: LinkageType | None = None
    edge_index: int | None = None
    evidence: AssignmentEvidence = AssignmentEvidence.GRAPH_ALIGNED
    score: float = 1.0

    @property
    def is_aligned(self) -> bool:
        """True if this assignment links a specific ORF (`gene_index >= 0`) from a paired `Locus`."""
        return self.gene_index >= 0 and self.evidence != AssignmentEvidence.RULE_INFERRED

    @property
    def is_pan_consensus(self) -> bool:
        """True if this assignment is supported by multi-record pan-graph homology or synteny consensus."""
        return self.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS

    @property
    def is_locus_completed(self) -> bool:
        """True if this assignment targets a modifier node artificially completed from a paired `Locus`."""
        return self.evidence == AssignmentEvidence.LOCUS_COMPLETED

    @property
    def is_backbone_linkage(self) -> bool:
        """True if this assignment targets a main-chain backbone glycosidic linkage."""
        return bool(self.linkage is not None and self.linkage.is_backbone)

    @property
    def is_branch_linkage(self) -> bool:
        """True if this assignment targets a side-chain branch glycosidic linkage."""
        return bool(self.linkage is not None and self.linkage.is_branch)

    @property
    def is_polymer_linkage(self) -> bool:
        """True if this assignment targets the inter-repeat `Wzy` polymer glycosidic linkage."""
        return bool(self.linkage is not None and self.linkage.is_polymer)

    @property
    def is_backbone_residue(self) -> bool:
        """True if this assignment targets a main-chain backbone core monosaccharide node."""
        return bool(self.residue is not None and self.residue.is_backbone)

    @property
    def is_branch_residue(self) -> bool:
        """True if this assignment targets a side-chain branch core monosaccharide node."""
        return bool(self.residue is not None and self.residue.is_branch)

    @property
    def target_index(self) -> int:
        """0-based index into `glycan.edges` (if linkage) or `glycan.nodes` (if residue/modifier)."""
        if self.edge_index is not None and self.edge_index >= 0:
            return self.edge_index
        return self.bipartite_glycan_index

    @property
    def component_int(self) -> int:
        """Bit-packed 32-bit integer (`LinkageType` with `TAG_EDGE=1` or `ResidueType` with `TAG_EDGE=0`)."""
        if self.linkage is not None:
            return int(self.linkage)
        if self.residue is not None:
            return int(self.residue)
        return 0

    @property
    def component_label(self) -> str:
        """Human-readable label of the target glycan component (e.g. `'*D-Galp'`, `'a1-3'`, `'OAcp'`)."""
        if self.linkage is not None:
            return self.linkage.name.decode("ascii", errors="replace")
        if self.residue is not None:
            return repr(self.residue)
        return "?"

    def format_short(self) -> str:
        """Compact summary token (e.g. `'Init[g0]->*D-Galp'`, `'AcT[g4]*->OAcp'`, or `'OxR->L-Rhap'`)."""
        role_str = self.role.abbreviation
        gene_tag = f"[g{self.gene_index}]" if self.gene_index >= 0 else ""
        comp_mark = "*" if self.evidence == AssignmentEvidence.LOCUS_COMPLETED else ""
        return f"{role_str}{gene_tag}{comp_mark}->{self.component_label}"


@dataclass(slots=True, frozen=True)
class GeneGlycanAssignments(RaggedArrayContainer[Any, "GeneGlycanAssignments"]):
    """Ragged Structure-of-Arrays (SoA) container for cross-domain gene-to-glycan component assignments.

    Stores variable-length bipartite assignment edges per `GlycanRecord` in flat contiguous
    NumPy arrays indexed via `offsets` and `lengths` (zero nesting, zero pickle).
    """

    gene_indices: np.ndarray[tuple[int], np.dtype[np.int16]]
    roles: np.ndarray[tuple[int], np.dtype[np.int8]]
    bipartite_indices: np.ndarray[tuple[int], np.dtype[np.int16]]
    target_types: np.ndarray[tuple[int], np.dtype[np.int8]]
    components: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_indices: np.ndarray[tuple[int], np.dtype[np.int16]]
    evidences: np.ndarray[tuple[int], np.dtype[np.int8]]
    scores: np.ndarray[tuple[int], np.dtype[np.float32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        """Return the number of records in the batch."""
        return len(self.offsets)

    @property
    def total_assignments(self) -> int:
        """Return the total number of assignment edges across all records in the batch."""
        return len(self.roles)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> tuple[GeneGlycanAssignment, ...]: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> GeneGlycanAssignments: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> tuple[GeneGlycanAssignment, ...] | GeneGlycanAssignments: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> tuple[GeneGlycanAssignment, ...] | GeneGlycanAssignments:
        """Access a single record's assignments tuple by int index, or slice/mask the batch container."""
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("GeneGlycanAssignments index out of range")
            off = int(self.offsets[idx])
            ln = int(self.lengths[idx])
            if ln == 0:
                return ()
            out: list[GeneGlycanAssignment] = []
            for k in range(off, off + ln):
                comp_val = int(self.components[k])
                is_link = bool(comp_val & TAG_EDGE)
                e_idx = int(self.edge_indices[k])
                out.append(
                    GeneGlycanAssignment(
                        gene_index=int(self.gene_indices[k]),
                        role=GeneRole(int(self.roles[k])),
                        bipartite_glycan_index=int(self.bipartite_indices[k]),
                        target_type=_TARGET_TYPES[int(self.target_types[k])],
                        residue=None if is_link else ResidueType(comp_val),
                        linkage=LinkageType(comp_val) if is_link else None,
                        edge_index=e_idx if e_idx >= 0 else None,
                        evidence=AssignmentEvidence(int(self.evidences[k])),
                        score=float(self.scores[k]),
                    )
                )
            return tuple(out)

        indices, new_offsets, new_lengths, sliced = self._slice_ragged(
            item,
            self.gene_indices,
            self.roles,
            self.bipartite_indices,
            self.target_types,
            self.components,
            self.edge_indices,
            self.evidences,
            self.scores,
        )
        if len(indices) == 0:
            return self.empty()
        return GeneGlycanAssignments(
            gene_indices=sliced[0],
            roles=sliced[1],
            bipartite_indices=sliced[2],
            target_types=sliced[3],
            components=sliced[4],
            edge_indices=sliced[5],
            evidences=sliced[6],
            scores=sliced[7],
            offsets=new_offsets,
            lengths=new_lengths,
        )

    @classmethod
    def empty(cls, length: int = 0) -> Self:
        """Create an empty `GeneGlycanAssignments` container."""
        return cls(
            gene_indices=np.empty(0, dtype=np.int16),
            roles=np.empty(0, dtype=np.int8),
            bipartite_indices=np.empty(0, dtype=np.int16),
            target_types=np.empty(0, dtype=np.int8),
            components=np.empty(0, dtype=np.uint32),
            edge_indices=np.empty(0, dtype=np.int16),
            evidences=np.empty(0, dtype=np.int8),
            scores=np.empty(0, dtype=np.float32),
            offsets=np.zeros(length, dtype=np.int32) if length > 0 else np.empty(0, dtype=np.int32),
            lengths=np.zeros(length, dtype=np.int32) if length > 0 else np.empty(0, dtype=np.int32),
        )

    @classmethod
    def from_list(cls, records_assignments: Sequence[Sequence[GeneGlycanAssignment]]) -> Self:
        """Compile a sequence of per-record `GeneGlycanAssignment` tuples into a flat SoA container."""
        n = len(records_assignments)
        if n == 0:
            return cls.empty(0)
        lengths = np.array([len(a) for a in records_assignments], dtype=np.int32)
        offsets = cls.lengths_to_offsets(lengths)
        tot = int(offsets[-1] + lengths[-1])
        if tot == 0:
            return cls.empty(n)

        gene_indices = np.empty(tot, dtype=np.int16)
        roles = np.empty(tot, dtype=np.int8)
        bipartite_indices = np.empty(tot, dtype=np.int16)
        target_types = np.empty(tot, dtype=np.int8)
        components = np.empty(tot, dtype=np.uint32)
        edge_indices = np.empty(tot, dtype=np.int16)
        evidences = np.empty(tot, dtype=np.int8)
        scores = np.empty(tot, dtype=np.float32)

        cursor = 0
        for rec_assigns in records_assignments:
            for item in rec_assigns:
                gene_indices[cursor] = item.gene_index
                roles[cursor] = int(item.role)
                bipartite_indices[cursor] = item.bipartite_glycan_index
                target_types[cursor] = _TARGET_TYPE_TO_INT.get(item.target_type, 0)
                components[cursor] = item.component_int
                edge_indices[cursor] = item.edge_index if item.edge_index is not None else -1
                evidences[cursor] = int(item.evidence)
                scores[cursor] = float(item.score)
                cursor += 1

        return cls(
            gene_indices=gene_indices,
            roles=roles,
            bipartite_indices=bipartite_indices,
            target_types=target_types,
            components=components,
            edge_indices=edge_indices,
            evidences=evidences,
            scores=scores,
            offsets=offsets,
            lengths=lengths,
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `GeneGlycanAssignments` containers into a single batch."""
        batch_list = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty(0)
        if len(batch_list) == 1:
            return batch_list[0]

        lengths = np.concatenate([b.lengths for b in batch_list])
        offsets = cls.lengths_to_offsets(lengths)
        non_empty = [b for b in batch_list if b.total_assignments > 0]
        if not non_empty:
            return cls.empty(len(lengths))

        return cls(
            gene_indices=np.concatenate([b.gene_indices for b in non_empty]),
            roles=np.concatenate([b.roles for b in non_empty]),
            bipartite_indices=np.concatenate([b.bipartite_indices for b in non_empty]),
            target_types=np.concatenate([b.target_types for b in non_empty]),
            components=np.concatenate([b.components for b in non_empty]),
            edge_indices=np.concatenate([b.edge_indices for b in non_empty]),
            evidences=np.concatenate([b.evidences for b in non_empty]),
            scores=np.concatenate([b.scores for b in non_empty]),
            offsets=offsets,
            lengths=lengths,
        )


@dataclass(slots=True, frozen=True)
class LocusMatchResult:
    """Result of matching and scoring a candidate `Locus` against a target `Glycan`."""

    locus_index: int
    locus: Locus
    archetype: LocusArchetype
    alignment: Alignment
    combined_score: float
    bio_score: float
    assignments: tuple[GeneGlycanAssignment, ...]


class BiosyntheticAssigner(GraphAligner):
    """Assigns candidate biosynthetic `Loci` to `Glycan` structures and maps individual genes to linkages/residues."""

    __slots__ = ("classifier_suite",)

    def __init__(
        self,
        config: AlignmentConfig | None = None,
        embedding_strategy: EmbeddingStrategy | None = None,
        sugars_db: type[Sugars] | Sugars = Sugars,
        pan_prior: PanGraphConsensusPrior | None = None,
        classifier_suite: Any | None = None,
    ) -> None:
        """Initialize `BiosyntheticAssigner` as a domain-configured `GraphAligner` with optional `PanGraphConsensusPrior`."""
        if pan_prior is not None and classifier_suite is not None:
            pan_prior.classifier_suite = classifier_suite
        cfg = config or AlignmentConfig(iterations=2, topo_weight=0.05, as_bipartite=True)
        registry = BiosyntheticRuleRegistry(sugars_db=sugars_db)
        super().__init__(
            cost_strategy=BiosyntheticCostStrategy(rule_registry=registry, pan_prior=pan_prior),
            embedding_strategy=embedding_strategy or WeisfeilerLehmanEmbedding(iterations=cfg.iterations),
            config=cfg,
        )
        object.__setattr__(self, "classifier_suite", classifier_suite)

    @property
    def registry(self) -> BiosyntheticRuleRegistry:
        """Return the `BiosyntheticRuleRegistry` from the underlying `BiosyntheticCostStrategy`."""
        if isinstance(self.cost_strategy, BiosyntheticCostStrategy):
            return self.cost_strategy.registry
        return BiosyntheticRuleRegistry()

    @property
    def pan_prior(self) -> PanGraphConsensusPrior | None:
        """Return the active `PanGraphConsensusPrior` from the underlying `BiosyntheticCostStrategy`."""
        if isinstance(self.cost_strategy, BiosyntheticCostStrategy):
            return self.cost_strategy.pan_prior
        return None

    _LPS_PRIMARY_INIT_SUGARS: frozenset[bytes] = frozenset(
        {
            b"GlcNAc",
            b"GalNAc",
            b"QuiNAc",
            b"FucNAc",
            b"FucN4N",
            b"QuiN4N",
            b"Bac",
            b"RhaNAc",
            b"ManNAc",
            b"6dAltNAc",
            b"6dTalNAc",
        }
    )

    _CPS_PRIMARY_INIT_SUGARS: frozenset[bytes] = frozenset(
        {
            b"Gal",
            b"Glc",
            b"GlcNAc",
            b"GalNAc",
            b"QuiNAc",
            b"FucNAc",
            b"FucN4N",
            b"QuiN4N",
            b"Bac",
        }
    )

    def resolve_biological_reducing_end(
        self,
        glycan: Glycan,
        locus: Locus | None = None,
        archetype: LocusArchetype = LocusArchetype.UNKNOWN,
        pan_prior: PanGraphConsensusPrior | None = None,
    ) -> Glycan:
        """Rotate `glycan` reducing end (`*Sugar`) to the true biological Und-PP initiating monosaccharide.

        Uses `INITIATOR` homology `cluster_id` consensus from `pan_prior` (`WbaP` -> `D-Galp`, `WcaJ` -> `D-Glcp`,
        `WecA` -> `D-GlcNAc`/`D-GalNAc`, `Itr`/`PglC` -> `D-QuiNAc`/`D-FucNAc`/`D-Gal`) together with
        archetype-specific Und-PP transferase biochemistry to resolve arbitrary literature repeat frames.
        """
        if len(glycan.nodes) <= 1 or not glycan.has_polymer_repeat:
            return glycan

        cycle = glycan.cycle_nodes()
        if not cycle:
            return glycan

        active_prior = pan_prior if pan_prior is not None else self.pan_prior
        curr_red = glycan.reducing_end_index
        cycle_sugars = {glycan.nodes[u].sugar for u in cycle}

        pref_init_sugar: int | None = None
        wzy_cluster_id: int = 0
        if locus is not None and len(locus.nodes) > 0 and active_prior is not None:
            for g_node in locus.nodes:
                if g_node.role == GeneRole.INITIATOR and g_node.cluster_id > 0 and pref_init_sugar is None:
                    pref_init_sugar = active_prior.preferred_init_sugar(int(g_node.cluster_id), cycle_sugars)
                elif g_node.role == GeneRole.POLYMERASE and g_node.cluster_id > 0 and wzy_cluster_id == 0:
                    wzy_cluster_id = int(g_node.cluster_id)

        eff_arch = locus.archetype if (locus is not None and archetype == LocusArchetype.UNKNOWN) else archetype
        init_compat = self.registry.get_rule(GeneRole.INITIATOR).compatible_sugars
        cycle_set = set(cycle)

        def _score_cycle_node(u: int) -> tuple[float, int]:
            s_idx = glycan.nodes[u].sugar
            s_name = Sugars.names[s_idx]
            sc = 0.0

            if pref_init_sugar is not None and s_idx == pref_init_sugar:
                sc += 8.0

            if eff_arch in (LocusArchetype.LPS_WZY, LocusArchetype.LPS_ABC):
                if s_name in self._LPS_PRIMARY_INIT_SUGARS:
                    sc += 4.5
                elif s_name in (b"Gal", b"Glc"):
                    sc += 3.0
                elif s_idx in init_compat:
                    sc += 1.0
            elif eff_arch == LocusArchetype.CPS_WZY:
                if s_name in self._CPS_PRIMARY_INIT_SUGARS:
                    sc += 4.5
                elif s_idx in init_compat:
                    sc += 1.0
            else:
                if s_name in self._LPS_PRIMARY_INIT_SUGARS:
                    sc += 4.2
                elif s_name in (b"Gal", b"Glc"):
                    sc += 3.8
                elif s_idx in init_compat:
                    sc += 1.0

            if u == curr_red:
                sc += 0.35

            if active_prior is not None and wzy_cluster_id > 0:
                for k_e, e_obj in enumerate(glycan.edges):
                    if e_obj.source == u and e_obj.target in cycle_set:
                        wzy_b, _ = active_prior.gt_edge_bonus(wzy_cluster_id, glycan, k_e)
                        sc += -2.0 * wzy_b
                        break

            return (sc, -u)

        best_u = max(cycle, key=_score_cycle_node)
        return glycan.with_reducing_end(best_u)

    @staticmethod
    def infer_expected_archetype(
        antigen_hint: bytes | str = b"",
    ) -> tuple[ArchetypeCategory | None, LocusArchetype | None]:
        """Infer expected `ArchetypeCategory` and optional specific `LocusArchetype` from an antigen hint string."""
        hint = antigen_hint.decode("utf-8", errors="replace") if isinstance(antigen_hint, bytes) else str(antigen_hint)
        h = hint.strip().lower()
        if not h:
            return None, None

        # Specific archetype matches first
        if "group 2" in h or "group 3" in h or "kps" in h or "cps_abc" in h:
            return ArchetypeCategory.CAPSULE, LocusArchetype.CPS_ABC
        if "group 1" in h or "group 4" in h or "cps_wzy" in h:
            return ArchetypeCategory.CAPSULE, LocusArchetype.CPS_WZY
        if "synthase" in h or "cps_synthase" in h or "hyaluron" in h:
            return ArchetypeCategory.CAPSULE, LocusArchetype.CPS_SYNTHASE
        if "lps_abc" in h or "wzm" in h:
            return ArchetypeCategory.O_ANTIGEN, LocusArchetype.LPS_ABC
        if "lps_wzy" in h:
            return ArchetypeCategory.O_ANTIGEN, LocusArchetype.LPS_WZY
        if h.startswith(("wta", "lta", "teichoic")):
            return ArchetypeCategory.OTHER, LocusArchetype.TEICHOIC_ACID
        if "alginate" in h:
            return ArchetypeCategory.EXOPOLYSACCHARIDE, LocusArchetype.EPS_ALGINATE
        if "cellulose" in h or "bcs" in h:
            return ArchetypeCategory.EXOPOLYSACCHARIDE, LocusArchetype.EPS_BCS
        if "pga" in h or "pnag" in h or "ica" in h:
            return ArchetypeCategory.EXOPOLYSACCHARIDE, LocusArchetype.EPS_PGA
        if "pel" in h:
            return ArchetypeCategory.EXOPOLYSACCHARIDE, LocusArchetype.EPS_PEL

        # High-level category matches
        if any(k in h for k in ("cps", "k-antigen", "k antigen", "capsul", "vi-antigen", "vi antigen")):
            return ArchetypeCategory.CAPSULE, None
        if any(
            k in h
            for k in (
                "o-antigen",
                "o antigen",
                "o-specific",
                "o-polysaccharide",
                "ops",
                "lps",
                "los",
                "lipopolysaccharide",
                "lipooligosaccharide",
            )
        ):
            return ArchetypeCategory.O_ANTIGEN, None
        if any(k in h for k in ("teichoic", "wta", "lta")):
            return ArchetypeCategory.OTHER, LocusArchetype.TEICHOIC_ACID
        if any(k in h for k in ("eps", "exopolysaccharide", "biofilm")):
            return ArchetypeCategory.EXOPOLYSACCHARIDE, None

        return None, None

    _PATHWAY_NACETYL_SUGARS: frozenset[bytes] = frozenset(
        {
            b"QuiNAc",
            b"FucNAc",
            b"RhaNAc",
            b"6dAltNAc",
            b"6dTalNAc",
            b"Pse",
            b"4ePse",
            b"8ePse",
            b"Leg",
            b"4eLeg",
            b"8eLeg",
            b"Aci",
            b"8eAci",
            b"Bac",
            b"Neu5Ac",
        }
    )

    _COMPLETABLE_MODIFIER_ROLES: tuple[tuple[GeneRole, bytes], ...] = (
        (GeneRole.PYRUVYLTRANSFERASE, b"Pyr"),
        (GeneRole.METHYLTRANSFERASE, b"OMe"),
        (GeneRole.FORMYLTRANSFERASE, b"Fo"),
        (GeneRole.ALANINE_TRANSFERASE, b"DAla"),
        (GeneRole.PHOSPHOTRANSFERASE, b"P"),
        (GeneRole.ACETYLTRANSFERASE, b"OAc"),
    )

    @staticmethod
    def _select_modifier_acceptor(
        nodes: Sequence[ResidueType],
        edges: Sequence[LinkageEdge],
        role: GeneRole,
        mod_name: bytes,
    ) -> int:
        """Select the most biochemically compatible core monosaccharide index in `nodes` for `mod_name`."""
        core_indices = [idx for idx, n in enumerate(nodes) if not n.is_modifier]
        if not core_indices:
            return 0

        # Track incoming glycosidic linkages and existing modifier attachments per core residue
        occupied_carbons: dict[int, set[int]] = {idx: set() for idx in core_indices}
        mod_counts: dict[int, int] = {idx: 0 for idx in core_indices}
        same_mod_counts: dict[int, int] = {idx: 0 for idx in core_indices}
        out_non_mod: dict[int, int] = {idx: 0 for idx in core_indices}
        in_non_mod: dict[int, int] = {idx: 0 for idx in core_indices}
        mod_s_idx = Sugars.name_to_index.get(mod_name, -1)

        for e in edges:
            if e.target in occupied_carbons and e.edge_type.acceptor > 0:
                occupied_carbons[e.target].add(e.edge_type.acceptor)
            if 0 <= e.source < len(nodes) and 0 <= e.target < len(nodes):
                s_res = nodes[e.source]
                t_res = nodes[e.target]
                if s_res.is_modifier and e.target in mod_counts:
                    mod_counts[e.target] += 1
                    if s_res.sugar == mod_s_idx:
                        same_mod_counts[e.target] += 1
                elif not s_res.is_modifier and not t_res.is_modifier:
                    if e.source in out_non_mod and e.source != 0:
                        out_non_mod[e.source] += 1
                    if e.target in in_non_mod and e.source != 0:
                        in_non_mod[e.target] += 1

        def _score_candidate(u: int) -> tuple[int, int, int, int, int]:
            res = nodes[u]
            s_name = Sugars.names[res.sugar]
            s_cls = Sugars.classes[res.sugar]
            occ = occupied_carbons[u]
            free_c4_c6 = int(4 not in occ) + int(6 not in occ)
            is_branch_tip = int(in_non_mod[u] == 0 and not res.is_reducing_end)

            biochem_pref = 0
            if role == GeneRole.PYRUVYLTRANSFERASE:
                if s_name in (b"Gal", b"Glc", b"Man", b"GlcA", b"GalA", b"GalNAc", b"GlcNAc"):
                    biochem_pref = 3 if free_c4_c6 >= 2 else (2 if free_c4_c6 == 1 else 1)
                elif s_cls in (b"Hex", b"HexA", b"HexNAc"):
                    biochem_pref = 1
            elif role == GeneRole.FORMYLTRANSFERASE:
                if s_name in (b"QuiN4N", b"FucN4N", b"RhaN3N", b"QuiN", b"FucN", b"RhaN", b"GlcN", b"GalN"):
                    biochem_pref = 4
                elif b"N" in s_name or s_cls in (b"dHexN", b"HexN", b"dHexNAc", b"HexNAc"):
                    biochem_pref = 2
            elif role == GeneRole.ALANINE_TRANSFERASE:
                if s_name == b"Gro" or s_cls in (b"HexA", b"HexNA"):
                    biochem_pref = 3
                elif s_cls in (b"Hex", b"HexNAc"):
                    biochem_pref = 1
            elif role == GeneRole.PHOSPHOTRANSFERASE:
                if s_name in (b"Gro", b"Gal", b"Glc", b"Man", b"GlcNAc", b"GalNAc") and free_c4_c6 >= 1:
                    biochem_pref = 3
                elif s_cls in (b"Hex", b"HexNAc", b"Pen"):
                    biochem_pref = 1
            else:
                # OAc / OMe: prefer core hexoses, deoxyhexoses, and N-acetylhexosamines with free hydroxyls
                if s_cls in (b"Hex", b"dHex", b"HexNAc", b"dHexNAc", b"HexA"):
                    biochem_pref = 2

            return (
                -same_mod_counts[u],
                biochem_pref,
                -mod_counts[u],
                is_branch_tip,
                free_c4_c6 - len(occ),
            )

        return max(core_indices, key=_score_candidate)

    def complete_glycan_modifiers(
        self,
        glycan: Glycan,
        locus: Locus | None,
        pan_prior: PanGraphConsensusPrior | None = None,
    ) -> tuple[Glycan, tuple[bytes, ...], dict[int, int]]:
        """Artificially complete `glycan` with modifier nodes inferred from excess modifier genes in `locus`.

        Subtracts 1 `ACETYLTRANSFERASE` per specialized N-acetylated sugar pathway (`QuiNAc`, `FucNAc`,
        `RhaNAc`, `Pse`, `Leg`, `Aci`, `Bac`, `Neu5Ac`) before adding `OAc` modifier nodes, while
        housekeeping `GlcNAc`/`GalNAc`/`ManNAc` consume 0 locus acetyltransferases.

        Returns:
            tuple[Glycan, tuple[bytes, ...], dict[int, int]]:
                - Completed `Glycan` graph (or original `glycan` if no modifiers were added).
                - Tuple of added modifier byte names (e.g. `(b"OAc", b"Pyr")`).
                - Mapping `{new_modifier_node_index: locus_gene_index}` for `LOCUS_COMPLETED` assignments.
        """
        if locus is None or len(locus.nodes) == 0 or len(glycan.nodes) == 0:
            return glycan, (), {}

        core_indices = [idx for idx, n in enumerate(glycan.nodes) if not n.is_modifier]
        if not core_indices:
            return glycan, (), {}

        n_acetyl_mod_nodes, _ = self._find_n_acetyl_nodes(glycan)
        pathway_act_sugars = {
            Sugars.names[n.sugar]
            for n in glycan.nodes
            if not n.is_modifier and Sugars.names[n.sugar] in self._PATHWAY_NACETYL_SUGARS
        }
        n_pathway_act = len(pathway_act_sugars)

        new_nodes = list(glycan.nodes)
        new_edges = list(glycan.edges)
        completed_names: list[bytes] = []
        completed_node_to_gene: dict[int, int] = {}

        for role, mod_bytes in self._COMPLETABLE_MODIFIER_ROLES:
            if mod_bytes not in Sugars.name_to_index:
                continue
            rule = self.registry.get_rule(role)
            role_gene_indices = [g_i for g_i, g_node in enumerate(locus.nodes) if g_node.role == role]
            if not role_gene_indices:
                continue

            existing_mods = [
                j
                for j, res in enumerate(new_nodes)
                if res.is_modifier and j not in n_acetyl_mod_nodes and res.sugar in rule.target_modifiers
            ]
            accounted = len(existing_mods) + (n_pathway_act if role == GeneRole.ACETYLTRANSFERASE else 0)
            if len(role_gene_indices) <= accounted:
                continue

            excess_genes = role_gene_indices[accounted:]
            mod_sugar_idx = Sugars.name_to_index[mod_bytes]
            for g_idx in excess_genes:
                best_u = self._select_modifier_acceptor(new_nodes, new_edges, role, mod_bytes)
                mod_edge_type = LinkageType(Anomer.UNKNOWN, donor=0, acceptor=0)
                cid_mod = int(locus.nodes[g_idx].cluster_id)
                if pan_prior is not None and cid_mod > 0:
                    spec_map = pan_prior.modifier_cluster_spec.get(cid_mod) or pan_prior.modifier_cluster_spec.get(
                        pan_prior.fine_to_med.get(cid_mod, 0), {}
                    )
                    if spec_map:
                        occupied_on: dict[int, set[int]] = {u_i: set() for u_i in core_indices}
                        for e_ex in new_edges:
                            if e_ex.target in occupied_on and e_ex.edge_type.acceptor > 0:
                                occupied_on[e_ex.target].add(int(e_ex.edge_type.acceptor))
                        for (m_sug, acc_sug, lnk_int), _w_m in sorted(
                            spec_map.items(), key=lambda kv: kv[1], reverse=True
                        ):
                            if m_sug != mod_sugar_idx:
                                continue
                            cand_lt = LinkageType(int(lnk_int)).with_topology(is_backbone=False, is_polymer=False)
                            matching_u = [
                                u_i
                                for u_i in core_indices
                                if int(new_nodes[u_i].sugar) == acc_sug
                                and (cand_lt.acceptor == 0 or int(cand_lt.acceptor) not in occupied_on.get(u_i, set()))
                            ]
                            if not matching_u:
                                matching_u = [u_i for u_i in core_indices if int(new_nodes[u_i].sugar) == acc_sug]
                            if matching_u:
                                best_u = matching_u[0]
                                mod_edge_type = cand_lt
                                break
                new_mod_idx = len(new_nodes)
                new_nodes.append(ResidueType.from_sugar(mod_sugar_idx, Chirality.UNKNOWN, False))
                new_edges.append(
                    LinkageEdge(
                        source=new_mod_idx,
                        target=best_u,
                        edge_type=mod_edge_type,
                    )
                )
                completed_names.append(mod_bytes)
                completed_node_to_gene[new_mod_idx] = g_idx

        if not completed_names:
            return glycan, (), {}

        completed_glycan = Glycan(
            canonical=glycan.canonical,
            nodes=tuple(new_nodes),
            edges=tuple(new_edges),
        )
        return completed_glycan, tuple(completed_names), completed_node_to_gene

    def rescue_cassette_roles(
        self,
        locus: Locus,
        glycan: Glycan,
        archetype: LocusArchetype = LocusArchetype.UNKNOWN,
        is_repeating: bool = True,
        pan_cluster_counts: dict[int, int] | None = None,
    ) -> Locus:
        """Promote variable-cassette `UNKNOWN` or duplicate non-conserved transport/hydrolase ORFs to `GT`/`POLYMERASE` when `locus` has a linkage deficit."""
        if len(locus.nodes) == 0 or len(glycan.nodes) == 0:
            return locus

        core_edges = [
            k
            for k, e in enumerate(glycan.edges)
            if not glycan.nodes[e.source].is_modifier and not glycan.nodes[e.target].is_modifier
        ]
        n_core = len(core_edges)
        if n_core == 0:
            return locus

        eff_arch = locus.archetype if archetype == LocusArchetype.UNKNOWN else archetype
        is_synthase_arch = eff_arch in (
            LocusArchetype.CPS_SYNTHASE,
            LocusArchetype.EPS_ALGINATE,
            LocusArchetype.EPS_BCS,
            LocusArchetype.EPS_PGA,
            LocusArchetype.EPS_PEL,
        )
        if is_synthase_arch:
            return locus

        is_wzy_arch = (
            eff_arch in (LocusArchetype.LPS_WZY, LocusArchetype.CPS_WZY, LocusArchetype.UNKNOWN)
            and is_repeating
            and n_core >= 2
        )

        gt_indices = [i for i, nd in enumerate(locus.nodes) if nd.role == GeneRole.GLYCOSYLTRANSFERASE]
        poly_indices = [i for i, nd in enumerate(locus.nodes) if nd.role == GeneRole.POLYMERASE]
        synth_indices = [i for i, nd in enumerate(locus.nodes) if nd.role == GeneRole.TRANSMEMBRANE_SYNTHASE]
        n_link_genes = len(gt_indices) + len(poly_indices) + len(synth_indices)

        deficit = max(0, n_core - n_link_genes)
        if deficit == 0:
            return locus

        cassette_anchors = [
            i
            for i, nd in enumerate(locus.nodes)
            if nd.role
            in (
                GeneRole.GLYCOSYLTRANSFERASE,
                GeneRole.POLYMERASE,
                GeneRole.INITIATOR,
                GeneRole.FLIPPASE,
            )
        ]
        if not cassette_anchors:
            cassette_anchors = list(range(len(locus.nodes)))

        gene_lengths = locus.lengths
        counts_map = pan_cluster_counts or {}

        # Identify duplicate non-conserved FLIPPASE / CAPSULE_ANCHOR ORFs when conserved backbone hubs exist
        dup_rescue_set: set[int] = set()
        for dup_role in (GeneRole.FLIPPASE, GeneRole.CAPSULE_ANCHOR):
            role_idxs = [i for i, nd in enumerate(locus.nodes) if nd.role == dup_role]
            if len(role_idxs) >= 2:
                role_idxs.sort(
                    key=lambda idx_g: (
                        counts_map.get(int(locus.nodes[idx_g].cluster_id), 0),
                        int(gene_lengths[idx_g]) if idx_g < len(gene_lengths) else 1000,
                    ),
                    reverse=True,
                )
                for extra_i in role_idxs[1:]:
                    if counts_map.get(int(locus.nodes[extra_i].cluster_id), 0) <= 8:
                        dup_rescue_set.add(extra_i)

        phage_probs = _cached_locus_phage_probs(locus)

        candidates: list[tuple[int, int, int, int, int]] = []
        for i, nd in enumerate(locus.nodes):
            bp_len = int(gene_lengths[i]) if i < len(gene_lengths) else 1000
            if bp_len < 420:
                continue
            if phage_probs is not None and i < len(phage_probs) and float(phage_probs[i]) >= 0.60:
                continue
            c_freq = counts_map.get(int(nd.cluster_id), 0)
            dist = min(abs(i - a_i) for a_i in cassette_anchors)
            if nd.role == GeneRole.UNKNOWN and c_freq <= 12:
                candidates.append((0, dist, c_freq, -bp_len, i))
            elif nd.role == GeneRole.GLYCOSYLHYDROLASE and c_freq <= 8:
                candidates.append((1, dist, c_freq, -bp_len, i))
            elif i in dup_rescue_set:
                candidates.append((2, dist, c_freq, -bp_len, i))

        if not candidates:
            return locus

        candidates.sort()
        selected = candidates[:deficit]
        new_nodes = list(locus.nodes)
        has_poly = len(poly_indices) >= 1

        for _, _, _, _, g_idx in selected:
            old_nd = new_nodes[g_idx]
            target_role = GeneRole.POLYMERASE if (is_wzy_arch and not has_poly) else GeneRole.GLYCOSYLTRANSFERASE
            if target_role == GeneRole.POLYMERASE:
                has_poly = True
            new_nodes[g_idx] = GeneNode(
                target_role,
                cluster_id=int(old_nd.cluster_id),
                is_frameshift=old_nd.is_frameshift,
            )

        return Locus(
            nodes=tuple(new_nodes),
            edges=locus.edges,
            genes=locus.genes,
            interval=locus.interval,
            contig_index=locus.contig_index,
            archetype=locus.archetype,
            completeness=locus.completeness,
            score=locus.score,
            is_tandem_split=locus.is_tandem_split,
            name=locus.name,
        )

    @staticmethod
    def _find_locus_completed_mod_nodes(glycan: Glycan) -> set[int]:
        """Identify modifier node indices attached via locus-completed wildcard linkages (`donor=0, acceptor=0`)."""
        completed_nodes: set[int] = set()
        for e in glycan.edges:
            if (
                0 <= e.source < len(glycan.nodes)
                and glycan.nodes[e.source].is_modifier
                and e.edge_type.donor == 0
                and e.edge_type.acceptor == 0
                and e.edge_type.anomer == Anomer.UNKNOWN
            ):
                completed_nodes.add(e.source)
        return completed_nodes

    def assign_genes(
        self,
        locus: Locus,
        glycan: Glycan,
        alignment: Alignment | None = None,
        completed_modifier_genes: dict[int, int] | None = None,
        pan_prior: PanGraphConsensusPrior | None = None,
    ) -> tuple[GeneGlycanAssignment, ...]:
        """Map active biosynthetic genes in `locus` to their aligned residues or linkages in `glycan`."""
        if len(locus.nodes) == 0 or len(glycan.nodes) == 0:
            return ()

        active_prior = pan_prior if pan_prior is not None else self.pan_prior
        if alignment is None:
            if active_prior is not None and active_prior is not self.pan_prior:
                temp_assigner = BiosyntheticAssigner(
                    config=self.config,
                    embedding_strategy=self.embedding_strategy,
                    sugars_db=self.registry.sugars_db,
                    pan_prior=active_prior,
                    classifier_suite=self.classifier_suite,
                )
                alignment = temp_assigner.align(locus, glycan)
            else:
                alignment = self.align(locus, glycan)

        _, cluster_labels = locus.as_graph(as_bipartite=True)
        _, glycan_labels = glycan.as_graph(as_bipartite=True)
        n_res = len(glycan.nodes)
        n_genes = len(locus.nodes)

        cost_strat = (
            self.cost_strategy
            if (isinstance(self.cost_strategy, BiosyntheticCostStrategy) and active_prior is self.pan_prior)
            else BiosyntheticCostStrategy(rule_registry=self.registry, pan_prior=active_prior)
        )
        c_mat = cost_strat.compute_cost(cluster_labels, glycan_labels, graph_a=locus, graph_b=glycan)

        completed_mod_nodes = self._find_locus_completed_mod_nodes(glycan)
        if completed_modifier_genes:
            completed_mod_nodes |= set(completed_modifier_genes.keys())
        gene_to_completed_mod: dict[int, int] = (
            {g_i: m_i for m_i, g_i in completed_modifier_genes.items()} if completed_modifier_genes else {}
        )

        out: list[GeneGlycanAssignment] = []
        matched_gene_indices: set[int] = set()
        matched_mod_nodes: set[int] = set()

        # 1. Explicit locus-completed modifier gene assignments first
        for g_idx, mod_idx in gene_to_completed_mod.items():
            if 0 <= g_idx < n_genes and 0 <= mod_idx < n_res:
                g_node = locus.nodes[g_idx]
                res = glycan.nodes[mod_idx]
                out.append(
                    GeneGlycanAssignment(
                        gene_index=int(g_idx),
                        role=g_node.role,
                        bipartite_glycan_index=int(mod_idx),
                        target_type=TargetType.MODIFIED_MONO,
                        residue=res,
                        linkage=None,
                        edge_index=None,
                        evidence=AssignmentEvidence.LOCUS_COMPLETED,
                        score=0.85,
                    )
                )
                matched_gene_indices.add(int(g_idx))
                matched_mod_nodes.add(int(mod_idx))

        # Pre-collect linkage matches and pan-graph support for calibrated Boltzmann swap marginals
        precursor_roles = {
            GeneRole.OXIDOREDUCTASE,
            GeneRole.EPIMERASE,
            GeneRole.AMINOTRANSFERASE,
            GeneRole.ALDOLASE,
            GeneRole.NUCLEOTIDASE,
            GeneRole.NUCLEOTIDYLTRANSFERASE,
        }
        gt_gene_indices = [
            idx_g
            for idx_g, nd in enumerate(locus.nodes)
            if nd.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.POLYMERASE)
        ]
        init_gene_indices = [idx_g for idx_g, nd in enumerate(locus.nodes) if nd.role == GeneRole.INITIATOR]
        prec_gene_indices = [idx_g for idx_g, nd in enumerate(locus.nodes) if nd.role in precursor_roles]
        core_sugars_in_glycan = {nd.sugar for nd in glycan.nodes if not nd.is_modifier}
        gt_prec_sugars: dict[int, set[int]] = {idx_g: set() for idx_g in gt_gene_indices}
        if active_prior is not None and gt_gene_indices and prec_gene_indices and core_sugars_in_glycan:
            sugar_to_prec_indices: dict[int, list[int]] = {}
            for p_i in prec_gene_indices:
                p_cid = int(locus.nodes[p_i].cluster_id)
                pref_s = active_prior.preferred_precursor_sugar(p_cid, core_sugars_in_glycan)
                if pref_s is not None:
                    sugar_to_prec_indices.setdefault(pref_s, []).append(p_i)
            for s_idx, p_list in sugar_to_prec_indices.items():
                dists = {g_i: min(abs(g_i - p_i) for p_i in p_list) for g_i in gt_gene_indices}
                min_d = min(dists.values())
                if min_d <= 3:
                    for g_i, d_val in dists.items():
                        if d_val == min_d:
                            gt_prec_sugars[g_i].add(s_idx)

        core_bipartite_cols = [
            n_res + k
            for k, e in enumerate(glycan.edges)
            if not glycan.nodes[e.source].is_modifier and not glycan.nodes[e.target].is_modifier
        ]
        linkage_matches: dict[int, int] = {}
        direct_pan_link_genes: set[int] = set()
        loc_translations = locus.genes.translations if len(locus.genes) > 0 else None

        for g_idx, gly_idx in alignment.matches.items():
            if (
                g_idx in matched_gene_indices
                or g_idx >= n_genes
                or g_idx >= len(cluster_labels)
                or gly_idx >= len(glycan_labels)
            ):
                continue
            g_val = int(cluster_labels[g_idx])
            gly_val = int(glycan_labels[gly_idx])
            if not (g_val & TAG_EDGE) and (gly_val & TAG_EDGE):
                linkage_matches[int(g_idx)] = int(gly_idx)
                edge_idx = int(gly_idx) - n_res
                if active_prior is not None and 0 <= edge_idx < len(glycan.edges):
                    gn_obj = GeneNode(g_val)
                    c_id = int(gn_obj.cluster_id)
                    aa_seq_g: bytes | None = (
                        bytes(loc_translations[g_idx].seq)
                        if (loc_translations is not None and g_idx < len(loc_translations))
                        else None
                    )
                    adj_clusters: list[tuple[int, int]] = []
                    for other_i in gt_gene_indices:
                        if other_i < g_idx and (g_idx - other_i) <= 4:
                            adj_clusters.append((int(locus.nodes[other_i].cluster_id), -1))
                        elif other_i > g_idx and (other_i - g_idx) <= 4:
                            adj_clusters.append((int(locus.nodes[other_i].cluster_id), +1))
                    adj_init = [
                        int(locus.nodes[init_i].cluster_id)
                        for init_i in init_gene_indices
                        if abs(g_idx - init_i) <= 2 and locus.nodes[init_i].cluster_id > 0
                    ]
                    adj_prec = [
                        int(locus.nodes[prec_i].cluster_id)
                        for prec_i in prec_gene_indices
                        if abs(g_idx - prec_i) <= 3 and locus.nodes[prec_i].cluster_id > 0
                    ]
                    _, has_pan = active_prior.gt_edge_bonus(
                        c_id,
                        glycan,
                        edge_idx,
                        adj_clusters=adj_clusters,
                        adj_init_clusters=adj_init,
                        adj_precursor_clusters=adj_prec,
                        adj_precursor_sugars=gt_prec_sugars.get(int(g_idx), set()),
                        aa_seq=aa_seq_g,
                    )
                    if has_pan:
                        direct_pan_link_genes.add(int(g_idx))

        # Stoichiometric elimination: if all remaining unresolved intra-unit GTs map to edges with a single
        # distinct (donor, linkage) signature (and Wzy locks the polymer edge), they are uniquely resolved
        wzy_edge_k, _, _ = glycan.polymer_edge_and_assembly_depths()
        wzy_bip_col = (n_res + wzy_edge_k) if wzy_edge_k is not None else -1
        has_wzy_locked = any(
            locus.nodes[g_i].role == GeneRole.POLYMERASE and b_i == wzy_bip_col for g_i, b_i in linkage_matches.items()
        )
        for g_i, b_i in linkage_matches.items():
            if locus.nodes[g_i].role == GeneRole.POLYMERASE and b_i == wzy_bip_col:
                direct_pan_link_genes.add(g_i)

        unresolved_gt_genes = [
            g_i
            for g_i, b_i in linkage_matches.items()
            if g_i not in direct_pan_link_genes
            and not (locus.nodes[g_i].role == GeneRole.POLYMERASE and b_i == wzy_bip_col)
        ]
        unmatched_core_cols = [col for col in core_bipartite_cols if col not in linkage_matches.values()]
        matched_dl_sigs = {
            _edge_signatures(glycan, b_i - n_res)[1]
            for b_i in linkage_matches.values()
            if 0 <= (b_i - n_res) < len(glycan.edges)
        }
        unmatched_distinct_cols = [
            col
            for col in unmatched_core_cols
            if 0 <= (col - n_res) < len(glycan.edges)
            and _edge_signatures(glycan, col - n_res)[1] not in matched_dl_sigs
        ]
        unresolved_dl_sigs = {
            _edge_signatures(glycan, linkage_matches[g_i] - n_res)[1]
            for g_i in unresolved_gt_genes
            if 0 <= (linkage_matches[g_i] - n_res) < len(glycan.edges)
        }
        if len(gt_gene_indices) == 1 or (
            len(unresolved_gt_genes) >= 1
            and len(unresolved_dl_sigs) == 1
            and len(unmatched_distinct_cols) == 0
            and (len(direct_pan_link_genes) >= 1 or has_wzy_locked)
        ):
            direct_pan_link_genes.update(unresolved_gt_genes)

        for g_idx, gly_idx in alignment.matches.items():
            if (
                g_idx in matched_gene_indices
                or g_idx >= n_genes
                or g_idx >= len(cluster_labels)
                or gly_idx >= len(glycan_labels)
            ):
                continue
            g_val = int(cluster_labels[g_idx])
            gly_val = int(glycan_labels[gly_idx])
            if g_val & TAG_EDGE:
                continue

            g_node = GeneNode(g_val)
            role = g_node.role
            if role == GeneRole.UNKNOWN:
                continue
            rule = self.registry.get_rule(role)
            if rule.target_type == TargetType.NONE:
                continue

            raw_cost = alignment.costs.get(g_idx, 0.20)
            base_quality = (
                float(np.clip(np.exp(-float(raw_cost)), 0.0, 1.0))
                if (raw_cost is not None and np.isfinite(raw_cost) and raw_cost < 1e4)
                else 0.5
            )

            if gly_val & TAG_EDGE:
                edge_idx = int(gly_idx) - n_res
                c_ii = (
                    float(c_mat[g_idx, gly_idx])
                    if np.isfinite(c_mat[g_idx, gly_idx]) and c_mat[g_idx, gly_idx] < 1e4
                    else 0.25
                )
                sig_i = _edge_signatures(glycan, edge_idx)[1] if 0 <= edge_idx < len(glycan.edges) else ()

                alt_weight_sum = 0.0
                tau = 0.085
                for g_k, b_k in linkage_matches.items():
                    if g_k == g_idx:
                        continue
                    edge_k = b_k - n_res
                    sig_k = _edge_signatures(glycan, edge_k)[1] if 0 <= edge_k < len(glycan.edges) else ()
                    if sig_i and sig_i == sig_k:
                        continue
                    c_kk = float(c_mat[g_k, b_k]) if np.isfinite(c_mat[g_k, b_k]) else 0.25
                    c_ik = float(c_mat[g_idx, b_k]) if np.isfinite(c_mat[g_idx, b_k]) else 2.0
                    c_ki = float(c_mat[g_k, gly_idx]) if np.isfinite(c_mat[g_k, gly_idx]) else 2.0
                    delta_swap = max(0.0, (c_ik + c_ki) - (c_ii + c_kk))
                    alt_weight_sum += float(np.exp(-min(delta_swap / tau, 30.0)))

                for b_u in unmatched_core_cols:
                    edge_u = b_u - n_res
                    sig_u = _edge_signatures(glycan, edge_u)[1] if 0 <= edge_u < len(glycan.edges) else ()
                    if sig_i and sig_i == sig_u:
                        continue
                    c_iu = float(c_mat[g_idx, b_u]) if np.isfinite(c_mat[g_idx, b_u]) else 2.0
                    delta_move = max(0.0, c_iu - c_ii)
                    alt_weight_sum += float(np.exp(-min(delta_move / tau, 30.0)))

                marginal_prob = 1.0 / (1.0 + alt_weight_sum)
                is_pan = int(g_idx) in direct_pan_link_genes
                if is_pan:
                    marginal_prob = max(marginal_prob, 0.85)
                match_score = float(np.clip(0.30 * np.exp(-c_ii) + 0.70 * marginal_prob, 0.10, 0.99))
                ev_link = AssignmentEvidence.PAN_GRAPH_CONSENSUS if is_pan else AssignmentEvidence.GRAPH_ALIGNED

                out.append(
                    GeneGlycanAssignment(
                        gene_index=int(g_idx),
                        role=role,
                        bipartite_glycan_index=int(gly_idx),
                        target_type=TargetType.LINKAGE,
                        residue=None,
                        linkage=LinkageType(gly_val),
                        edge_index=edge_idx if 0 <= edge_idx < len(glycan.edges) else None,
                        evidence=ev_link,
                        score=match_score,
                    )
                )
                matched_gene_indices.add(int(g_idx))
            else:
                if int(gly_idx) in matched_mod_nodes:
                    continue
                res = ResidueType(gly_val)
                t_type = (
                    TargetType.MODIFIED_MONO
                    if res.is_modifier
                    else (TargetType.REDUCING_END if res.is_reducing_end else TargetType.MONOSACCHARIDE)
                )
                if int(gly_idx) in completed_mod_nodes:
                    ev = AssignmentEvidence.LOCUS_COMPLETED
                    node_score = 0.85
                elif (
                    t_type == TargetType.REDUCING_END
                    and active_prior is not None
                    and active_prior.has_init_consensus(int(g_node.cluster_id), res.sugar)
                ):
                    ev = AssignmentEvidence.PAN_GRAPH_CONSENSUS
                    node_score = max(base_quality, 0.96)
                else:
                    ev = AssignmentEvidence.GRAPH_ALIGNED
                    node_score = base_quality
                out.append(
                    GeneGlycanAssignment(
                        gene_index=int(g_idx),
                        role=role,
                        bipartite_glycan_index=int(gly_idx),
                        target_type=t_type,
                        residue=res,
                        linkage=None,
                        edge_index=None,
                        evidence=ev,
                        score=node_score,
                    )
                )
                matched_gene_indices.add(int(g_idx))
                if res.is_modifier:
                    matched_mod_nodes.add(int(gly_idx))

        # Processive / bifunctional GT & transmembrane synthase multi-linkage completion for remaining core edges
        if unmatched_core_cols:
            cand_multi_genes = [
                idx_g
                for idx_g, nd in enumerate(locus.nodes)
                if nd.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.TRANSMEMBRANE_SYNTHASE)
            ]
            if cand_multi_genes:
                gene_lengths = locus.lengths
                term_leaves = set(glycan.terminal_leaf_indices())
                for b_u in unmatched_core_cols:
                    edge_u = b_u - n_res
                    if not (0 <= edge_u < len(glycan.edges)):
                        continue
                    if b_u == wzy_bip_col and has_wzy_locked:
                        continue
                    e_u = glycan.edges[edge_u]
                    _, sig_dl_u, sig_da_u, _ = _edge_signatures(glycan, edge_u)

                    def _score_multi_gt(
                        g_c: int,
                        _b_u: int = b_u,
                        _e_u: Any = e_u,
                        _sig_dl_u: Any = sig_dl_u,
                        _sig_da_u: Any = sig_da_u,
                    ) -> tuple[int, int, int, float, int, int]:
                        unmatched_g = int(g_c not in linkage_matches)
                        b_p = linkage_matches.get(g_c, -1)
                        edge_p = b_p - n_res
                        has_p = 0 <= edge_p < len(glycan.edges)
                        same_dl = int(has_p and _edge_signatures(glycan, edge_p)[1] == _sig_dl_u)
                        same_da = int(has_p and _edge_signatures(glycan, edge_p)[2] == _sig_da_u)
                        is_adj = int(
                            has_p
                            and (
                                glycan.edges[edge_p].target == _e_u.source or glycan.edges[edge_p].source == _e_u.target
                            )
                        )
                        c_val = (
                            float(c_mat[g_c, _b_u])
                            if np.isfinite(c_mat[g_c, _b_u])
                            else (float(c_mat[g_c, b_p]) if (has_p and np.isfinite(c_mat[g_c, b_p])) else 1.0)
                        )
                        bp_len = int(gene_lengths[g_c]) if g_c < len(gene_lengths) else 1000
                        return (unmatched_g, same_dl, same_da, -round(c_val, 3), is_adj, bp_len)

                    best_g = max(cand_multi_genes, key=_score_multi_gt)
                    b_p = linkage_matches.get(best_g, -1)
                    edge_p = b_p - n_res
                    same_dl = 0 <= edge_p < len(glycan.edges) and _edge_signatures(glycan, edge_p)[1] == sig_dl_u
                    same_da_count = sum(
                        1
                        for g_o in cand_multi_genes
                        if g_o in linkage_matches
                        and 0 <= (linkage_matches[g_o] - n_res) < len(glycan.edges)
                        and _edge_signatures(glycan, linkage_matches[g_o] - n_res)[2] == sig_da_u
                    )
                    has_pan_u = False
                    if active_prior is not None and locus.nodes[best_g].cluster_id > 0:
                        _, has_pan_u = active_prior.gt_edge_bonus(
                            int(locus.nodes[best_g].cluster_id),
                            glycan,
                            edge_u,
                        )

                    # Extra-locus GtrABC-style terminal side-chain alpha-Glc/Gal branch when all locus GTs are saturated
                    s_u_node = glycan.nodes[e_u.source]
                    s_u_name = Sugars.names[s_u_node.sugar]
                    if (
                        best_g in linkage_matches
                        and len(cand_multi_genes) >= 1
                        and locus.nodes[best_g].role != GeneRole.TRANSMEMBRANE_SYNTHASE
                        and not same_dl
                        and not has_pan_u
                        and e_u.edge_type.is_branch
                        and e_u.source in term_leaves
                        and s_u_name in (b"Glc", b"Gal")
                        and e_u.edge_type.anomer == Anomer.ALPHA
                        and (0 <= edge_p < len(glycan.edges) and glycan.edges[edge_p].edge_type.is_backbone)
                    ):
                        out.append(
                            GeneGlycanAssignment(
                                gene_index=-1,
                                role=GeneRole.GLYCOSYLTRANSFERASE,
                                bipartite_glycan_index=int(b_u),
                                target_type=TargetType.LINKAGE,
                                residue=None,
                                linkage=e_u.edge_type,
                                edge_index=edge_u,
                                evidence=AssignmentEvidence.RULE_INFERRED,
                                score=0.75,
                            )
                        )
                        continue
                    is_pan_u = (
                        has_pan_u
                        or len(cand_multi_genes) == 1
                        or (same_dl and (best_g in direct_pan_link_genes or same_da_count == 1))
                    )
                    c_u = float(c_mat[best_g, b_u]) if np.isfinite(c_mat[best_g, b_u]) else 0.20
                    score_u = (
                        float(np.clip(0.30 * np.exp(-c_u) + 0.62, 0.82, 0.96))
                        if is_pan_u
                        else float(np.clip(0.80 * np.exp(-c_u), 0.55, 0.84))
                    )
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=int(best_g),
                            role=locus.nodes[best_g].role,
                            bipartite_glycan_index=int(b_u),
                            target_type=TargetType.LINKAGE,
                            residue=None,
                            linkage=e_u.edge_type,
                            edge_index=edge_u,
                            evidence=(
                                AssignmentEvidence.PAN_GRAPH_CONSENSUS if is_pan_u else AssignmentEvidence.GRAPH_ALIGNED
                            ),
                            score=score_u,
                        )
                    )
                    matched_gene_indices.add(int(best_g))
                    if best_g not in linkage_matches:
                        linkage_matches[int(best_g)] = int(b_u)

        # Identify N-linked OAc modifier nodes (e.g. Ac(1-2) on GlcN -> GlcNAc) vs true O-acetylation
        n_acetyl_mod_nodes, n_acetylated_sugar_nodes = self._find_n_acetyl_nodes(glycan)

        # Many-to-many pathway completion: link additional precursor synthesis / modifier genes in the locus
        # (e.g., rmlABCD or Neu5Ac/Pse/Leg pathway genes) to their compatible monosaccharide / modifier nodes
        pathway_roles = (
            GeneRole.OXIDOREDUCTASE,
            GeneRole.EPIMERASE,
            GeneRole.AMINOTRANSFERASE,
            GeneRole.ALDOLASE,
            GeneRole.NUCLEOTIDASE,
            GeneRole.DEACETYLASE,
            GeneRole.PYRUVYLTRANSFERASE,
            GeneRole.METHYLTRANSFERASE,
            GeneRole.ACETYLTRANSFERASE,
            GeneRole.PHOSPHOTRANSFERASE,
            GeneRole.FORMYLTRANSFERASE,
            GeneRole.ALANINE_TRANSFERASE,
        )
        for g_idx in range(n_genes):
            if g_idx in matched_gene_indices:
                continue
            g_node = locus.nodes[g_idx]
            role = g_node.role
            if role not in pathway_roles:
                continue
            rule = self.registry.get_rule(role)
            # Prefer unmatched modifier nodes first before sharing an already-matched modifier node
            candidate_node_order = sorted(
                range(len(glycan.nodes)),
                key=lambda j: (j in matched_mod_nodes, j not in completed_mod_nodes, j),
            )
            for j in candidate_node_order:
                res = glycan.nodes[j]
                if res.is_modifier:
                    if j in n_acetyl_mod_nodes or (j in matched_mod_nodes and j in completed_mod_nodes):
                        continue
                    if res.sugar in rule.target_modifiers:
                        ev = (
                            AssignmentEvidence.LOCUS_COMPLETED
                            if j in completed_mod_nodes
                            else AssignmentEvidence.GRAPH_ALIGNED
                        )
                        out.append(
                            GeneGlycanAssignment(
                                gene_index=g_idx,
                                role=role,
                                bipartite_glycan_index=j,
                                target_type=TargetType.MODIFIED_MONO,
                                residue=res,
                                linkage=None,
                                edge_index=None,
                                evidence=ev,
                                score=0.85 if ev == AssignmentEvidence.LOCUS_COMPLETED else 0.90,
                            )
                        )
                        matched_mod_nodes.add(j)
                        break
                else:
                    if role == GeneRole.DEACETYLASE and j in n_acetylated_sugar_nodes:
                        continue
                    if res.sugar in rule.compatible_sugars or res.sugar in rule.target_modifiers:
                        out.append(
                            GeneGlycanAssignment(
                                gene_index=g_idx,
                                role=role,
                                bipartite_glycan_index=j,
                                target_type=(
                                    TargetType.MODIFIED_MONO
                                    if role == GeneRole.DEACETYLASE
                                    else TargetType.MONOSACCHARIDE
                                ),
                                residue=res,
                                linkage=None,
                                edge_index=None,
                                evidence=AssignmentEvidence.GRAPH_ALIGNED,
                                score=0.88,
                            )
                        )
                        break

        out.sort(key=lambda a: (a.gene_index if a.gene_index >= 0 else 9999, a.bipartite_glycan_index))
        return tuple(out)

    @staticmethod
    def _find_n_acetyl_nodes(glycan: Glycan) -> tuple[set[int], set[int]]:
        """Identify N-linked `OAc` modifier node indices (e.g. `Ac(1-2)` on `GlcN`) and N-acetylated sugar node indices."""
        oac_idx = Sugars.name_to_index.get(b"OAc", -1)
        n_acetyl_mods: set[int] = set()
        n_acetylated_sugars: set[int] = set()
        if oac_idx < 0:
            return n_acetyl_mods, n_acetylated_sugars

        for edge in glycan.edges:
            s_node = glycan.nodes[edge.source]
            t_node = glycan.nodes[edge.target]
            acc = edge.edge_type.acceptor
            if s_node.sugar == oac_idx and not t_node.is_modifier and not (edge.edge_type.donor == 0 and acc == 0):
                t_name = Sugars.names[t_node.sugar]
                t_cls = Sugars.classes[t_node.sugar]
                is_free_amino = (b"N" in t_name and b"NAc" not in t_name) or (t_cls in (b"HexN", b"dHexN", b"HexNA"))
                if is_free_amino and acc in (0, 2, 4, 5, 7):
                    n_acetyl_mods.add(edge.source)
                    if acc in (0, 2):
                        n_acetylated_sugars.add(edge.target)
        return n_acetyl_mods, n_acetylated_sugars

    def infer_rule_assignments(
        self,
        glycan: Glycan,
        archetype: LocusArchetype = LocusArchetype.UNKNOWN,
        is_repeating: bool = True,
    ) -> tuple[GeneGlycanAssignment, ...]:
        """Infer stoichiometric `GeneRole` -> glycan component assignments via `BiosyntheticRuleRegistry`."""
        n_res = len(glycan.nodes)
        if n_res == 0:
            return ()

        n_acetyl_mod_nodes, n_acetylated_sugar_nodes = self._find_n_acetyl_nodes(glycan)

        red_indices = {j for j, r in enumerate(glycan.nodes) if r.is_reducing_end and not r.is_modifier}
        if not red_indices:
            for j, r in enumerate(glycan.nodes):
                if not r.is_modifier:
                    red_indices.add(j)
                    break

        is_synthase_arch = archetype in (
            LocusArchetype.CPS_SYNTHASE,
            LocusArchetype.EPS_ALGINATE,
            LocusArchetype.EPS_BCS,
            LocusArchetype.EPS_PGA,
            LocusArchetype.EPS_PEL,
        )
        is_wzy_arch = archetype in (
            LocusArchetype.LPS_WZY,
            LocusArchetype.CPS_WZY,
            LocusArchetype.UNKNOWN,
        )

        mod_roles = (
            GeneRole.PYRUVYLTRANSFERASE,
            GeneRole.METHYLTRANSFERASE,
            GeneRole.ACETYLTRANSFERASE,
            GeneRole.PHOSPHOTRANSFERASE,
            GeneRole.FORMYLTRANSFERASE,
            GeneRole.ALANINE_TRANSFERASE,
        )
        oxr_rule = self.registry.get_rule(GeneRole.OXIDOREDUCTASE)
        atn_rule = self.registry.get_rule(GeneRole.AMINOTRANSFERASE)
        ald_rule = self.registry.get_rule(GeneRole.ALDOLASE)
        deac_rule = self.registry.get_rule(GeneRole.DEACETYLASE)

        out: list[GeneGlycanAssignment] = []

        # 1. Residue and Modifier Nodes
        for j, res in enumerate(glycan.nodes):
            s_idx = res.sugar
            if res.is_modifier:
                if j in n_acetyl_mod_nodes:
                    continue
                matched_mod_role = GeneRole.ACETYLTRANSFERASE
                for m_role in mod_roles:
                    if s_idx in self.registry.get_rule(m_role).target_modifiers:
                        matched_mod_role = m_role
                        break
                out.append(
                    GeneGlycanAssignment(
                        gene_index=-1,
                        role=matched_mod_role,
                        bipartite_glycan_index=j,
                        target_type=TargetType.MODIFIED_MONO,
                        residue=res,
                        linkage=None,
                        edge_index=None,
                        evidence=AssignmentEvidence.RULE_INFERRED,
                        score=0.90,
                    )
                )
            else:
                if j in red_indices:
                    init_role = GeneRole.TRANSMEMBRANE_SYNTHASE if is_synthase_arch else GeneRole.INITIATOR
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=-1,
                            role=init_role,
                            bipartite_glycan_index=j,
                            target_type=TargetType.REDUCING_END,
                            residue=res,
                            linkage=None,
                            edge_index=None,
                            evidence=AssignmentEvidence.RULE_INFERRED,
                            score=0.90,
                        )
                    )

                # Specialized monosaccharide precursor pathways & deacetylation
                s_name = Sugars.names[s_idx]
                if s_idx in deac_rule.target_modifiers and j not in n_acetylated_sugar_nodes:
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=-1,
                            role=GeneRole.DEACETYLASE,
                            bipartite_glycan_index=j,
                            target_type=TargetType.MODIFIED_MONO,
                            residue=res,
                            linkage=None,
                            edge_index=None,
                            evidence=AssignmentEvidence.RULE_INFERRED,
                            score=0.85,
                        )
                    )
                if s_idx in oxr_rule.compatible_sugars:
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=-1,
                            role=GeneRole.OXIDOREDUCTASE,
                            bipartite_glycan_index=j,
                            target_type=TargetType.MONOSACCHARIDE,
                            residue=res,
                            linkage=None,
                            edge_index=None,
                            evidence=AssignmentEvidence.RULE_INFERRED,
                            score=0.85,
                        )
                    )
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=-1,
                            role=GeneRole.EPIMERASE,
                            bipartite_glycan_index=j,
                            target_type=TargetType.MONOSACCHARIDE,
                            residue=res,
                            linkage=None,
                            edge_index=None,
                            evidence=AssignmentEvidence.RULE_INFERRED,
                            score=0.80,
                        )
                    )
                if s_idx in atn_rule.compatible_sugars and s_name not in (b"GlcN", b"GalN", b"ManN"):
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=-1,
                            role=GeneRole.AMINOTRANSFERASE,
                            bipartite_glycan_index=j,
                            target_type=TargetType.MONOSACCHARIDE,
                            residue=res,
                            linkage=None,
                            edge_index=None,
                            evidence=AssignmentEvidence.RULE_INFERRED,
                            score=0.85,
                        )
                    )
                if s_idx in ald_rule.compatible_sugars:
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=-1,
                            role=GeneRole.ALDOLASE,
                            bipartite_glycan_index=j,
                            target_type=TargetType.MONOSACCHARIDE,
                            residue=res,
                            linkage=None,
                            edge_index=None,
                            evidence=AssignmentEvidence.RULE_INFERRED,
                            score=0.85,
                        )
                    )
                    out.append(
                        GeneGlycanAssignment(
                            gene_index=-1,
                            role=GeneRole.NUCLEOTIDASE,
                            bipartite_glycan_index=j,
                            target_type=TargetType.MONOSACCHARIDE,
                            residue=res,
                            linkage=None,
                            edge_index=None,
                            evidence=AssignmentEvidence.RULE_INFERRED,
                            score=0.80,
                        )
                    )

        # 2. Glycosidic Linkage Edges
        core_edge_indices = [
            k
            for k, edge in enumerate(glycan.edges)
            if not glycan.nodes[edge.source].is_modifier and not glycan.nodes[edge.target].is_modifier
        ]
        wzy_edge_idx: int | None = None
        if is_repeating and is_wzy_arch and len(core_edge_indices) >= 2:
            red_idx = min(red_indices) if red_indices else 0
            wzy_edge_idx, _, _ = glycan.polymer_edge_and_assembly_depths(red_idx)
            if wzy_edge_idx is None:
                wzy_edge_idx = core_edge_indices[0]

        for k in core_edge_indices:
            edge = glycan.edges[k]
            if is_synthase_arch:
                edge_role = GeneRole.TRANSMEMBRANE_SYNTHASE
            elif k == wzy_edge_idx:
                edge_role = GeneRole.POLYMERASE
            else:
                edge_role = GeneRole.GLYCOSYLTRANSFERASE
            out.append(
                GeneGlycanAssignment(
                    gene_index=-1,
                    role=edge_role,
                    bipartite_glycan_index=n_res + k,
                    target_type=TargetType.LINKAGE,
                    residue=None,
                    linkage=edge.edge_type,
                    edge_index=k,
                    evidence=AssignmentEvidence.RULE_INFERRED,
                    score=0.85,
                )
            )

        return tuple(out)

    def assign_record(
        self,
        glycan: Glycan | None,
        locus: Locus | None = None,
        archetype: LocusArchetype = LocusArchetype.UNKNOWN,
        is_repeating: bool = True,
        alignment: Alignment | None = None,
        completed_modifier_genes: dict[int, int] | None = None,
        pan_prior: PanGraphConsensusPrior | None = None,
    ) -> tuple[GeneGlycanAssignment, ...]:
        """Resolve complete gene-to-glycan component assignments for a record (graph-aligned + rule fallback)."""
        if glycan is None or len(glycan.nodes) == 0:
            return ()

        rule_assigns = self.infer_rule_assignments(glycan, archetype=archetype, is_repeating=is_repeating)
        if locus is None or len(locus.nodes) == 0:
            return rule_assigns

        aligned = self.assign_genes(
            locus,
            glycan,
            alignment=alignment,
            completed_modifier_genes=completed_modifier_genes,
            pan_prior=pan_prior,
        )
        if not aligned:
            return rule_assigns

        covered_bipartite_types = {(a.bipartite_glycan_index, a.target_type) for a in aligned}
        covered_bipartite_roles = {(a.bipartite_glycan_index, a.role) for a in aligned}

        merged = list(aligned)
        for ra in rule_assigns:
            if ra.target_type in (TargetType.LINKAGE, TargetType.REDUCING_END):
                if (ra.bipartite_glycan_index, ra.target_type) in covered_bipartite_types:
                    continue
            elif (ra.bipartite_glycan_index, ra.role) in covered_bipartite_roles or (
                ra.bipartite_glycan_index,
                ra.target_type,
            ) in covered_bipartite_types:
                continue
            merged.append(ra)

        return tuple(merged)

    def score_locus(
        self,
        locus: Locus,
        glycan: Glycan,
        locus_index: int = 0,
        expected_category: ArchetypeCategory | None = None,
        expected_archetype: LocusArchetype | None = None,
    ) -> LocusMatchResult:
        """Score a single candidate `Locus` against `glycan` by combining archetype, completeness, and alignment."""
        locus = locus.canonicalize_strand()
        if len(locus.nodes) == 0 or len(glycan.nodes) == 0:
            empty_align = Alignment(
                matches={},
                costs={},
                total_cost=float("inf"),
                graph_a=locus,
                graph_b=glycan,
                unmatched_a=tuple(range(len(locus.nodes))),
                unmatched_b=tuple(range(len(glycan.nodes))),
            )
            return LocusMatchResult(
                locus_index=locus_index,
                locus=locus,
                archetype=locus.archetype,
                alignment=empty_align,
                combined_score=0.0,
                bio_score=0.0,
                assignments=(),
            )

        alignment = self.align(locus, glycan)
        assignments = self.assign_genes(locus, glycan, alignment=alignment)

        # Count non-modifier glycosidic linkages vs GT/polymerase/synthase count
        n_core_linkages = sum(
            1 for e in glycan.edges if not glycan.nodes[e.source].is_modifier and not glycan.nodes[e.target].is_modifier
        )
        roles = locus.roles
        n_gt_like = int(
            np.sum(
                (roles == int(GeneRole.GLYCOSYLTRANSFERASE))
                | (roles == int(GeneRole.POLYMERASE))
                | (roles == int(GeneRole.TRANSMEMBRANE_SYNTHASE))
            )
        )
        gt_ratio_score = (
            1.0 - min(abs(n_gt_like - n_core_linkages) / max(n_core_linkages, n_gt_like, 1), 1.0)
            if (n_core_linkages > 0 or n_gt_like > 0)
            else 0.5
        )

        # Evaluate biochemical match costs across assigned gene nodes
        matched_costs: list[float] = []
        for a in assignments:
            c_val = alignment.costs.get(a.gene_index)
            if c_val is not None and np.isfinite(c_val) and c_val < 1e4:
                matched_costs.append(float(c_val))

        if matched_costs:
            mean_match_quality = float(np.mean(np.exp(-np.asarray(matched_costs, dtype=np.float64))))
        else:
            mean_match_quality = 0.0

        bio_score = float(np.clip(0.55 * mean_match_quality + 0.45 * gt_ratio_score, 0.0, 1.0))

        # Archetype compatibility score
        arch = locus.archetype
        if expected_archetype is not None and arch == expected_archetype:
            arch_bonus = 1.0
        elif expected_category is not None and arch.category == expected_category:
            arch_bonus = 0.85
        elif expected_category is None and arch != LocusArchetype.UNKNOWN:
            arch_bonus = 0.65
        elif arch == LocusArchetype.UNKNOWN:
            arch_bonus = 0.25
        else:
            arch_bonus = 0.0

        comp = float(np.clip(locus.completeness, 0.0, 1.0))
        seg_score = float(np.tanh(max(locus.score, 0.0) / 12.0))
        intact_factor = float(
            np.clip(1.0 - 0.35 * locus.frameshift_fraction - (0.15 if locus.is_fragmented else 0.0), 0.3, 1.0)
        )
        combined = float((0.40 * arch_bonus + 0.35 * bio_score + 0.15 * comp + 0.10 * seg_score) * intact_factor)

        return LocusMatchResult(
            locus_index=locus_index,
            locus=locus,
            archetype=arch,
            alignment=alignment,
            combined_score=combined,
            bio_score=bio_score,
            assignments=assignments,
        )

    def rank_loci(
        self,
        glycan: Glycan,
        loci: Loci | Sequence[Locus],
        *,
        antigen_hint: bytes | str = b"",
        expected_category: ArchetypeCategory | None = None,
        expected_archetype: LocusArchetype | None = None,
    ) -> list[LocusMatchResult]:
        """Score and rank all candidate `Loci` against `glycan`, prioritizing matching archetypes."""
        if expected_category is None and expected_archetype is None and antigen_hint:
            expected_category, expected_archetype = self.infer_expected_archetype(antigen_hint)

        candidates = [loci[i] for i in range(len(loci))] if isinstance(loci, Loci) else list(loci)
        if not candidates:
            return []

        results = [
            self.score_locus(
                loc,
                glycan,
                locus_index=idx,
                expected_category=expected_category,
                expected_archetype=expected_archetype,
            )
            for idx, loc in enumerate(candidates)
        ]

        # If any candidate matches the expected archetype or category, filter/prioritize those first
        if expected_archetype is not None:
            exact = [r for r in results if r.archetype == expected_archetype]
            if exact:
                exact.sort(key=lambda r: r.combined_score, reverse=True)
                others = [r for r in results if r.archetype != expected_archetype]
                others.sort(key=lambda r: r.combined_score, reverse=True)
                return exact + others

        if expected_category is not None:
            cat_matches = [r for r in results if r.archetype.category == expected_category]
            if cat_matches:
                cat_matches.sort(key=lambda r: r.combined_score, reverse=True)
                others = [r for r in results if r.archetype.category != expected_category]
                others.sort(key=lambda r: r.combined_score, reverse=True)
                return cat_matches + others

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results

    def match_locus(
        self,
        glycan: Glycan,
        loci: Loci | Sequence[Locus],
        *,
        antigen_hint: bytes | str = b"",
        expected_category: ArchetypeCategory | None = None,
        expected_archetype: LocusArchetype | None = None,
        require_category_match: bool = False,
    ) -> LocusMatchResult | None:
        """Select the highest-scoring candidate `Locus` for `glycan` from `loci`."""
        if expected_category is None and expected_archetype is None and antigen_hint:
            expected_category, expected_archetype = self.infer_expected_archetype(antigen_hint)

        ranked = self.rank_loci(
            glycan,
            loci,
            antigen_hint=antigen_hint,
            expected_category=expected_category,
            expected_archetype=expected_archetype,
        )
        if not ranked:
            return None

        best = ranked[0]
        if require_category_match and expected_category is not None:
            if best.archetype.category != expected_category:
                return None
        return best
