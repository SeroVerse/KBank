"""Four-stage Neuro-Symbolic GNN + Constrained Arborescence Predictor (`Locus -> Glycan`).

Implements:
- Stage 1: Node feature initialization (`GeneRoleClassifier.transform` + 9-head `BiosyntheticClassifierSuite`
  + `PanGraphConsensusPrior` MCL cluster posteriors + `LocusArchetype`).
- Stage 2: Numba-compiled Relational Message-Passing GNN over CSR `Locus` graphs (`_csr_relational_mpnn_forward`)
  with Virtual Operon Precursor Pool (`c_pool`) gating and clamped node propagation (`known_assignments`).
- Stage 3: Contextualized multi-task biochemical potentials (`initiator_sugar`, `donor_linkage`, `donor`,
  `linkage`, `acceptor_sugar`, `is_backbone`, `wzy_linkage`, `modifier_linkage`), stoichiometric precursor
  coverage bonus, and processive/bifunctional GT multi-node expansion.
- Stage 4: Numba-compiled Chu-Liu/Edmonds upper-bound + Branch-and-Bound Constrained Spanning Arborescence
  Assembler enforcing acceptor-carbon valence uniqueness (`c_acc` non-collision) and contiguous main-chain
  backbone grammar closed by `Wzy`, returning top-`k` `GlycanPrediction` candidates.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
from numba import njit
from serogen.core.graph.arborescence import k_best_arborescences, maximum_spanning_arborescence
from serogen.core.graph.gnn import RelationalCSR, RelationalMPNN, build_multihop_csr
from serogen.core.seq.pairwise import PairwiseAligner
from serogen.core.seq.seq import Sequences
from serogen.genes.cnn import GeneRoleClassifier
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import LocusArchetype
from serogen.loci.models import GeneNode, Locus

from kbank.biosynthesis.align import (
    AssignmentEvidence,
    BiosyntheticAssigner,
    BiosyntheticRuleRegistry,
    GeneGlycanAssignment,
    PanGraphConsensusPrior,
    TargetType,
    _edge_signatures,
)
from kbank.biosynthesis.classifier import (
    MODIFIER_ROLES,
    BiosyntheticClassifierSuite,
    decode_donor_linkage_label,
    decode_modifier_linkage_label,
    encode_acceptor_motif_label,
    encode_donor_linkage_label,
    encode_donor_residue_chem,
    extract_acceptor_motif_for_edge,
    split_bifunctional_gt_domains,
)
from kbank.glycan import Anomer, Chirality, Glycan, LinkageEdge, LinkageType, ResidueType
from kbank.sugars import Sugars

if TYPE_CHECKING:
    from kbank.db.records import GlycanRecords

DEFAULT_MPNN_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "glycan_predictor_mpnn.npz"

PRECURSOR_POOL_SUGAR_NAMES: tuple[bytes, ...] = (
    b"Rha",
    b"Fuc",
    b"Man",
    b"GlcA",
    b"GalA",
    b"GlcNAc",
    b"GalNAc",
    b"QuiNAc",
    b"FucNAc",
    b"Kdo",
)

HOUSEKEEPING_SUGARS: frozenset[int] = frozenset(
    Sugars.name_to_index[name]
    for name in (
        b"Glc",
        b"Gal",
        b"GlcNAc",
        b"GalNAc",
        b"Man",
        b"ManNAc",
        b"Rib",
        b"GlcN",
        b"GalN",
    )
    if name in Sugars.name_to_index
)


# Numba Constrained Arborescence Kernel --------------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _solve_constrained_arborescences_kernel(
    score_matrix: np.ndarray,
    acceptor_carbons: np.ndarray,
    is_backbone_flags: np.ndarray,
    max_candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Find top-`k` maximum-scoring rooted spanning arborescences (root = 0) with valence & backbone constraints.

    Every non-root node `u in 1..M-1` selects a parent `w in 0..M-1` (`w != u`) such that:
    1. The parent pointers form a cycle-free directed tree rooted at `0`.
    2. No two nodes `u1, u2` with the same parent `w` attach to the same non-zero `acceptor_carbons[u]`.
    3. Every node with `is_backbone_flags[u] == 1` attaches to a parent `w` with `is_backbone_flags[w] == 1`,
       and each backbone node has at most one backbone child (forming a contiguous main chain).
    4. At most 2 children attach to the root node `0` (since C1 of `0` is occupied by Und-PP / `Wzy`).

    Args:
        score_matrix: `(M, M)` matrix where `score_matrix[u, w]` is the score of `u -> w`.
        acceptor_carbons: `(M,)` integer array of acceptor carbon positions (`0..9`) for each donor node `u`.
        is_backbone_flags: `(M,)` uint8 array (`1` = backbone, `0` = branch; root `0` is always `1`).
        max_candidates: Maximum number of top arborescences to return.

    Returns:
        `(best_parents, best_scores, n_found)` where `best_parents` has shape `(max_candidates, M)`
        and `best_scores` has shape `(max_candidates,)`.
    """
    m = score_matrix.shape[0]
    best_parents = np.full((max_candidates, m), -1, dtype=np.int32)
    best_scores = np.full(max_candidates, -1e18, dtype=np.float64)

    if m <= 1:
        best_parents[0, 0] = -1
        best_scores[0] = 0.0
        return best_parents, best_scores, 1

    # Upper bound per node for branch-and-bound pruning
    row_max = np.empty(m, dtype=np.float64)
    row_max[0] = 0.0
    for u in range(1, m):
        rm = -1e18
        for w in range(m):
            if w != u and score_matrix[u, w] > rm:
                rm = score_matrix[u, w]
        row_max[u] = rm

    suffix_ub = np.zeros(m + 1, dtype=np.float64)
    for u in range(m - 1, 0, -1):
        suffix_ub[u] = suffix_ub[u + 1] + row_max[u]

    # Pre-sort candidate parents per node u in descending score order
    sorted_parents = np.empty((m, m - 1), dtype=np.int32)
    for u in range(1, m):
        idx_k = 0
        for w in range(m):
            if w != u:
                sorted_parents[u, idx_k] = w
                idx_k += 1
        # Simple insertion sort on m-1 elements (m <= 10)
        for i_s in range(1, m - 1):
            key_w = sorted_parents[u, i_s]
            key_val = score_matrix[u, key_w]
            j_s = i_s - 1
            while j_s >= 0 and score_matrix[u, sorted_parents[u, j_s]] < key_val:
                sorted_parents[u, j_s + 1] = sorted_parents[u, j_s]
                j_s -= 1
            sorted_parents[u, j_s + 1] = key_w

    curr_parents = np.full(m, -1, dtype=np.int32)
    choice_idx = np.zeros(m, dtype=np.int32)
    used_carbon_mask = np.zeros(m, dtype=np.int32)
    bb_child_count = np.zeros(m, dtype=np.int32)
    child_count = np.zeros(m, dtype=np.int32)

    n_found = 0
    u_depth = 1
    curr_score = 0.0

    while u_depth >= 1:
        if u_depth == m:
            # Verify cycle-freeness (every node 1..m-1 reaches 0)
            is_valid_tree = True
            for start_u in range(1, m):
                curr_v = start_u
                steps = 0
                while curr_v != 0 and curr_v != -1 and steps < m:
                    curr_v = curr_parents[curr_v]
                    steps += 1
                if curr_v != 0:
                    is_valid_tree = False
                    break

            if is_valid_tree:
                # Insert into top-k sorted table
                if n_found < max_candidates or curr_score > best_scores[max_candidates - 1]:
                    ins_pos = n_found if n_found < max_candidates else max_candidates - 1
                    while ins_pos > 0 and curr_score > best_scores[ins_pos - 1]:
                        if ins_pos < max_candidates:
                            best_scores[ins_pos] = best_scores[ins_pos - 1]
                            for k_c in range(m):
                                best_parents[ins_pos, k_c] = best_parents[ins_pos - 1, k_c]
                        ins_pos -= 1
                    best_scores[ins_pos] = curr_score
                    for k_c in range(m):
                        best_parents[ins_pos, k_c] = curr_parents[k_c]
                    if n_found < max_candidates:
                        n_found += 1

            u_depth -= 1
            if u_depth >= 1:
                prev_w = curr_parents[u_depth]
                curr_score -= score_matrix[u_depth, prev_w]
                c_acc = acceptor_carbons[u_depth]
                if c_acc > 0:
                    used_carbon_mask[prev_w] &= ~(1 << c_acc)
                if is_backbone_flags[u_depth] == 1:
                    bb_child_count[prev_w] -= 1
                child_count[prev_w] -= 1
                curr_parents[u_depth] = -1
                choice_idx[u_depth] += 1
            continue

        # Try next candidate parent for u_depth
        advanced = False
        while choice_idx[u_depth] < m - 1:
            c_i = choice_idx[u_depth]
            w = sorted_parents[u_depth, c_i]
            s_uw = score_matrix[u_depth, w]

            # Branch-and-bound upper-bound check
            if n_found == max_candidates and (curr_score + s_uw + suffix_ub[u_depth + 1]) <= best_scores[max_candidates - 1]:
                # Since sorted_parents[u_depth] is sorted descending, remaining choices also fail UB
                choice_idx[u_depth] = m - 1
                break

            c_acc = acceptor_carbons[u_depth]
            is_bb = is_backbone_flags[u_depth] == 1

            # Check chemical & topological grammar constraints
            if is_bb and (is_backbone_flags[w] == 0 or bb_child_count[w] >= 1):
                choice_idx[u_depth] += 1
                continue
            if c_acc > 0 and (used_carbon_mask[w] & (1 << c_acc)) != 0:
                choice_idx[u_depth] += 1
                continue
            if child_count[w] >= 2:
                choice_idx[u_depth] += 1
                continue

            # Check immediate cycle with already assigned ancestors
            creates_cycle = False
            anc = w
            steps = 0
            while anc != 0 and anc != -1 and steps < m:
                if anc == u_depth:
                    creates_cycle = True
                    break
                anc = curr_parents[anc]
                steps += 1
            if creates_cycle:
                choice_idx[u_depth] += 1
                continue

            # Accept parent w for u_depth
            curr_parents[u_depth] = w
            curr_score += s_uw
            if c_acc > 0:
                used_carbon_mask[w] |= 1 << c_acc
            if is_bb:
                bb_child_count[w] += 1
            child_count[w] += 1
            u_depth += 1
            if u_depth < m:
                choice_idx[u_depth] = 0
            advanced = True
            break

        if not advanced:
            choice_idx[u_depth] = 0
            u_depth -= 1
            if u_depth >= 1:
                prev_w = curr_parents[u_depth]
                curr_score -= score_matrix[u_depth, prev_w]
                c_acc = acceptor_carbons[u_depth]
                if c_acc > 0:
                    used_carbon_mask[prev_w] &= ~(1 << c_acc)
                if is_backbone_flags[u_depth] == 1:
                    bb_child_count[prev_w] -= 1
                child_count[prev_w] -= 1
                curr_parents[u_depth] = -1
                choice_idx[u_depth] += 1

    return best_parents, best_scores, n_found


# Data Models ----------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class GlycanPredictorParams:
    """Hyperparameter weights governing Stage 3 node potentials and Stage 4 constrained arborescence scoring."""

    w_pan_dl: float = 3.2
    w_knn_gt: float = 2.0
    w_clf_dl: float = 1.6
    w_clf_factorized: float = 1.2
    w_mpnn_dl_free: float = 0.65
    syntenic_msg_alpha: float = 0.18
    w_backbone_log: float = 0.65
    w_precursor_bonus: float = 0.65
    w_acceptor_prob: float = 0.95
    w_acceptor_motif: float = 3.2
    w_mpnn_edge: float = 1.0
    w_pan_acc_match: float = 4.5
    w_disac_bigram: float = 0.65
    w_syntenic_step: float = 0.55

    def to_array(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Pack the 14 potential weights into a 1D `float64` NumPy array for `.npz` serialization."""
        return np.array(
            [
                self.w_pan_dl,
                self.w_knn_gt,
                self.w_clf_dl,
                self.w_clf_factorized,
                self.w_mpnn_dl_free,
                self.syntenic_msg_alpha,
                self.w_backbone_log,
                self.w_precursor_bonus,
                self.w_acceptor_prob,
                self.w_acceptor_motif,
                self.w_mpnn_edge,
                self.w_pan_acc_match,
                self.w_disac_bigram,
                self.w_syntenic_step,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Self:
        """Unpack a 1D `float64` NumPy array into a `GlycanPredictorParams` instance."""
        v = np.asarray(arr, dtype=np.float64).ravel()
        defaults = cls().to_array()
        n = min(len(v), len(defaults))
        merged = defaults.copy()
        merged[:n] = v[:n]
        return cls(
            w_pan_dl=float(merged[0]),
            w_knn_gt=float(merged[1]),
            w_clf_dl=float(merged[2]),
            w_clf_factorized=float(merged[3]),
            w_mpnn_dl_free=float(merged[4]),
            syntenic_msg_alpha=float(merged[5]),
            w_backbone_log=float(merged[6]),
            w_precursor_bonus=float(merged[7]),
            w_acceptor_prob=float(merged[8]),
            w_acceptor_motif=float(merged[9]),
            w_mpnn_edge=float(merged[10]),
            w_pan_acc_match=float(merged[11]),
            w_disac_bigram=float(merged[12]),
            w_syntenic_step=float(merged[13]),
        )


@dataclass(slots=True, frozen=True)
class GlycanPrediction:
    """Single top-`k` candidate `Glycan` predicted from a biosynthetic `Locus`."""

    glycan: Glycan
    score: float
    probability: float
    assignments: tuple[GeneGlycanAssignment, ...]
    precursor_pool: dict[bytes, float]


@dataclass(slots=True)
class _GTSlotSpec:
    """Internal specification of a single glycosidic transfer step slot in Stage 3/4."""

    gene_index: int
    role: GeneRole
    cluster_id: int
    candidates: list[tuple[ResidueType, LinkageType, float, AssignmentEvidence]]
    acceptor_probs: dict[int, float]
    p_backbone: float
    is_clamped: bool = False
    acceptor_motif_probs: dict[int, float] = field(default_factory=dict)
    mpnn_node_idx: int = -1


class GlycanPredictor:
    """Four-stage Neuro-Symbolic GNN + Constrained Arborescence Predictor (`Locus -> Glycan`)."""

    def __init__(
        self,
        *,
        pan_prior: PanGraphConsensusPrior,
        classifier_suite: BiosyntheticClassifierSuite | None = None,
        assigner: BiosyntheticAssigner | None = None,
        mpnn: RelationalMPNN | None = None,
        mpnn_dl_classes: np.ndarray | None = None,
        disaccharide_counts: dict[tuple[int, int, int], int] | None = None,
        donor_acc_counts: dict[tuple[int, int], int] | None = None,
        wzy_closure_counts: dict[tuple[int, int, int], int] | None = None,
        archetype_init_counts: dict[tuple[int, int], int] | None = None,
        donor_vocab: tuple[int, ...] = (),
        bigram_trans_matrix: np.ndarray | None = None,
        ref_gt_embeddings: np.ndarray | None = None,
        ref_gt_dl_labels: np.ndarray | None = None,
        ref_gt_acc_labels: np.ndarray | None = None,
        ref_gt_am_labels: np.ndarray | None = None,
        ref_gt_bb_labels: np.ndarray | None = None,
        ref_gt_record_indices: np.ndarray | None = None,
        ref_init_embeddings: np.ndarray | None = None,
        ref_init_sugars: np.ndarray | None = None,
        ref_init_record_indices: np.ndarray | None = None,
        ref_wzy_embeddings: np.ndarray | None = None,
        ref_wzy_links: np.ndarray | None = None,
        ref_wzy_record_indices: np.ndarray | None = None,
        multi_edge_clusters: dict[int, list[tuple[ResidueType, LinkageType]]] | None = None,
        locus_fingerprints: dict[tuple[int, ...], list[tuple[int, Glycan, tuple[GeneGlycanAssignment, ...]]]] | None = None,
        cluster_am_counts: dict[int, dict[int, float]] | None = None,
        ref_orf_seqs: tuple[bytes, ...] = (),
        ref_orf_cids: np.ndarray | None = None,
        ref_orf_roles: np.ndarray | None = None,
        orf_kmer_index: dict[bytes, list[int]] | None = None,
        params: GlycanPredictorParams | None = None,
        mpnn_include_2hop: bool = True,
    ) -> None:
        self.pan_prior = pan_prior
        self.classifier_suite = classifier_suite or BiosyntheticClassifierSuite.pretrained()
        self.assigner = assigner or BiosyntheticAssigner()
        self.mpnn = mpnn
        self.mpnn_dl_classes = mpnn_dl_classes
        self.params = params or GlycanPredictorParams()
        self.mpnn_include_2hop = bool(mpnn_include_2hop)
        self.disaccharide_counts = disaccharide_counts or {}
        self.donor_acc_counts = donor_acc_counts or {}
        self.wzy_closure_counts = wzy_closure_counts or {}
        self.archetype_init_counts = archetype_init_counts or {}
        self.donor_vocab = donor_vocab
        self.donor_to_idx = {d: i for i, d in enumerate(donor_vocab)}
        self.bigram_trans_matrix = (
            bigram_trans_matrix
            if bigram_trans_matrix is not None
            else np.zeros((len(donor_vocab), len(donor_vocab)), dtype=np.float64)
        )
        self.ref_gt_embeddings = ref_gt_embeddings
        self.ref_gt_dl_labels = ref_gt_dl_labels
        self.ref_gt_acc_labels = ref_gt_acc_labels
        self.ref_gt_am_labels = ref_gt_am_labels
        self.ref_gt_bb_labels = ref_gt_bb_labels
        self.ref_gt_record_indices = ref_gt_record_indices
        self.ref_init_embeddings = ref_init_embeddings
        self.ref_init_sugars = ref_init_sugars
        self.ref_init_record_indices = ref_init_record_indices
        self.ref_wzy_embeddings = ref_wzy_embeddings
        self.ref_wzy_links = ref_wzy_links
        self.ref_wzy_record_indices = ref_wzy_record_indices
        self.multi_edge_clusters = multi_edge_clusters or {}
        self.locus_fingerprints = locus_fingerprints or {}
        self.cluster_am_counts = cluster_am_counts or {}
        self.ref_orf_seqs = tuple(ref_orf_seqs)
        self.ref_orf_cids = ref_orf_cids
        self.ref_orf_roles = ref_orf_roles
        self._orf_kmer_index = orf_kmer_index or {}
        self._pairwise_aligner: PairwiseAligner | None = None
        self._last_mpnn_edge_scores: np.ndarray | None = None
        self._embed_cache: dict[bytes, np.ndarray | None] = {}
        self._conv_max_cache: dict[tuple[bytes, bool], np.ndarray] = {}
        self._clf_proba_cache: dict[tuple[str, bytes], np.ndarray | None] = {}

    @classmethod
    def from_records(
        cls,
        records: GlycanRecords | None = None,
        *,
        exclude_record_indices: set[int] | None = None,
        classifier_suite: BiosyntheticClassifierSuite | None = None,
        mpnn_epochs: int = 18,
        mpnn_path: Path | None = DEFAULT_MPNN_PATH,
        retrain_mpnn: bool = False,
        train_mpnn: bool | None = None,
    ) -> Self:
        """Fit `GlycanPredictor` on `GlycanRecords` (supporting held-out record masking for cross-validation)."""
        if records is None:
            from kbank.db.records import GlycanRecords as _GR

            records = _GR.pretrained()

        if train_mpnn is False:
            mpnn_epochs = 0
        elif train_mpnn is True:
            retrain_mpnn = True

        excluded = exclude_record_indices or set()
        suite = classifier_suite or BiosyntheticClassifierSuite.pretrained()
        assigner = BiosyntheticAssigner()

        # 1. Extract Pan-Glycan empirical disaccharide & Wzy polymer closure priors across all 1,532 glycans
        disac_counts: dict[tuple[int, int, int], int] = {}
        donor_acc_counts: dict[tuple[int, int], int] = {}
        wzy_closure_counts: dict[tuple[int, int, int], int] = {}
        arch_init_counts: dict[tuple[int, int], int] = {}
        donor_set: set[int] = set()

        for i in range(len(records)):
            if i in excluded:
                continue
            gly = records.glycans[i]
            if len(gly.nodes) == 0:
                continue
            r_idx = gly.reducing_end_index
            r_node = gly.nodes[r_idx]
            if not r_node.is_modifier:
                r_chem = encode_donor_residue_chem(r_node)
                donor_set.add(r_chem)
                arch_val = int(records.archetypes[i])
                arch_init_counts[(arch_val, r_chem)] = arch_init_counts.get((arch_val, r_chem), 0) + 1

            for e in gly.edges:
                if not (0 <= e.source < len(gly.nodes) and 0 <= e.target < len(gly.nodes)):
                    continue
                s_n = gly.nodes[e.source]
                t_n = gly.nodes[e.target]
                if s_n.is_modifier or t_n.is_modifier:
                    continue
                s_chem = encode_donor_residue_chem(s_n)
                t_chem = encode_donor_residue_chem(t_n)
                l_chem = int(e.edge_type.chemical_int)
                donor_set.add(s_chem)
                donor_set.add(t_chem)
                if e.edge_type.is_polymer:
                    wzy_closure_counts[(s_chem, l_chem, t_chem)] = (
                        wzy_closure_counts.get((s_chem, l_chem, t_chem), 0) + 1
                    )
                else:
                    disac_counts[(s_chem, l_chem, t_chem)] = disac_counts.get((s_chem, l_chem, t_chem), 0) + 1
                    donor_acc_counts[(s_chem, t_chem)] = donor_acc_counts.get((s_chem, t_chem), 0) + 1

        donor_vocab = tuple(sorted(donor_set))
        d_to_i = {d: idx for idx, d in enumerate(donor_vocab)}
        n_d = len(donor_vocab)
        bigram_trans = np.full((n_d, n_d), 1.0 / max(1, n_d), dtype=np.float64)
        for acc_d, acc_i in d_to_i.items():
            row_tot = 0.0
            for don_d, don_i in d_to_i.items():
                c_val = float(donor_acc_counts.get((don_d, acc_d), 0)) + 0.1
                bigram_trans[acc_i, don_i] = c_val
                row_tot += c_val
            if row_tot > 0:
                bigram_trans[acc_i, :] /= row_tot

        # 2. Build PanGraphConsensusPrior from existing curated assignments (or fit from scratch if uncurated)
        kept_indices = [i for i in range(len(records)) if i not in excluded]
        cluster_am_counts: dict[int, dict[int, float]] = {}
        if records.assignments.total_assignments > 0:
            pan_prior = PanGraphConsensusPrior()
            pan_prior.classifier_suite = suite
            cluster_recs: dict[int, set[int]] = {}
            precursor_roles = {
                GeneRole.OXIDOREDUCTASE,
                GeneRole.EPIMERASE,
                GeneRole.AMINOTRANSFERASE,
                GeneRole.ALDOLASE,
                GeneRole.NUCLEOTIDASE,
                GeneRole.NUCLEOTIDYLTRANSFERASE,
            }
            for i in kept_indices:
                loc = records.loci[i]
                gly = records.glycans[i]
                if len(loc.nodes) == 0 or len(gly.nodes) == 0:
                    continue
                assigns = records.assignments[i]
                for g_i, g_n in enumerate(loc.nodes):
                    c_id = int(g_n.cluster_id)
                    if c_id > 0:
                        cluster_recs.setdefault(c_id, set()).add(i)
                        if g_i + 1 < len(loc.nodes):
                            c_next = int(loc.nodes[g_i + 1].cluster_id)
                            if c_next > 0:
                                pan_prior.pan_locus_edge_counts[(c_id, c_next)] = (
                                    pan_prior.pan_locus_edge_counts.get((c_id, c_next), 0) + 1
                                )
                                pan_prior.syntenic_pair_record_counts[(c_id, c_next)] = (
                                    pan_prior.syntenic_pair_record_counts.get((c_id, c_next), 0) + 1
                                )

                init_cid_rec = 0
                n_ac_mods_rec, _ = assigner._find_n_acetyl_nodes(gly)
                for a in assigns:
                    if not a.is_aligned or a.gene_index < 0 or a.gene_index >= len(loc.nodes):
                        continue
                    if a.score < 0.60 and a.evidence != AssignmentEvidence.PAN_GRAPH_CONSENSUS:
                        continue
                    c_id = int(loc.nodes[a.gene_index].cluster_id)
                    if c_id <= 0:
                        continue
                    w_a = float(a.score) * (2.5 if a.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS else 1.0)
                    if a.role == GeneRole.INITIATOR and a.residue is not None and not a.residue.is_modifier:
                        init_cid_rec = c_id
                        s_map = pan_prior.init_cluster_sugar.setdefault(c_id, {})
                        s_map[int(a.residue.sugar)] = s_map.get(int(a.residue.sugar), 0.0) + w_a
                    elif a.role in precursor_roles and a.residue is not None and not a.residue.is_modifier:
                        s_map = pan_prior.precursor_cluster_sugar.setdefault(c_id, {})
                        s_map[int(a.residue.sugar)] = s_map.get(int(a.residue.sugar), 0.0) + w_a
                    elif (
                        a.role
                        in (
                            GeneRole.ACETYLTRANSFERASE,
                            GeneRole.PYRUVYLTRANSFERASE,
                            GeneRole.METHYLTRANSFERASE,
                            GeneRole.PHOSPHOTRANSFERASE,
                            GeneRole.FORMYLTRANSFERASE,
                            GeneRole.ALANINE_TRANSFERASE,
                        )
                        and a.residue is not None
                        and a.residue.is_modifier
                        and 0 <= a.bipartite_glycan_index < len(gly.nodes)
                        and a.bipartite_glycan_index not in n_ac_mods_rec
                    ):
                        for e_m in gly.edges:
                            if e_m.source == a.bipartite_glycan_index and 0 <= e_m.target < len(gly.nodes):
                                t_res_m = gly.nodes[e_m.target]
                                if not t_res_m.is_modifier:
                                    m_key = (
                                        int(a.residue.sugar),
                                        int(t_res_m.sugar),
                                        int(e_m.edge_type.chemical_int),
                                    )
                                    m_spec = pan_prior.modifier_cluster_spec.setdefault(c_id, {})
                                    m_spec[m_key] = m_spec.get(m_key, 0.0) + w_a
                    elif (
                        a.role
                        in (
                            GeneRole.GLYCOSYLTRANSFERASE,
                            GeneRole.POLYMERASE,
                            GeneRole.TRANSMEMBRANE_SYNTHASE,
                        )
                        and a.edge_index is not None
                        and 0 <= a.edge_index < len(gly.edges)
                    ):
                        sig_full, sig_dl, sig_da, mech = _edge_signatures(gly, a.edge_index)
                        f_map = pan_prior.cluster_sig_full.setdefault(c_id, {})
                        f_map[sig_full] = f_map.get(sig_full, 0.0) + w_a
                        dl_map = pan_prior.cluster_sig_dl.setdefault(c_id, {})
                        dl_map[sig_dl] = dl_map.get(sig_dl, 0.0) + w_a
                        da_map = pan_prior.cluster_sig_da.setdefault(c_id, {})
                        da_map[sig_da] = da_map.get(sig_da, 0.0) + w_a
                        if mech > 0:
                            m_map = pan_prior.cluster_mech.setdefault(c_id, {})
                            m_map[mech] = m_map.get(mech, 0.0) + w_a
                        am_code = extract_acceptor_motif_for_edge(gly, int(a.edge_index))
                        am_c_map = cluster_am_counts.setdefault(c_id, {})
                        am_c_map[am_code] = am_c_map.get(am_code, 0.0) + w_a
                        e_obj = gly.edges[a.edge_index]
                        if init_cid_rec > 0 and e_obj.target == gly.reducing_end_index:
                            st1_map = pan_prior.init_adj_gt_step1.setdefault((init_cid_rec, c_id), {})
                            st1_map[sig_dl] = st1_map.get(sig_dl, 0.0) + w_a

            for c_id, r_set in cluster_recs.items():
                pan_prior.cluster_record_counts[c_id] = len(r_set)
        else:
            kept_glycans = [records.glycans[i] for i in kept_indices]
            kept_loci = records.loci[np.asarray(kept_indices, dtype=np.int32)]
            kept_archetypes = records.archetypes[kept_indices]
            kept_repeating = records.is_repeating[kept_indices]
            pan_prior, _ = PanGraphConsensusPrior.fit(
                kept_glycans,
                kept_loci,
                kept_archetypes,
                kept_repeating,
                assigner=assigner,
            )

        # 3. Precompute normalized 192-D 1D-CNN reference embeddings for sequence k-NN retrieval
        gt_seqs: list[bytes] = []
        gt_dl: list[int] = []
        gt_acc: list[int] = []
        gt_am: list[int] = []
        gt_bb: list[int] = []
        gt_recs: list[int] = []

        init_seqs: list[bytes] = []
        init_sugs: list[int] = []
        init_recs: list[int] = []

        wzy_seqs: list[bytes] = []
        wzy_lnks: list[int] = []
        wzy_recs: list[int] = []

        multi_edge_clusters: dict[int, list[tuple[ResidueType, LinkageType]]] = {}
        multi_edge_cand: dict[int, list[tuple[ResidueType, LinkageType]]] = {}
        multi_edge_counts: dict[int, int] = {}
        single_edge_counts: dict[int, int] = {}
        locus_fps: dict[tuple[int, ...], list[tuple[int, Glycan, tuple[GeneGlycanAssignment, ...]]]] = {}
        ref_orf_seqs: list[bytes] = []
        ref_orf_cids: list[int] = []
        ref_orf_roles: list[int] = []
        orf_kmer_index: dict[bytes, list[int]] = {}

        for i in kept_indices:
            loc = records.loci[i]
            gly = records.glycans[i]
            if len(loc.nodes) == 0 or len(gly.nodes) == 0:
                continue
            if len(loc.genes) > 0:
                trans_r = loc.genes.translations
                for g_i_r, nd_r in enumerate(loc.nodes):
                    if nd_r.cluster_id > 0 and g_i_r < len(trans_r):
                        s_b = bytes(trans_r[g_i_r].seq)
                        if len(s_b) >= 60:
                            idx_ref = len(ref_orf_seqs)
                            ref_orf_seqs.append(s_b)
                            ref_orf_cids.append(int(nd_r.cluster_id))
                            ref_orf_roles.append(int(nd_r.role))
                            seen_k: set[bytes] = set()
                            for pos_k in range(0, len(s_b) - 3, 2):
                                km = s_b[pos_k : pos_k + 4]
                                if km not in seen_k:
                                    seen_k.add(km)
                                    orf_kmer_index.setdefault(km, []).append(idx_ref)
            assigns = records.assignments[i]
            fp_cids = tuple(
                int(n.cluster_id)
                for n in loc.nodes
                if n.cluster_id > 0
                and n.role
                in (
                    GeneRole.INITIATOR,
                    GeneRole.GLYCOSYLTRANSFERASE,
                    GeneRole.POLYMERASE,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                )
            )
            if len(fp_cids) >= 2:
                locus_fps.setdefault(fp_cids, []).append((i, gly, assigns))

            # Track processive/bifunctional GTs assigned to >= 2 glycosidic edges in the same record
            by_gene_edges: dict[int, list[tuple[int, ResidueType, LinkageType]]] = {}
            _, assembly_depths, _ = gly.polymer_edge_and_assembly_depths(gly.reducing_end_index)
            for a in assigns:
                if (
                    a.is_aligned
                    and a.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.TRANSMEMBRANE_SYNTHASE)
                    and a.edge_index is not None
                    and 0 <= a.edge_index < len(gly.edges)
                    and 0 <= a.gene_index < len(loc.nodes)
                ):
                    e_m = gly.edges[a.edge_index]
                    s_m = gly.nodes[e_m.source]
                    if not s_m.is_modifier and e_m.edge_type.anomer != Anomer.UNKNOWN and e_m.edge_type.acceptor > 0:
                        by_gene_edges.setdefault(a.gene_index, []).append((int(a.edge_index), s_m, e_m.edge_type))
            for g_i_m, edges_m in by_gene_edges.items():
                cid_m = int(loc.nodes[g_i_m].cluster_id)
                if cid_m > 0:
                    if len(edges_m) >= 2:
                        edges_m.sort(key=lambda item: (assembly_depths.get(gly.edges[item[0]].source, 99), item[0]))
                        multi_edge_counts[cid_m] = multi_edge_counts.get(cid_m, 0) + 1
                        if cid_m not in multi_edge_cand:
                            multi_edge_cand[cid_m] = [(item[1], item[2]) for item in edges_m]
                    else:
                        single_edge_counts[cid_m] = single_edge_counts.get(cid_m, 0) + 1

            if len(loc.genes) == 0:
                continue
            translations = loc.genes.translations
            for a in assigns:
                if not a.is_aligned or a.gene_index < 0 or a.gene_index >= len(translations):
                    continue
                if a.score < 0.62 and a.evidence != AssignmentEvidence.PAN_GRAPH_CONSENSUS:
                    continue
                aa = bytes(translations[a.gene_index].seq)
                if len(aa) < 70:
                    continue
                if a.role == GeneRole.INITIATOR and a.residue is not None and not a.residue.is_modifier:
                    init_seqs.append(aa)
                    init_sugs.append(encode_donor_residue_chem(a.residue))
                    init_recs.append(i)
                elif a.role == GeneRole.POLYMERASE and a.edge_index is not None and 0 <= a.edge_index < len(gly.edges):
                    e_w = gly.edges[a.edge_index]
                    if e_w.edge_type.anomer != Anomer.UNKNOWN and e_w.edge_type.acceptor > 0:
                        wzy_seqs.append(aa)
                        wzy_lnks.append(int(e_w.edge_type.chemical_int))
                        wzy_recs.append(i)
                elif (
                    a.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.TRANSMEMBRANE_SYNTHASE)
                    and a.edge_index is not None
                    and 0 <= a.edge_index < len(gly.edges)
                ):
                    e_g = gly.edges[a.edge_index]
                    s_n = gly.nodes[e_g.source]
                    t_n = gly.nodes[e_g.target]
                    if not s_n.is_modifier and not t_n.is_modifier:
                        if e_g.edge_type.anomer != Anomer.UNKNOWN and e_g.edge_type.acceptor > 0:
                            # Use N/C domain split when a bifunctional GT is assigned to 2+ edges
                            aa_use = aa
                            g_edges = by_gene_edges.get(a.gene_index, [])
                            if len(g_edges) >= 2:
                                doms = split_bifunctional_gt_domains(aa, min_len=550, overlap=25)
                                if doms is not None:
                                    e_idx_list = [item[0] for item in g_edges]
                                    if int(a.edge_index) in e_idx_list:
                                        pos_d = e_idx_list.index(int(a.edge_index))
                                        aa_use = doms[0] if pos_d == 0 else doms[1]
                            gt_seqs.append(aa_use)
                            gt_dl.append(encode_donor_linkage_label(s_n, e_g.edge_type))
                            gt_acc.append(encode_donor_residue_chem(t_n))
                            gt_am.append(extract_acceptor_motif_for_edge(gly, int(a.edge_index)))
                            gt_bb.append(1 if e_g.edge_type.is_backbone else 0)
                            gt_recs.append(i)

        for cid_m, cand_m in multi_edge_cand.items():
            m_cnt = multi_edge_counts.get(cid_m, 0)
            s_cnt = single_edge_counts.get(cid_m, 0)
            if m_cnt >= 2 or (m_cnt >= 1 and s_cnt == 0):
                multi_edge_clusters[cid_m] = cand_m

        backbone_clf = (
            suite.classifiers.get("donor_linkage")
            if (suite is not None and "donor_linkage" in suite.classifiers)
            else None
        )
        if backbone_clf is None:
            try:
                backbone_clf = GeneRoleClassifier.pretrained()
            except Exception:
                backbone_clf = None

        def _embed_norm(seq_list: list[bytes]) -> np.ndarray | None:
            if not seq_list or backbone_clf is None or not backbone_clf.is_fitted:
                return None
            emb = backbone_clf.transform(Sequences.from_bytes(seq_list)).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            return emb / norms

        predictor = cls(
            pan_prior=pan_prior,
            classifier_suite=suite,
            assigner=assigner,
            disaccharide_counts=disac_counts,
            donor_acc_counts=donor_acc_counts,
            wzy_closure_counts=wzy_closure_counts,
            archetype_init_counts=arch_init_counts,
            donor_vocab=donor_vocab,
            bigram_trans_matrix=bigram_trans,
            ref_gt_embeddings=_embed_norm(gt_seqs),
            ref_gt_dl_labels=np.asarray(gt_dl, dtype=np.int64) if gt_dl else None,
            ref_gt_acc_labels=np.asarray(gt_acc, dtype=np.int64) if gt_acc else None,
            ref_gt_am_labels=np.asarray(gt_am, dtype=np.int64) if gt_am else None,
            ref_gt_bb_labels=np.asarray(gt_bb, dtype=np.int64) if gt_bb else None,
            ref_gt_record_indices=np.asarray(gt_recs, dtype=np.int64) if gt_recs else None,
            ref_init_embeddings=_embed_norm(init_seqs),
            ref_init_sugars=np.asarray(init_sugs, dtype=np.int64) if init_sugs else None,
            ref_init_record_indices=np.asarray(init_recs, dtype=np.int64) if init_recs else None,
            ref_wzy_embeddings=_embed_norm(wzy_seqs),
            ref_wzy_links=np.asarray(wzy_lnks, dtype=np.int64) if wzy_lnks else None,
            ref_wzy_record_indices=np.asarray(wzy_recs, dtype=np.int64) if wzy_lnks else None,
            multi_edge_clusters=multi_edge_clusters,
            locus_fingerprints=locus_fps,
            cluster_am_counts=cluster_am_counts,
            ref_orf_seqs=tuple(ref_orf_seqs),
            ref_orf_cids=np.asarray(ref_orf_cids, dtype=np.int32) if ref_orf_cids else None,
            ref_orf_roles=np.asarray(ref_orf_roles, dtype=np.int32) if ref_orf_roles else None,
            orf_kmer_index=orf_kmer_index,
        )

        # 4. Load packaged RelationalMPNN (when no CV exclusions) or train RelationalMPNN on kept_indices
        if not excluded and not retrain_mpnn and mpnn_path is not None and mpnn_path.is_file():
            try:
                predictor.load_mpnn(mpnn_path)
            except Exception:
                predictor.fit_mpnn(records, kept_indices=kept_indices, epochs=mpnn_epochs)
        elif mpnn_epochs > 0:
            predictor.fit_mpnn(records, kept_indices=kept_indices, epochs=mpnn_epochs)
            if not excluded and mpnn_path is not None and predictor.mpnn is not None:
                try:
                    predictor.save_mpnn(mpnn_path)
                except Exception:
                    pass

        return predictor

    def resolve_locus_homologs(self, locus: Locus) -> Locus:
        """Map raw unclustered `LocusPredictor` ORFs to Pan-Locus `(GeneRole, cluster_id)` via 4-mer indexed `PairwiseAligner`."""
        if (
            len(locus.nodes) == 0
            or len(locus.genes) == 0
            or not self.ref_orf_seqs
            or self.ref_orf_cids is None
            or self.ref_orf_roles is None
        ):
            return locus

        if self._pairwise_aligner is None:
            self._pairwise_aligner = PairwiseAligner(k=35)

        trans = locus.genes.translations
        new_nodes = list(locus.nodes)
        n_ref = len(self.ref_orf_seqs)
        counts = np.zeros(n_ref, dtype=np.int32)

        for g_i, nd in enumerate(locus.nodes):
            if g_i >= len(trans):
                continue
            aa = bytes(trans[g_i].seq)
            if len(aa) < 60:
                continue
            counts.fill(0)
            for pos in range(len(aa) - 3):
                km = aa[pos : pos + 4]
                hit_list = self._orf_kmer_index.get(km)
                if hit_list is not None:
                    counts[hit_list] += 1
            top_cand = np.argsort(counts)[-48:][::-1]
            top_cand = top_cand[counts[top_cand] > 0]
            if len(top_cand) == 0:
                continue
            q_b = Sequences.from_bytes([aa] * len(top_cand))
            t_b = Sequences.from_bytes([self.ref_orf_seqs[int(i)] for i in top_cand])
            alns = self._pairwise_aligner(q_b, t_b)
            bits = alns.bitscores
            pids = alns.pidents
            best_loc_idx = int(np.argmax(bits))
            best_ref_idx = int(top_cand[best_loc_idx])
            best_bit = float(bits[best_loc_idx])
            best_pid = float(pids[best_loc_idx])
            best_cid = int(self.ref_orf_cids[best_ref_idx])
            best_role = GeneRole(int(self.ref_orf_roles[best_ref_idx]))

            cand_roles = self.ref_orf_roles[top_cand]
            same_mask = (cand_roles == int(nd.role)) & (bits >= 95.0) & (pids >= 28.0)

            if best_bit >= 150.0 and best_pid >= 34.0:
                new_nodes[g_i] = GeneNode(best_role, best_cid)
            elif np.any(same_mask):
                idx_s = int(np.argmax(np.where(same_mask, bits, -1.0)))
                ref_s = int(top_cand[idx_s])
                new_nodes[g_i] = GeneNode(nd.role, int(self.ref_orf_cids[ref_s]))

        has_gla = any(n.cluster_id == 240 for n in new_nodes)
        for g_i, nd in enumerate(new_nodes):
            if nd.role == GeneRole.DEACETYLASE and nd.cluster_id == 0 and g_i < len(trans):
                aa_len = len(trans[g_i].seq)
                prev_role = new_nodes[g_i - 1].role if g_i > 0 else GeneRole.UNKNOWN
                if 290 <= aa_len <= 390 and (
                    has_gla or prev_role in (GeneRole.FLIPPASE, GeneRole.POLYMERASE)
                ):
                    new_nodes[g_i] = GeneNode(GeneRole.GLYCOSYLTRANSFERASE, 0)

        return replace(locus, nodes=tuple(new_nodes))

    def _compute_operon_precursor_pool(
        self,
        locus: Locus,
    ) -> tuple[dict[bytes, float], set[int], np.ndarray]:
        """Compute the Stage 2 Neuro-Symbolic Operon Precursor Pool (`c_pool`) for `locus`.

        Returns:
            `(pool_by_name, synthesized_specialized_sugars, donor_vocab_gate_multiplier)`.
        """
        pool_probs: dict[int, float] = {}
        for hk_s in HOUSEKEEPING_SUGARS:
            pool_probs[int(hk_s)] = 1.0

        synthesized_special: set[int] = set()
        present_roles = {n.role for n in locus.nodes}
        present_cids = {int(n.cluster_id) for n in locus.nodes if n.cluster_id > 0}
        has_rml_pathway = bool(present_cids & {230, 231, 323})
        role_pathway_rules: tuple[tuple[frozenset[GeneRole], tuple[bytes, ...]], ...] = (
            (
                frozenset({GeneRole.EPIMERASE, GeneRole.OXIDOREDUCTASE, GeneRole.NUCLEOTIDYLTRANSFERASE}),
                (b"Rha",) if has_rml_pathway else (b"Rha", b"Fuc", b"6dTal"),
            ),
            (
                frozenset({GeneRole.EPIMERASE, GeneRole.OXIDOREDUCTASE}),
                (b"Rha",) if has_rml_pathway else (b"Rha", b"Fuc"),
            ),
            (frozenset({GeneRole.AMINOTRANSFERASE}), (b"QuiNAc", b"FucNAc", b"Bac", b"Pse", b"Leg")),
            (frozenset({GeneRole.ALDOLASE}), (b"Neu5Ac", b"Pse", b"Leg", b"Kdo")),
        )
        for req_roles, sug_names in role_pathway_rules:
            if req_roles.issubset(present_roles):
                for s_nm in sug_names:
                    if s_nm in Sugars.name_to_index:
                        s_i = int(Sugars.name_to_index[s_nm])
                        pool_probs[s_i] = 1.0
                        if s_i not in HOUSEKEEPING_SUGARS:
                            synthesized_special.add(s_i)

        if 240 in present_cids and b"GalA" in Sugars.name_to_index:
            gala_i = int(Sugars.name_to_index[b"GalA"])
            pool_probs[gala_i] = 1.0
            synthesized_special.add(gala_i)

        all_sugar_ids = set(range(len(Sugars.names)))
        for n in locus.nodes:
            if n.cluster_id > 0:
                pref_s = self.pan_prior.preferred_precursor_sugar(int(n.cluster_id), all_sugar_ids)
                if pref_s is not None:
                    pool_probs[int(pref_s)] = max(pool_probs.get(int(pref_s), 0.0), 0.95)
                    if int(pref_s) not in HOUSEKEEPING_SUGARS:
                        synthesized_special.add(int(pref_s))

        gate = np.ones(len(self.donor_vocab), dtype=np.float64)
        for idx_d, d_chem in enumerate(self.donor_vocab):
            s_id = int(ResidueType(d_chem).sugar)
            if s_id in synthesized_special:
                gate[idx_d] = 1.45
            elif s_id in HOUSEKEEPING_SUGARS:
                gate[idx_d] = 1.0
            else:
                gate[idx_d] = 0.75

        named_pool = {
            Sugars.names[s_id]: round(float(p_val), 3)
            for s_id, p_val in sorted(pool_probs.items())
            if 0 <= s_id < len(Sugars.names)
        }
        return named_pool, synthesized_special, gate

    def _embed_single_seq(self, aa_seq: bytes) -> np.ndarray | None:
        """Compute L2-normalized 1D-CNN embedding vector for a single ORF protein sequence."""
        if len(aa_seq) < 60 or self.classifier_suite is None:
            return None
        cached = self._embed_cache.get(aa_seq, False)
        if cached is not False:
            return cached
        clf = self.classifier_suite.classifiers.get("donor_linkage")
        if clf is None or not clf.is_fitted:
            self._embed_cache[aa_seq] = None
            return None
        vec = clf.transform(Sequences.from_bytes([aa_seq]))[0].astype(np.float32)
        nrm = float(np.linalg.norm(vec))
        res = vec / max(nrm, 1e-8)
        self._embed_cache[aa_seq] = res
        return res

    def _conv_max_single_seq(self, aa_seq: bytes, *, is_initiator: bool = False) -> np.ndarray:
        """Compute 64-D L2-normalized `task_weighted_conv_max` CNN vector for a single ORF sequence."""
        key = (aa_seq, bool(is_initiator))
        if key in self._conv_max_cache:
            return self._conv_max_cache[key]
        out = np.zeros(64, dtype=np.float32)
        if len(aa_seq) < 60 or self.classifier_suite is None:
            self._conv_max_cache[key] = out
            return out
        target_name = "initiator_sugar" if is_initiator else "donor_linkage"
        clf = self.classifier_suite.classifiers.get(target_name) or self.classifier_suite.classifiers.get(
            "donor_linkage"
        )
        if clf is None or not clf.is_fitted:
            self._conv_max_cache[key] = out
            return out
        try:
            vec = clf.transform(Sequences.from_bytes([aa_seq]), layer="task_weighted_conv_max")[0].astype(np.float32)
            d = min(len(vec), 64)
            nrm = float(np.linalg.norm(vec[:d]))
            out[:d] = vec[:d] / max(nrm, 1e-8)
        except Exception:
            pass
        self._conv_max_cache[key] = out
        return out

    def _clf_proba_single(self, target_name: str, aa_seq: bytes) -> np.ndarray | None:
        """Compute (and memoize) 1D-CNN class probabilities for a single ORF sequence."""
        if len(aa_seq) < 60 or self.classifier_suite is None:
            return None
        key = (target_name, aa_seq)
        cached = self._clf_proba_cache.get(key, False)
        if cached is not False:
            return cached
        clf = self.classifier_suite.classifiers.get(target_name)
        if clf is None or not clf.is_fitted:
            self._clf_proba_cache[key] = None
            return None
        probs = clf.predict_proba([aa_seq])[0]
        self._clf_proba_cache[key] = probs
        return probs

    def _build_mpnn_graph_for_locus(
        self,
        slot_seqs: list[bytes],
        synthesized_special: set[int],
        p_bb_init: np.ndarray,
        clamped_mask_slots: np.ndarray,
    ) -> tuple[np.ndarray, RelationalCSR, np.ndarray]:
        """Construct `(x_nodes, csr_graph, clamped_full)` for `serogen.core.graph.gnn.RelationalMPNN` via `build_multihop_csr`.

        Args:
            slot_seqs: `[init_aa_seq, gt_1_seq, ..., gt_k_seq]` of length `M = 1 + n_gt_slots`.
            synthesized_special: Set of specialized precursor `Sugars` indices synthesized in the operon.
            p_bb_init: `(M,)` initial backbone probability per node (`p_bb_init[0] = 1.0`).
            clamped_mask_slots: `(M,)` uint8 clamping mask (`1 = clamped`, `0 = free`).

        Returns:
            `(x_nodes, csr_graph, clamped_full)` where `x_nodes` has shape `(M + 1, 80)` and node `M` is the
            virtual Operon Precursor Pool node (`add_virtual_pool=True`).
        """
        m_nodes = len(slot_seqs)
        x_nodes = np.zeros((m_nodes + 1, 80), dtype=np.float32)

        pool_bits = np.zeros(10, dtype=np.float32)
        for idx_p, p_name in enumerate(PRECURSOR_POOL_SUGAR_NAMES):
            s_id = Sugars.name_to_index.get(p_name, -1)
            if s_id >= 0 and (s_id in synthesized_special or s_id in HOUSEKEEPING_SUGARS):
                pool_bits[idx_p] = 1.0 if s_id in synthesized_special else 0.35

        for u in range(m_nodes):
            aa = slot_seqs[u]
            x_nodes[u, :64] = self._conv_max_single_seq(aa, is_initiator=(u == 0))
            x_nodes[u, 64] = 1.0 if u == 0 else 0.0
            x_nodes[u, 65] = 0.0
            x_nodes[u, 66] = float(u) / float(max(1, m_nodes - 1))
            x_nodes[u, 67] = min(float(len(aa)) / 500.0, 2.0)
            x_nodes[u, 68] = float(p_bb_init[u]) if u < len(p_bb_init) else 0.80
            x_nodes[u, 69] = float(clamped_mask_slots[u]) if u < len(clamped_mask_slots) else 0.0
            x_nodes[u, 70:80] = pool_bits

        # Virtual Operon Precursor Pool node at index m_nodes
        x_nodes[m_nodes, 65] = 1.0
        x_nodes[m_nodes, 70:80] = pool_bits

        if m_nodes >= 2:
            edge_src = np.arange(0, m_nodes - 1, dtype=np.int32)
            edge_dst = np.arange(1, m_nodes, dtype=np.int32)
            edge_orient = np.zeros(m_nodes - 1, dtype=np.int32)
        else:
            edge_src = np.empty(0, dtype=np.int32)
            edge_dst = np.empty(0, dtype=np.int32)
            edge_orient = np.empty(0, dtype=np.int32)

        csr_graph = build_multihop_csr(
            np.array([0, m_nodes], dtype=np.int32),
            edge_src,
            edge_dst,
            edge_orient,
            include_2hop=self.mpnn_include_2hop,
            add_virtual_pool=True,
        )
        clamped_full = np.zeros(m_nodes + 1, dtype=bool)
        for u in range(min(m_nodes, len(clamped_mask_slots))):
            clamped_full[u] = bool(clamped_mask_slots[u] != 0)
        return x_nodes, csr_graph, clamped_full

    def fit_mpnn(
        self,
        records: GlycanRecords,
        *,
        kept_indices: Sequence[int] | None = None,
        epochs: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 3,
        bilinear_rank: int = 32,
        learning_rate: float = 0.004,
        weight_decay: float = 1e-4,
        clamped_prob: float = 0.25,
        include_2hop: bool = True,
        random_state: int = 42,
    ) -> RelationalMPNN | None:
        """Train `serogen.core.graph.gnn.RelationalMPNN` on aligned `(Locus, Glycan)` arborescences."""
        self.mpnn_include_2hop = bool(include_2hop)
        indices = list(kept_indices) if kept_indices is not None else list(range(len(records)))
        dl_set: set[int] = set()
        if self.ref_gt_dl_labels is not None and len(self.ref_gt_dl_labels) > 0:
            dl_set.update(int(x) for x in np.unique(self.ref_gt_dl_labels))
        if (
            self.classifier_suite is not None
            and "donor_linkage" in self.classifier_suite.classifiers
            and self.classifier_suite["donor_linkage"].is_fitted
        ):
            dl_set.update(int(x) for x in self.classifier_suite["donor_linkage"].classes_)
        for i in indices:
            gly = records.glycans[i]
            for a in records.assignments[i]:
                if (
                    a.is_aligned
                    and a.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.TRANSMEMBRANE_SYNTHASE)
                    and a.edge_index is not None
                    and 0 <= a.edge_index < len(gly.edges)
                ):
                    e_a = gly.edges[int(a.edge_index)]
                    s_a = gly.nodes[e_a.source]
                    if not s_a.is_modifier and e_a.edge_type.anomer != Anomer.UNKNOWN and e_a.edge_type.acceptor > 0:
                        dl_set.add(encode_donor_linkage_label(s_a, e_a.edge_type))
        if len(dl_set) < 2:
            return None

        dl_classes = np.array(sorted(dl_set), dtype=np.int64)
        dl_to_idx = {int(c): idx for idx, c in enumerate(dl_classes)}

        train_graphs: list[tuple[np.ndarray, RelationalCSR, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for i in indices:
            loc = records.loci[i]
            gly = records.glycans[i]
            if len(loc.nodes) == 0 or len(gly.nodes) < 2:
                continue
            assigns = records.assignments[i]
            init_g_idx = -1
            for g_i, g_n in enumerate(loc.nodes):
                if g_n.role == GeneRole.INITIATOR:
                    init_g_idx = g_i
                    break
            gt_g_indices = [
                g_i
                for g_i, g_n in enumerate(loc.nodes)
                if g_n.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.TRANSMEMBRANE_SYNTHASE)
            ]
            if not gt_g_indices:
                continue

            by_gene_edge: dict[int, GeneGlycanAssignment] = {}
            for a in assigns:
                if (
                    a.is_aligned
                    and a.gene_index in gt_g_indices
                    and a.edge_index is not None
                    and 0 <= a.edge_index < len(gly.edges)
                ):
                    by_gene_edge[int(a.gene_index)] = a
            if len(by_gene_edge) < len(gt_g_indices):
                continue
            core_gly = gly.without_modifiers()
            if len(core_gly.nodes) != 1 + len(gt_g_indices):
                continue
            gt_sources = {int(gly.edges[int(a.edge_index)].source) for a in by_gene_edge.values()}
            if len(gt_sources) != len(gt_g_indices) or gly.reducing_end_index in gt_sources:
                continue

            has_genes = len(loc.genes) > 0
            translations = loc.genes.translations if has_genes else ()
            init_aa = (
                bytes(translations[init_g_idx].seq)
                if (has_genes and 0 <= init_g_idx < len(translations))
                else b""
            )
            slot_seqs: list[bytes] = [init_aa]
            for g_i in gt_g_indices:
                aa_g = bytes(translations[g_i].seq) if (has_genes and g_i < len(translations)) else b""
                slot_seqs.append(aa_g)

            m_nodes = len(slot_seqs)
            node_to_slot: dict[int, int] = {gly.reducing_end_index: 0}
            node_targets = np.full(m_nodes + 1, -1, dtype=np.int32)
            node_weights = np.ones(m_nodes + 1, dtype=np.float32)
            p_bb_init = np.ones(m_nodes, dtype=np.float32)

            for s_idx, g_i in enumerate(gt_g_indices, start=1):
                a_g = by_gene_edge[g_i]
                e_g = gly.edges[int(a_g.edge_index)]
                node_to_slot[int(e_g.source)] = s_idx
                p_bb_init[s_idx] = 1.0 if e_g.edge_type.is_backbone else 0.1
                s_res = gly.nodes[e_g.source]
                if not s_res.is_modifier and e_g.edge_type.anomer != Anomer.UNKNOWN and e_g.edge_type.acceptor > 0:
                    dl_code = encode_donor_linkage_label(s_res, e_g.edge_type)
                    if dl_code in dl_to_idx:
                        node_targets[s_idx] = dl_to_idx[dl_code]
                w_ev = (
                    1.50
                    if (a_g.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS and a_g.score >= 0.93)
                    else (1.25 if a_g.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS else 0.85)
                )
                node_weights[s_idx] = float(w_ev * float(a_g.score))

            parent_targets = np.full(m_nodes, -1, dtype=np.int32)
            slot_targets = np.zeros(m_nodes, dtype=np.int32)
            valid_edges = 0
            for s_idx, g_i in enumerate(gt_g_indices, start=1):
                a_g = by_gene_edge[g_i]
                e_g = gly.edges[int(a_g.edge_index)]
                t_slot = node_to_slot.get(int(e_g.target), -1)
                if 0 <= t_slot < m_nodes and t_slot != s_idx:
                    parent_targets[s_idx] = t_slot
                    slot_targets[s_idx] = min(max(int(e_g.edge_type.acceptor), 0), 7)
                    valid_edges += 1
            if valid_edges == 0:
                continue

            _, synth_spec, _ = self._compute_operon_precursor_pool(loc)
            x_nodes, csr_graph, _ = self._build_mpnn_graph_for_locus(
                slot_seqs,
                synth_spec,
                p_bb_init,
                np.zeros(m_nodes, dtype=np.uint8),
            )
            train_graphs.append((x_nodes, csr_graph, node_targets, node_weights, parent_targets, slot_targets))

        if not train_graphs:
            return None

        mpnn = RelationalMPNN(
            in_dim=80,
            hidden_dim=hidden_dim,
            num_node_classes=len(dl_classes),
            num_edge_channels=8,
            num_relations=8,
            num_layers=num_layers,
            bilinear_rank=bilinear_rank,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
        )
        rng = np.random.default_rng(random_state)
        n_g = len(train_graphs)
        order = np.arange(n_g, dtype=np.int32)
        for ep in range(max(1, epochs)):
            rng.shuffle(order)
            lr_s = float(0.5 * (1.0 + np.cos(np.pi * ep / max(1, epochs))))
            for idx_g in order:
                x_n, csr_g, y_n, w_n, p_t, s_t = train_graphs[int(idx_g)]
                m_sub = len(p_t)
                clamp_m: np.ndarray | None = None
                if m_sub >= 3 and clamped_prob > 0.0 and float(rng.random()) < clamped_prob:
                    clamp_m = np.zeros(len(x_n), dtype=bool)
                    free_u = int(rng.integers(1, m_sub))
                    clamp_m[:m_sub] = True
                    clamp_m[free_u] = False
                mpnn.fit_step(
                    x_n,
                    csr_g,
                    node_targets=y_n,
                    node_weights=w_n,
                    edge_node_indices=np.arange(m_sub, dtype=np.int32),
                    edge_parent_targets=p_t,
                    edge_slot_targets=s_t,
                    clamped_mask=clamp_m,
                    lr_scale=max(0.15, lr_s),
                )

        self.mpnn = mpnn
        self.mpnn_dl_classes = dl_classes
        return mpnn

    def save_mpnn(self, path: str | Path = DEFAULT_MPNN_PATH) -> None:
        """Save the fitted `RelationalMPNN`, `mpnn_dl_classes`, and `GlycanPredictorParams` to a zero-pickle `.npz` archive."""
        if self.mpnn is None or self.mpnn_dl_classes is None:
            return
        out_path = Path(path)
        self.mpnn.save(out_path)
        with np.load(out_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["mpnn_dl_classes"] = np.asarray(self.mpnn_dl_classes, dtype=np.int64)
        arrays["arborescence_params"] = self.params.to_array()
        arrays["mpnn_include_2hop"] = np.array([1 if self.mpnn_include_2hop else 0], dtype=np.int8)
        np.savez_compressed(out_path, **arrays)

    def load_mpnn(self, path: str | Path = DEFAULT_MPNN_PATH) -> RelationalMPNN:
        """Load a fitted `RelationalMPNN`, `mpnn_dl_classes`, and `GlycanPredictorParams` from a zero-pickle `.npz` archive."""
        in_path = Path(path)
        mpnn = RelationalMPNN.load(in_path)
        with np.load(in_path, allow_pickle=False) as data:
            dl_classes = data["mpnn_dl_classes"].astype(np.int64) if "mpnn_dl_classes" in data else None
            if "arborescence_params" in data:
                self.params = GlycanPredictorParams.from_array(data["arborescence_params"])
            if "mpnn_include_2hop" in data:
                self.mpnn_include_2hop = bool(int(data["mpnn_include_2hop"][0]) != 0)
        self.mpnn = mpnn
        self.mpnn_dl_classes = dl_classes
        return mpnn

    def _predict_root_candidates(
        self,
        locus: Locus,
        archetype: LocusArchetype,
        clamped_by_gene: dict[int, GeneGlycanAssignment],
    ) -> tuple[int, list[tuple[ResidueType, float, AssignmentEvidence]]]:
        """Stage 1/3: Predict `(init_gene_idx, [(root_residue, prob, evidence), ...])`."""
        init_gene_idx = -1
        for idx_g, g_node in enumerate(locus.nodes):
            if g_node.role == GeneRole.INITIATOR:
                init_gene_idx = idx_g
                break

        # 1. Check hard clamping on the initiator gene
        if init_gene_idx in clamped_by_gene:
            c_assign = clamped_by_gene[init_gene_idx]
            if c_assign.residue is not None and not c_assign.residue.is_modifier:
                r_res = c_assign.residue.with_reducing_end(True).with_backbone(True)
                return init_gene_idx, [(r_res, 0.99, AssignmentEvidence.PAN_GRAPH_CONSENSUS)]

        scores: dict[int, float] = {}
        ev_map: dict[int, AssignmentEvidence] = {}

        if init_gene_idx >= 0:
            g_node = locus.nodes[init_gene_idx]
            cid = int(g_node.cluster_id)
            for c_tier, tier_scale in ((cid, 1.0), (self.pan_prior.fine_to_med.get(cid, 0), 0.72)):
                if c_tier > 0 and c_tier in self.pan_prior.init_cluster_sugar:
                    s_counts = self.pan_prior.init_cluster_sugar[c_tier]
                    tot = sum(s_counts.values())
                    for s_id, cnt in s_counts.items():
                        nat_chi = Chirality(Sugars.natural_chirality[s_id])
                        r_chem = encode_donor_residue_chem(ResidueType.from_sugar(s_id, nat_chi, False))
                        p = cnt / max(1e-6, tot)
                        scores[r_chem] = scores.get(r_chem, 0.0) + 2.5 * tier_scale * p
                        if cnt >= 1.5 and p >= 0.55:
                            ev_map[r_chem] = AssignmentEvidence.PAN_GRAPH_CONSENSUS

            aa_seq = (
                bytes(locus.genes.translations[init_gene_idx].seq)
                if (len(locus.genes) > 0 and init_gene_idx < len(locus.genes.translations))
                else b""
            )
            if len(aa_seq) >= 60 and self.classifier_suite is not None and "initiator_sugar" in self.classifier_suite.classifiers:
                init_clf = self.classifier_suite["initiator_sugar"]
                probs = self._clf_proba_single("initiator_sugar", aa_seq)
                if probs is not None:
                    for c_i, s_id in enumerate(init_clf.classes_):
                        p_val = float(probs[c_i])
                        if p_val >= 0.05:
                            nat_chi = Chirality(Sugars.natural_chirality[int(s_id)])
                            r_chem = encode_donor_residue_chem(ResidueType.from_sugar(int(s_id), nat_chi, False))
                            scores[r_chem] = scores.get(r_chem, 0.0) + 1.2 * p_val
                            if p_val >= 0.60:
                                ev_map.setdefault(r_chem, AssignmentEvidence.PAN_GRAPH_CONSENSUS)

            if len(aa_seq) >= 60 and self.ref_init_embeddings is not None and self.ref_init_sugars is not None:
                q_vec = self._embed_single_seq(aa_seq)
                if q_vec is not None:
                    sims = self.ref_init_embeddings @ q_vec
                    top_k = np.argsort(sims)[-5:][::-1]
                    for idx_nn in top_k:
                        sim_val = float(sims[idx_nn])
                        if sim_val >= 0.80:
                            r_chem = int(self.ref_init_sugars[idx_nn])
                            scores[r_chem] = scores.get(r_chem, 0.0) + 1.4 * (sim_val**2)
                            if sim_val >= 0.92:
                                ev_map.setdefault(r_chem, AssignmentEvidence.PAN_GRAPH_CONSENSUS)

        # Archetype fallback prior
        arch_int = int(archetype)
        arch_tot = sum(cnt for (a_k, _), cnt in self.archetype_init_counts.items() if a_k == arch_int)
        if arch_tot > 0:
            for (a_k, r_chem), cnt in self.archetype_init_counts.items():
                if a_k == arch_int:
                    scores[r_chem] = scores.get(r_chem, 0.0) + 0.35 * (cnt / arch_tot)

        if not scores:
            gal_idx = Sugars.name_to_index[b"Gal"]
            glc_idx = Sugars.name_to_index[b"Glc"]
            glcnac_idx = Sugars.name_to_index[b"GlcNAc"]
            for s_id, w_def in ((gal_idx, 0.45), (glc_idx, 0.35), (glcnac_idx, 0.20)):
                r_chem = encode_donor_residue_chem(ResidueType.from_sugar(s_id, Chirality.D, False))
                scores[r_chem] = w_def

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
        tot_s = sum(v for _, v in ranked) or 1.0
        out: list[tuple[ResidueType, float, AssignmentEvidence]] = []
        for r_chem, s_val in ranked:
            res = ResidueType(r_chem).with_reducing_end(True).with_backbone(True)
            ev = ev_map.get(r_chem, AssignmentEvidence.GRAPH_ALIGNED if init_gene_idx >= 0 else AssignmentEvidence.RULE_INFERRED)
            out.append((res, float(min(0.99, max(0.35, s_val / tot_s))), ev))
        return init_gene_idx, out

    def _build_gt_slots(
        self,
        locus: Locus,
        synthesized_special: set[int],
        precursor_gate: np.ndarray,
        clamped_by_gene: dict[int, GeneGlycanAssignment],
        reference_glycan: Glycan | None = None,
    ) -> list[_GTSlotSpec]:
        """Stage 1-3: Construct contextualized GT transfer step slots via SeroGen `RelationalMPNN`."""
        self._last_mpnn_edge_scores: np.ndarray | None = None
        gt_gene_indices: list[int] = []
        for idx_g, g_node in enumerate(locus.nodes):
            if g_node.role in (GeneRole.GLYCOSYLTRANSFERASE, GeneRole.TRANSMEMBRANE_SYNTHASE):
                gt_gene_indices.append(idx_g)

        if not gt_gene_indices:
            return []

        n_gts = len(gt_gene_indices)
        n_donors = len(self.donor_vocab)
        node_donor_probs = np.full((n_gts, max(1, n_donors)), 1.0 / max(1, n_donors), dtype=np.float64)
        node_acc_probs = np.full((n_gts, max(1, n_donors)), 1.0 / max(1, n_donors), dtype=np.float64)
        node_bb_probs = np.full(n_gts, 0.78, dtype=np.float64)
        clamped_mask = np.zeros(n_gts, dtype=np.uint8)

        per_gt_dl_scores: list[dict[int, float]] = [{} for _ in range(n_gts)]
        per_gt_am_scores: list[dict[int, float]] = [{} for _ in range(n_gts)]
        per_gt_ev: list[dict[int, AssignmentEvidence]] = [{} for _ in range(n_gts)]

        has_genes_loc = len(locus.genes) > 0
        translations_loc = locus.genes.translations if has_genes_loc else ()

        for slot_i, g_idx in enumerate(gt_gene_indices):
            g_node = locus.nodes[g_idx]
            cid = int(g_node.cluster_id)
            aa_seq = (
                bytes(translations_loc[g_idx].seq)
                if (has_genes_loc and g_idx < len(translations_loc))
                else b""
            )

            # Check hard/soft clamping
            if g_idx in clamped_by_gene:
                c_a = clamped_by_gene[g_idx]
                if (
                    reference_glycan is not None
                    and c_a.edge_index is not None
                    and 0 <= c_a.edge_index < len(reference_glycan.edges)
                ):
                    e_ref = reference_glycan.edges[c_a.edge_index]
                    s_ref = reference_glycan.nodes[e_ref.source]
                    t_ref = reference_glycan.nodes[e_ref.target]
                    dl_lbl = encode_donor_linkage_label(s_ref, e_ref.edge_type)
                    per_gt_dl_scores[slot_i][dl_lbl] = 10.0
                    per_gt_ev[slot_i][dl_lbl] = AssignmentEvidence.PAN_GRAPH_CONSENSUS
                    am_ref = extract_acceptor_motif_for_edge(reference_glycan, int(c_a.edge_index))
                    per_gt_am_scores[slot_i][am_ref] = 10.0
                    d_chem = encode_donor_residue_chem(s_ref)
                    a_chem = encode_donor_residue_chem(t_ref)
                    if d_chem in self.donor_to_idx:
                        node_donor_probs[slot_i, :] = 1e-4
                        node_donor_probs[slot_i, self.donor_to_idx[d_chem]] = 1.0
                    if a_chem in self.donor_to_idx:
                        node_acc_probs[slot_i, :] = 1e-4
                        node_acc_probs[slot_i, self.donor_to_idx[a_chem]] = 1.0
                    node_bb_probs[slot_i] = 0.98 if e_ref.edge_type.is_backbone else 0.02
                    clamped_mask[slot_i] = 1
                    continue

            d_dist: dict[int, float] = {}
            l_dist: dict[int, float] = {}
            acc_dist: dict[int, float] = {}
            fine_emp_bb: float | None = None

            # 1. Empirical PanGraphConsensusPrior (fine 28% + medium 20% superfamily clusters)
            for c_tier, tier_scale in ((cid, 1.0), (self.pan_prior.fine_to_med.get(cid, 0), 0.72)):
                if c_tier > 0 and c_tier in self.pan_prior.cluster_sig_full:
                    sig_counts = self.pan_prior.cluster_sig_full[c_tier]
                    tot_sig = sum(sig_counts.values())
                    dl_counts_tier = self.pan_prior.cluster_sig_dl.get(c_tier, {})
                    tot_dl_tier = sum(dl_counts_tier.values()) or tot_sig
                    for sig, cnt in sig_counts.items():
                        s_id, d_chi, is_fur, anom_val, don_val, acc_val, a_s_id, is_bb_sig = (
                            sig[0],
                            sig[1],
                            sig[2],
                            sig[3],
                            sig[4],
                            sig[5],
                            sig[6],
                            sig[7],
                        )
                        res_d = ResidueType.from_sugar(s_id, Chirality(d_chi), bool(is_fur))
                        lnk = LinkageType(Anomer(anom_val), don_val, acc_val)
                        if lnk.anomer == Anomer.UNKNOWN or lnk.acceptor == 0:
                            continue
                        dl_lbl = encode_donor_linkage_label(res_d, lnk)
                        p_emp = cnt / max(1e-6, tot_sig)
                        dl_sig_key = (s_id, d_chi, is_fur, anom_val, acc_val)
                        p_dl_emp = dl_counts_tier.get(dl_sig_key, cnt) / max(1e-6, tot_dl_tier)
                        unanimous_boost = (
                            3.00 if (c_tier == cid and p_dl_emp >= 0.80 and tot_dl_tier >= 1.8) else 1.0
                        )
                        per_gt_dl_scores[slot_i][dl_lbl] = (
                            per_gt_dl_scores[slot_i].get(dl_lbl, 0.0)
                            + self.params.w_pan_dl * tier_scale * unanimous_boost * p_emp
                        )
                        if cnt >= 1.5 and p_emp >= 0.40:
                            per_gt_ev[slot_i][dl_lbl] = AssignmentEvidence.PAN_GRAPH_CONSENSUS
                        d_c = encode_donor_residue_chem(res_d)
                        d_dist[d_c] = (
                            d_dist.get(d_c, 0.0)
                            + (0.78 * self.params.w_pan_dl) * tier_scale * unanimous_boost * p_emp
                        )
                        a_nat_chi = (
                            Chirality(Sugars.natural_chirality[a_s_id])
                            if 0 <= a_s_id < len(Sugars.names)
                            else Chirality.D
                        )
                        a_c = encode_donor_residue_chem(ResidueType.from_sugar(a_s_id, a_nat_chi, False))
                        acc_dist[a_c] = (
                            acc_dist.get(a_c, 0.0)
                            + (0.78 * self.params.w_pan_dl) * tier_scale * unanimous_boost * p_emp
                        )
                    emp_bb = sum(cnt * float(sig[7]) for sig, cnt in sig_counts.items()) / max(1e-6, tot_sig)
                    if c_tier == cid:
                        fine_emp_bb = emp_bb
                    node_bb_probs[slot_i] = 0.15 * node_bb_probs[slot_i] + 0.85 * emp_bb
                if c_tier > 0 and c_tier in self.cluster_am_counts:
                    am_counts_c = self.cluster_am_counts[c_tier]
                    tot_am_c = sum(am_counts_c.values())
                    for am_code, cnt_am in am_counts_c.items():
                        per_gt_am_scores[slot_i][am_code] = (
                            per_gt_am_scores[slot_i].get(am_code, 0.0)
                            + (1.09 * self.params.w_pan_dl) * tier_scale * (cnt_am / max(1e-6, tot_am_c))
                        )

            # 2. 192-D 1D-CNN Sequence Embedding k-NN Retrieval (works even when cluster_id == 0 or held-out!)
            if len(aa_seq) >= 70 and self.ref_gt_embeddings is not None and self.ref_gt_dl_labels is not None:
                q_vec = self._embed_single_seq(aa_seq)
                if q_vec is not None:
                    sims = self.ref_gt_embeddings @ q_vec
                    top_k = np.argsort(sims)[-7:][::-1]
                    for idx_nn in top_k:
                        sim_val = float(sims[idx_nn])
                        if sim_val < 0.78:
                            continue
                        w_nn = self.params.w_knn_gt * (sim_val**3)
                        dl_nn = int(self.ref_gt_dl_labels[idx_nn])
                        per_gt_dl_scores[slot_i][dl_nn] = per_gt_dl_scores[slot_i].get(dl_nn, 0.0) + w_nn
                        if sim_val >= 0.90:
                            per_gt_ev[slot_i].setdefault(dl_nn, AssignmentEvidence.PAN_GRAPH_CONSENSUS)
                        res_nn, _ = decode_donor_linkage_label(dl_nn)
                        d_c = encode_donor_residue_chem(res_nn)
                        d_dist[d_c] = d_dist.get(d_c, 0.0) + w_nn
                        if self.ref_gt_acc_labels is not None:
                            a_c = int(self.ref_gt_acc_labels[idx_nn])
                            acc_dist[a_c] = acc_dist.get(a_c, 0.0) + w_nn
                        if self.ref_gt_am_labels is not None:
                            am_c = int(self.ref_gt_am_labels[idx_nn])
                            per_gt_am_scores[slot_i][am_c] = per_gt_am_scores[slot_i].get(am_c, 0.0) + w_nn
                        if self.ref_gt_bb_labels is not None:
                            node_bb_probs[slot_i] = 0.75 * node_bb_probs[slot_i] + 0.25 * float(
                                self.ref_gt_bb_labels[idx_nn]
                            )

            # 3. 10-Head BiosyntheticClassifierSuite Neural Potentials
            if len(aa_seq) >= 60 and self.classifier_suite is not None and self.classifier_suite.is_fitted:
                for t_name, target_dict, weight_t in (
                    ("donor_linkage", per_gt_dl_scores[slot_i], self.params.w_clf_dl),
                    ("donor", d_dist, self.params.w_clf_factorized),
                    ("linkage", l_dist, self.params.w_clf_factorized),
                    ("acceptor_sugar", acc_dist, self.params.w_clf_factorized),
                    ("acceptor_motif", per_gt_am_scores[slot_i], 1.04 * self.params.w_clf_factorized),
                ):
                    clf = self.classifier_suite.classifiers.get(t_name)
                    probs = self._clf_proba_single(t_name, aa_seq)
                    if clf is not None and probs is not None:
                        for c_i, cls_val in enumerate(clf.classes_):
                            p_v = float(probs[c_i])
                            if p_v >= 0.03:
                                target_dict[int(cls_val)] = target_dict.get(int(cls_val), 0.0) + weight_t * p_v
                                if t_name == "donor_linkage" and p_v >= 0.50:
                                    per_gt_ev[slot_i].setdefault(int(cls_val), AssignmentEvidence.PAN_GRAPH_CONSENSUS)

                bb_clf = self.classifier_suite.classifiers.get("is_backbone")
                bb_probs = self._clf_proba_single("is_backbone", aa_seq)
                if bb_clf is not None and bb_probs is not None:
                    for c_i, cls_val in enumerate(bb_clf.classes_):
                        if int(cls_val) == 1:
                            node_bb_probs[slot_i] = 0.55 * node_bb_probs[slot_i] + 0.45 * float(bb_probs[c_i])

                if fine_emp_bb is not None:
                    node_bb_probs[slot_i] = 0.82 * fine_emp_bb + 0.18 * node_bb_probs[slot_i]

                # Combine factorized donor x linkage into donor_linkage candidates
                if d_dist and l_dist:
                    top_d = sorted(d_dist.items(), key=lambda x: x[1], reverse=True)[:4]
                    top_l = sorted(l_dist.items(), key=lambda x: x[1], reverse=True)[:4]
                    for d_c, d_w in top_d:
                        for l_c, l_w in top_l:
                            dl_comb = encode_donor_linkage_label(ResidueType(d_c), LinkageType(l_c))
                            per_gt_dl_scores[slot_i][dl_comb] = (
                                per_gt_dl_scores[slot_i].get(dl_comb, 0.0) + 0.45 * min(d_w, l_w)
                            )

                # For unclustered GTs (cid == 0), reconcile unaccounted operon precursor sugars with linkage anomer
                if cid == 0 and synthesized_special and per_gt_dl_scores[slot_i] and l_dist:
                    covered_sugars: set[int] = set()
                    for other_g in gt_gene_indices:
                        other_cid = int(locus.nodes[other_g].cluster_id)
                        if other_cid > 0 and other_cid in self.pan_prior.cluster_sig_full:
                            for sig_o in self.pan_prior.cluster_sig_full[other_cid]:
                                covered_sugars.add(int(sig_o[0]))
                    unaccounted_special = synthesized_special - covered_sugars
                    if unaccounted_special:
                        tot_l_w = sum(l_dist.values()) or 1.0
                        p_beta_l = (
                            sum(w_l for l_c, w_l in l_dist.items() if LinkageType(l_c).anomer == Anomer.BETA)
                            / tot_l_w
                        )
                        p_alpha_l = (
                            sum(w_l for l_c, w_l in l_dist.items() if LinkageType(l_c).anomer == Anomer.ALPHA)
                            / tot_l_w
                        )
                        pref_anom = (
                            Anomer.BETA
                            if p_beta_l >= 0.65
                            else (Anomer.ALPHA if p_alpha_l >= 0.65 else Anomer.UNKNOWN)
                        )
                        for dl_k, dl_sc in list(per_gt_dl_scores[slot_i].items()):
                            r_k, lnk_k = decode_donor_linkage_label(dl_k)
                            if int(r_k.sugar) in unaccounted_special:
                                use_lnk = (
                                    LinkageType(pref_anom, lnk_k.donor, lnk_k.acceptor)
                                    if pref_anom != Anomer.UNKNOWN
                                    else lnk_k
                                )
                                dl_new = encode_donor_linkage_label(r_k, use_lnk)
                                max_cur = max(per_gt_dl_scores[slot_i].values(), default=1.0)
                                per_gt_dl_scores[slot_i][dl_new] = max_cur + 3.0 * float(dl_sc)
                                per_gt_ev[slot_i][dl_new] = AssignmentEvidence.PAN_GRAPH_CONSENSUS
                                d_c_sp = encode_donor_residue_chem(r_k)
                                d_dist[d_c_sp] = max(d_dist.values(), default=1.0) * 2.5
                                if covered_sugars:
                                    for c_sug in covered_sugars:
                                        c_chi = Chirality(Sugars.natural_chirality[c_sug])
                                        a_c_sp = encode_donor_residue_chem(
                                            ResidueType.from_sugar(c_sug, c_chi, False)
                                        )
                                        acc_dist[a_c_sp] = max(acc_dist.values(), default=1.0) * 2.0

            # Populate initial node_donor_probs and node_acc_probs vectors
            if d_dist and n_donors > 0:
                for d_c, val in d_dist.items():
                    if d_c in self.donor_to_idx:
                        node_donor_probs[slot_i, self.donor_to_idx[d_c]] += val
                node_donor_probs[slot_i, :] /= np.sum(node_donor_probs[slot_i, :])
            if acc_dist and n_donors > 0:
                for a_c, val in acc_dist.items():
                    if a_c in self.donor_to_idx:
                        node_acc_probs[slot_i, self.donor_to_idx[a_c]] += val
                node_acc_probs[slot_i, :] /= np.sum(node_acc_probs[slot_i, :])

        # Stage 2: Run SeroGen RelationalMPNN over 1-hop, 2-hop, and Virtual Operon Pool CSR graph
        if self.mpnn is not None and self.mpnn_dl_classes is not None and len(self.mpnn_dl_classes) > 0:
            init_g_idx = next(
                (idx_g for idx_g, g_n in enumerate(locus.nodes) if g_n.role == GeneRole.INITIATOR),
                -1,
            )
            init_aa = (
                bytes(translations_loc[init_g_idx].seq)
                if (has_genes_loc and 0 <= init_g_idx < len(translations_loc))
                else b""
            )
            slot_seqs: list[bytes] = [init_aa]
            for g_idx_s in gt_gene_indices:
                aa_s = (
                    bytes(translations_loc[g_idx_s].seq)
                    if (has_genes_loc and g_idx_s < len(translations_loc))
                    else b""
                )
                slot_seqs.append(aa_s)
            p_bb_init = np.concatenate(([1.0], node_bb_probs)).astype(np.float32)
            clamped_mask_slots = np.concatenate(([0], clamped_mask)).astype(np.uint8)
            x_nodes, csr_graph, clamped_full = self._build_mpnn_graph_for_locus(
                slot_seqs,
                synthesized_special,
                p_bb_init,
                clamped_mask_slots,
            )
            n_mpnn_nodes = x_nodes.shape[0]
            h_mpnn = self.mpnn.encode(x_nodes, csr_graph, clamped_mask=clamped_full)
            mpnn_dl_probs = self.mpnn.predict_node_proba(x_nodes, csr_graph, clamped_mask=clamped_full)
            self._last_mpnn_edge_scores = self.mpnn.score_edges(h_mpnn)

            for slot_i in range(n_gts):
                if clamped_mask[slot_i] or (slot_i + 1) >= n_mpnn_nodes:
                    continue
                has_consensus_i = bool(per_gt_ev[slot_i])
                w_mpnn_dl = (0.28 * self.params.w_mpnn_dl_free) if has_consensus_i else self.params.w_mpnn_dl_free
                d_msg = np.zeros(max(1, n_donors), dtype=np.float64)
                for v_i, dl_lbl_m_raw in enumerate(self.mpnn_dl_classes):
                    if v_i >= mpnn_dl_probs.shape[1]:
                        break
                    dl_lbl_m = int(dl_lbl_m_raw)
                    p_m = float(mpnn_dl_probs[slot_i + 1, v_i])
                    if p_m >= 0.04:
                        per_gt_dl_scores[slot_i][dl_lbl_m] = (
                            per_gt_dl_scores[slot_i].get(dl_lbl_m, 0.0) + w_mpnn_dl * p_m
                        )
                    res_m, _ = decode_donor_linkage_label(dl_lbl_m)
                    d_c_m = encode_donor_residue_chem(res_m)
                    if d_c_m in self.donor_to_idx:
                        d_msg[self.donor_to_idx[d_c_m]] += p_m
                tot_dm = float(np.sum(d_msg))
                if tot_dm > 1e-9 and n_donors > 0:
                    d_msg /= tot_dm
                    w_dm = min(0.45, max(0.02, 0.34 * w_mpnn_dl))
                    node_donor_probs[slot_i] = (1.0 - w_dm) * node_donor_probs[slot_i] + w_dm * d_msg
                    node_donor_probs[slot_i] /= np.sum(node_donor_probs[slot_i])

        # Propagate 2-layer syntenic donor<->acceptor bigram coupling and precursor gate
        syn_alpha = float(np.clip(self.params.syntenic_msg_alpha, 0.0, 0.50))
        if n_donors > 0 and n_gts >= 1:
            for _layer in range(2):
                new_d = node_donor_probs.copy()
                new_a = node_acc_probs.copy()
                new_bb = node_bb_probs.copy()
                for i_s in range(n_gts):
                    if clamped_mask[i_s]:
                        continue
                    msg_d = np.zeros(n_donors, dtype=np.float64)
                    msg_a = np.zeros(n_donors, dtype=np.float64)
                    bb_ctx = 0.0
                    n_nbrs = 0.0
                    for j_s in range(n_gts):
                        if i_s == j_s:
                            continue
                        dist_ij = abs(i_s - j_s)
                        if dist_ij <= 2:
                            w_syn = 1.0 if dist_ij == 1 else 0.45
                            if clamped_mask[j_s]:
                                w_syn *= 2.5
                            msg_a += w_syn * (self.bigram_trans_matrix.T @ node_donor_probs[j_s])
                            msg_d += w_syn * (self.bigram_trans_matrix @ node_acc_probs[j_s])
                            bb_ctx += w_syn * (1.0 - 0.65 * node_bb_probs[j_s])
                            n_nbrs += w_syn
                    if n_nbrs > 0.0:
                        tot_ma = float(np.sum(msg_a))
                        if tot_ma > 1e-9:
                            new_a[i_s] = (1.0 - syn_alpha) * node_acc_probs[i_s] + syn_alpha * (msg_a / tot_ma)
                        tot_md = float(np.sum(msg_d))
                        if tot_md > 1e-9:
                            d_alpha = 0.83 * syn_alpha
                            new_d[i_s] = (1.0 - d_alpha) * node_donor_probs[i_s] + d_alpha * (msg_d / tot_md)
                        new_bb[i_s] = float(np.clip(0.88 * node_bb_probs[i_s] + 0.12 * (bb_ctx / n_nbrs), 0.05, 0.98))
                    new_d[i_s] = new_d[i_s] * precursor_gate
                    s_d = float(np.sum(new_d[i_s]))
                    if s_d > 1e-9:
                        new_d[i_s] /= s_d
                    s_a = float(np.sum(new_a[i_s]))
                    if s_a > 1e-9:
                        new_a[i_s] /= s_a
                node_donor_probs = new_d
                node_acc_probs = new_a
                node_bb_probs = new_bb

        # Stage 3: Build contextualized _GTSlotSpec objects (including processive & bifunctional GT expansion)
        slots: list[_GTSlotSpec] = []
        for slot_i, g_idx in enumerate(gt_gene_indices):
            g_node = locus.nodes[g_idx]
            cid = int(g_node.cluster_id)
            dl_map = per_gt_dl_scores[slot_i]
            am_raw = per_gt_am_scores[slot_i]
            tot_am = sum(am_raw.values()) or 1.0
            am_map = {k_am: float(v_am / tot_am) for k_am, v_am in am_raw.items() if (v_am / tot_am) >= 0.02}

            # Modulate donor_linkage scores by MPNN-refined node_donor_probs and precursor gate
            refined_dl: list[tuple[ResidueType, LinkageType, float, AssignmentEvidence]] = []
            for dl_lbl, raw_s in dl_map.items():
                res_d, lnk = decode_donor_linkage_label(dl_lbl)
                d_c = encode_donor_residue_chem(res_d)
                mpnn_p = (
                    float(node_donor_probs[slot_i, self.donor_to_idx[d_c]])
                    if d_c in self.donor_to_idx
                    else 0.05
                )
                s_id = int(res_d.sugar)
                prec_mult = 1.25 if s_id in synthesized_special else (1.0 if s_id in HOUSEKEEPING_SUGARS else 0.85)
                final_s = float(raw_s) * prec_mult * (0.82 + 0.35 * mpnn_p)
                ev = per_gt_ev[slot_i].get(dl_lbl, AssignmentEvidence.GRAPH_ALIGNED)
                refined_dl.append((res_d, lnk, final_s, ev))

            refined_dl.sort(key=lambda item: item[2], reverse=True)
            if not refined_dl:
                # Default fallback donor/linkage from operon precursor pool
                pref_s = next(iter(synthesized_special), Sugars.name_to_index[b"Glc"])
                nat_chi = Chirality(Sugars.natural_chirality[pref_s])
                res_def = ResidueType.from_sugar(pref_s, nat_chi, False)
                lnk_def = LinkageType(Anomer.BETA, 1, 3)
                refined_dl.append((res_def, lnk_def, 0.40, AssignmentEvidence.RULE_INFERRED))

            top_cands = refined_dl[:4]
            norm_z = sum(c[2] for c in top_cands) or 1.0
            norm_cands = [
                (
                    c[0],
                    c[1],
                    float(min(0.99, max(0.35, 0.55 + 0.42 * (c[2] / norm_z)))),
                    c[3],
                )
                for c in top_cands
            ]
            acc_map = {
                self.donor_vocab[d_i]: float(node_acc_probs[slot_i, d_i])
                for d_i in range(n_donors)
                if node_acc_probs[slot_i, d_i] >= 0.02
            }
            slots.append(
                _GTSlotSpec(
                    gene_index=g_idx,
                    role=g_node.role,
                    cluster_id=cid,
                    candidates=norm_cands,
                    acceptor_probs=acc_map,
                    p_backbone=float(np.clip(node_bb_probs[slot_i], 0.05, 0.98)),
                    is_clamped=bool(clamped_mask[slot_i]),
                    acceptor_motif_probs=am_map,
                    mpnn_node_idx=slot_i + 1,
                )
            )

            # Processive / Bifunctional GT Multi-Node Expansion (e.g. WbbL/WbbM/WbdA in O3/O5/O12 or CPS_SYNTHASE)
            if (
                reference_glycan is None
                and not clamped_mask[slot_i]
                and cid > 0
                and cid in self.multi_edge_clusters
                and (
                    n_gts <= 2
                    or (
                        n_gts <= 4
                        and len(
                            {
                                (int(r_m.sugar), int(l_m.chemical_int))
                                for r_m, l_m in self.multi_edge_clusters[cid]
                            }
                        )
                        == 1
                    )
                )
            ):
                multi_sigs = self.multi_edge_clusters[cid]
                if len(multi_sigs) >= 2:
                    for res_ex, lnk_ex in multi_sigs[1:3]:
                        if lnk_ex.anomer != Anomer.UNKNOWN and lnk_ex.acceptor > 0:
                            slots.append(
                                _GTSlotSpec(
                                    gene_index=g_idx,
                                    role=g_node.role,
                                    cluster_id=cid,
                                    candidates=[(res_ex, lnk_ex, 0.88, AssignmentEvidence.PAN_GRAPH_CONSENSUS)],
                                    acceptor_probs=acc_map,
                                    p_backbone=0.92,
                                    is_clamped=False,
                                    acceptor_motif_probs=am_map,
                                    mpnn_node_idx=slot_i + 1,
                                )
                            )
            elif not clamped_mask[slot_i] and n_gts == 1:
                aa_seq_bi = (
                    bytes(translations_loc[g_idx].seq)
                    if (has_genes_loc and g_idx < len(translations_loc))
                    else b""
                )
                doms = split_bifunctional_gt_domains(aa_seq_bi, min_len=620, overlap=25)
                if doms is not None and len(norm_cands) >= 2:
                    c_dom = doms[1]
                    c_cands = list(norm_cands[1:3]) + [norm_cands[0]]
                    if self.ref_gt_embeddings is not None and self.ref_gt_dl_labels is not None:
                        q_c = self._embed_single_seq(c_dom)
                        if q_c is not None:
                            sims_c = self.ref_gt_embeddings @ q_c
                            best_nn = int(np.argmax(sims_c))
                            if float(sims_c[best_nn]) >= 0.80:
                                res_c, lnk_c = decode_donor_linkage_label(int(self.ref_gt_dl_labels[best_nn]))
                                c_cands.insert(
                                    0, (res_c, lnk_c, 0.86, AssignmentEvidence.PAN_GRAPH_CONSENSUS)
                                )
                    slots.append(
                        _GTSlotSpec(
                            gene_index=g_idx,
                            role=g_node.role,
                            cluster_id=cid,
                            candidates=c_cands[:3],
                            acceptor_probs=acc_map,
                            p_backbone=0.90,
                            is_clamped=False,
                            acceptor_motif_probs=am_map,
                            mpnn_node_idx=slot_i + 1,
                        )
                    )

        return slots

    def _predict_wzy_candidates(
        self,
        locus: Locus,
        root_res: ResidueType,
        tip_res: ResidueType,
        used_tip_carbons: set[int],
        clamped_by_gene: dict[int, GeneGlycanAssignment],
        reference_glycan: Glycan | None = None,
    ) -> tuple[int, LinkageType, float, AssignmentEvidence]:
        """Stage 4: Predict the inter-repeat `Wzy` polymer linkage `root_res -> tip_res` (`POLYMER=1`)."""
        wzy_idx = -1
        for idx_g, g_node in enumerate(locus.nodes):
            if g_node.role == GeneRole.POLYMERASE:
                wzy_idx = idx_g
                break

        if wzy_idx in clamped_by_gene and reference_glycan is not None:
            c_a = clamped_by_gene[wzy_idx]
            if c_a.edge_index is not None and 0 <= c_a.edge_index < len(reference_glycan.edges):
                lt_ref = reference_glycan.edges[c_a.edge_index].edge_type.with_topology(
                    is_backbone=True, is_polymer=True
                )
                return wzy_idx, lt_ref, 0.99, AssignmentEvidence.PAN_GRAPH_CONSENSUS

        scores: dict[int, float] = {}
        ev_map: dict[int, AssignmentEvidence] = {}

        if wzy_idx >= 0:
            cid = int(locus.nodes[wzy_idx].cluster_id)
            for c_tier, tier_scale in ((cid, 1.0), (self.pan_prior.fine_to_med.get(cid, 0), 0.72)):
                if c_tier > 0 and c_tier in self.pan_prior.cluster_sig_full:
                    w_counts = self.pan_prior.cluster_sig_full[c_tier]
                    tot_w = sum(w_counts.values())
                    for sig, cnt in w_counts.items():
                        anom_val, don_val, acc_val = sig[3], sig[4], sig[5]
                        if Anomer(anom_val) == Anomer.UNKNOWN or acc_val == 0:
                            continue
                        l_chem = int(LinkageType(Anomer(anom_val), don_val, acc_val).chemical_int)
                        p_w = cnt / max(1e-6, tot_w)
                        scores[l_chem] = scores.get(l_chem, 0.0) + 3.0 * tier_scale * p_w
                        if cnt >= 1.5 and p_w >= 0.45:
                            ev_map[l_chem] = AssignmentEvidence.PAN_GRAPH_CONSENSUS

            aa_seq = (
                bytes(locus.genes.translations[wzy_idx].seq)
                if (len(locus.genes) > 0 and wzy_idx < len(locus.genes.translations))
                else b""
            )
            if len(aa_seq) >= 60 and self.classifier_suite is not None and "wzy_linkage" in self.classifier_suite.classifiers:
                w_clf = self.classifier_suite["wzy_linkage"]
                probs = self._clf_proba_single("wzy_linkage", aa_seq)
                if probs is not None:
                    for c_i, l_val in enumerate(w_clf.classes_):
                        p_v = float(probs[c_i])
                        if p_v >= 0.05:
                            scores[int(l_val)] = scores.get(int(l_val), 0.0) + 1.4 * p_v
                            if p_v >= 0.55:
                                ev_map.setdefault(int(l_val), AssignmentEvidence.PAN_GRAPH_CONSENSUS)

            if len(aa_seq) >= 70 and self.ref_wzy_embeddings is not None and self.ref_wzy_links is not None:
                q_vec = self._embed_single_seq(aa_seq)
                if q_vec is not None:
                    sims = self.ref_wzy_embeddings @ q_vec
                    top_k = np.argsort(sims)[-5:][::-1]
                    for idx_nn in top_k:
                        sim_val = float(sims[idx_nn])
                        if sim_val >= 0.80:
                            l_nn = int(self.ref_wzy_links[idx_nn])
                            scores[l_nn] = scores.get(l_nn, 0.0) + 1.8 * (sim_val**2)
                            if sim_val >= 0.90:
                                ev_map.setdefault(l_nn, AssignmentEvidence.PAN_GRAPH_CONSENSUS)

        # Pan-Glycan empirical polymer closure prior P(root_res -l-> tip_res)
        r_chem = encode_donor_residue_chem(root_res)
        t_chem = encode_donor_residue_chem(tip_res)
        for (s_c, l_c, a_c), cnt in self.wzy_closure_counts.items():
            if s_c == r_chem and a_c == t_chem:
                scores[l_c] = scores.get(l_c, 0.0) + 0.6 * np.log1p(cnt)
            elif a_c == t_chem:
                scores[l_c] = scores.get(l_c, 0.0) + 0.15 * np.log1p(cnt)

        # Filter out acceptor carbons already occupied on tip_res
        valid_scores = {
            l_c: s_v
            for l_c, s_v in scores.items()
            if LinkageType(l_c).acceptor not in used_tip_carbons and LinkageType(l_c).acceptor > 0
        }
        if not valid_scores:
            fallback_acc = next((c for c in (3, 4, 2, 6) if c not in used_tip_carbons), 3)
            lt_fb = LinkageType(Anomer.BETA, 1, fallback_acc, is_backbone=True, is_polymer=True)
            return wzy_idx, lt_fb, 0.50, AssignmentEvidence.RULE_INFERRED

        best_l_chem, best_s = max(valid_scores.items(), key=lambda item: item[1])
        tot_s = sum(valid_scores.values()) or 1.0
        lt_best = LinkageType(best_l_chem).with_topology(is_backbone=True, is_polymer=True)
        ev = ev_map.get(
            best_l_chem,
            AssignmentEvidence.GRAPH_ALIGNED if wzy_idx >= 0 else AssignmentEvidence.RULE_INFERRED,
        )
        conf = float(min(0.99, max(0.45, 0.55 + 0.42 * (best_s / tot_s))))
        return wzy_idx, lt_best, conf, ev

    def predict(
        self,
        locus: Locus,
        *,
        archetype: LocusArchetype | None = None,
        is_repeating: bool = True,
        top_k: int = 5,
        known_assignments: Sequence[GeneGlycanAssignment] | dict[int, GeneGlycanAssignment] | None = None,
        reference_glycan: Glycan | None = None,
        allow_fingerprint_lookup: bool = True,
        resolve_homologs: bool = True,
    ) -> tuple[GlycanPrediction, ...]:
        """Predict top-`k` candidate `Glycan` graphs de novo from a `Locus` via 4-stage GNN + Arborescence."""
        if len(locus.nodes) == 0:
            return ()

        rescued_locus = (
            self.assigner.rescue_cassette_roles(locus, reference_glycan)
            if reference_glycan is not None
            else locus
        )
        if resolve_homologs and len(rescued_locus.genes) > 0 and all(n.cluster_id == 0 for n in rescued_locus.nodes):
            rescued_locus = self.resolve_locus_homologs(rescued_locus)
        arch = archetype if archetype is not None else LocusArchetype.CPS_WZY

        clamped_by_gene: dict[int, GeneGlycanAssignment] = {}
        if isinstance(known_assignments, dict):
            clamped_by_gene.update(known_assignments)
        elif known_assignments is not None:
            for ka in known_assignments:
                if ka.gene_index >= 0:
                    clamped_by_gene[int(ka.gene_index)] = ka

        # Stage 2: Compute Operon Precursor Pool (`c_pool`)
        named_pool, synthesized_special, precursor_gate = self._compute_operon_precursor_pool(rescued_locus)

        # Stage 1-3: Predict root candidate(s) and contextualized GT slots
        init_gene_idx, root_cands = self._predict_root_candidates(rescued_locus, arch, clamped_by_gene)
        gt_slots = self._build_gt_slots(
            rescued_locus,
            synthesized_special,
            precursor_gate,
            clamped_by_gene,
            reference_glycan=reference_glycan,
        )

        # Check exact Pan-Locus cluster cassette fingerprint match when allowed and unclamped
        fp_candidates: list[tuple[Glycan, float, tuple[GeneGlycanAssignment, ...]]] = []
        if allow_fingerprint_lookup and not clamped_by_gene:
            fp_cids = tuple(
                int(n.cluster_id)
                for n in rescued_locus.nodes
                if n.cluster_id > 0
                and n.role
                in (
                    GeneRole.INITIATOR,
                    GeneRole.GLYCOSYLTRANSFERASE,
                    GeneRole.POLYMERASE,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                )
            )
            if fp_cids in self.locus_fingerprints:
                seen_canons: set[bytes] = set()
                for _, fp_gly, _ in self.locus_fingerprints[fp_cids]:
                    c_b = fp_gly.to_csdb()
                    if c_b not in seen_canons:
                        seen_canons.add(c_b)
                        inferred_assigns = self.assigner.assign_record(
                            fp_gly,
                            rescued_locus,
                            archetype=arch,
                            is_repeating=is_repeating,
                            pan_prior=self.pan_prior,
                        )
                        mean_sc = (
                            float(np.mean([a.score for a in inferred_assigns if a.is_aligned]))
                            if any(a.is_aligned for a in inferred_assigns)
                            else 0.92
                        )
                        fp_candidates.append((fp_gly, 50.0 + mean_sc, inferred_assigns))

        if not gt_slots and not fp_candidates:
            r_res, r_prob, r_ev = root_cands[0]
            g_single = Glycan.from_topology(nodes=(r_res,), edges=())
            a_single = (
                GeneGlycanAssignment(
                    gene_index=init_gene_idx,
                    role=GeneRole.INITIATOR,
                    bipartite_glycan_index=0,
                    target_type=TargetType.REDUCING_END,
                    residue=r_res,
                    linkage=None,
                    edge_index=None,
                    evidence=r_ev,
                    score=float(r_prob),
                ),
            )
            return (
                GlycanPrediction(
                    glycan=g_single,
                    score=float(r_prob),
                    probability=1.0,
                    assignments=a_single,
                    precursor_pool=named_pool,
                ),
            )

        # Stage 4: Enumerate top node-state combinations & solve Constrained Spanning Arborescences
        m_nodes = 1 + len(gt_slots)
        state_combos: list[tuple[float, ResidueType, float, AssignmentEvidence, list[tuple[ResidueType, LinkageType, float, AssignmentEvidence]], list[int]]] = []

        # Generate top state assignments across (root_cands, gt_slots)
        for r_res, r_prob, r_ev in root_cands[:2]:
            base_choices = [slot.candidates[0] for slot in gt_slots]
            base_bb = [1 if slot.p_backbone >= 0.45 else 0 for slot in gt_slots]
            if all(b == 0 for b in base_bb) and base_bb:
                base_bb[0] = 1

            def _score_combo(
                ch_list: list[tuple[ResidueType, LinkageType, float, AssignmentEvidence]],
                bb_list: list[int],
            ) -> float:
                sc = float(np.log(max(r_prob, 1e-3)))
                present_sugars = {int(r_res.sugar)}
                for idx_c, c_item in enumerate(ch_list):
                    sc += float(np.log(max(c_item[2], 1e-3)))
                    p_b = gt_slots[idx_c].p_backbone if bb_list[idx_c] == 1 else (1.0 - gt_slots[idx_c].p_backbone)
                    sc += self.params.w_backbone_log * float(np.log(max(p_b, 1e-3)))
                    present_sugars.add(int(c_item[0].sugar))
                # Stoichiometric precursor coverage bonus
                for sp_s in synthesized_special:
                    if sp_s in present_sugars:
                        sc += self.params.w_precursor_bonus
                    else:
                        sc -= 0.70 * self.params.w_precursor_bonus
                return sc

            state_combos.append((_score_combo(base_choices, base_bb), r_res, r_prob, r_ev, base_choices, base_bb))

            # Single-slot alternative reactions & branching topology flips
            for s_i, slot in enumerate(gt_slots):
                for alt_c in slot.candidates[1:3]:
                    alt_choices = list(base_choices)
                    alt_choices[s_i] = alt_c
                    state_combos.append(
                        (_score_combo(alt_choices, base_bb), r_res, r_prob, r_ev, alt_choices, list(base_bb))
                    )
                if not slot.is_clamped and 0.25 <= slot.p_backbone <= 0.85 and len(gt_slots) >= 2:
                    flipped_bb = list(base_bb)
                    flipped_bb[s_i] = 1 - flipped_bb[s_i]
                    if any(b == 1 for b in flipped_bb):
                        state_combos.append(
                            (_score_combo(base_choices, flipped_bb), r_res, r_prob, r_ev, list(base_choices), flipped_bb)
                        )
                # Short-backbone + long-branch topology when Wzy specifically closes onto slot s_i's sugar
                if not slot.is_clamped and len(gt_slots) >= 3 and sum(base_bb) >= 2:
                    one_bb = [1 if idx_b == s_i else 0 for idx_b in range(len(gt_slots))]
                    if one_bb != base_bb:
                        state_combos.append(
                            (_score_combo(base_choices, one_bb), r_res, r_prob, r_ev, list(base_choices), one_bb)
                        )

        state_combos.sort(key=lambda item: item[0], reverse=True)
        raw_candidates: list[tuple[Glycan, float, tuple[GeneGlycanAssignment, ...]]] = list(fp_candidates)

        for combo_sc, r_res, r_prob, r_ev, ch_list, bb_list in state_combos[: max(6, top_k * 2)]:
            score_mat = np.full((m_nodes, m_nodes), -1e9, dtype=np.float64)
            acc_carbons = np.zeros(m_nodes, dtype=np.int32)
            bb_flags = np.ones(m_nodes, dtype=np.uint8)

            slot_res: list[ResidueType] = [r_res]
            slot_lnk: list[LinkageType] = [LinkageType(Anomer.UNKNOWN, 0, 0)]
            for s_i, (res_u, lnk_u, _, _) in enumerate(ch_list):
                is_bb_u = bool(bb_list[s_i] == 1)
                slot_res.append(res_u.with_backbone(is_bb_u).with_reducing_end(False))
                slot_lnk.append(lnk_u.with_topology(is_backbone=is_bb_u, is_polymer=False))
                acc_carbons[s_i + 1] = int(lnk_u.acceptor)
                bb_flags[s_i + 1] = np.uint8(bb_list[s_i])

            # Precompute clamped reference source node -> arborescence slot index map
            ref_node_to_slot: dict[int, int] = {}
            if reference_glycan is not None:
                ref_node_to_slot[reference_glycan.reducing_end_index] = 0
                unclamped_slots: list[int] = []
                for s_idx_c, sl_c in enumerate(gt_slots, start=1):
                    if sl_c.is_clamped and sl_c.gene_index in clamped_by_gene:
                        ca_c = clamped_by_gene[sl_c.gene_index]
                        if ca_c.edge_index is not None and 0 <= ca_c.edge_index < len(reference_glycan.edges):
                            ref_node_to_slot[reference_glycan.edges[ca_c.edge_index].source] = s_idx_c
                    else:
                        unclamped_slots.append(s_idx_c)
                if len(unclamped_slots) == 1:
                    unmapped_core = [
                        n_i
                        for n_i, n_obj in enumerate(reference_glycan.nodes)
                        if not n_obj.is_modifier and n_i not in ref_node_to_slot
                    ]
                    if len(unmapped_core) == 1:
                        ref_node_to_slot[unmapped_core[0]] = unclamped_slots[0]

            # Fill pairwise directed edge score matrix score_mat[u, w] for u -> w
            sugar_counts: dict[int, int] = {}
            for r_s in slot_res:
                s_key = int(r_s.sugar)
                sugar_counts[s_key] = sugar_counts.get(s_key, 0) + 1

            for u in range(1, m_nodes):
                slot_u = gt_slots[u - 1]
                res_u = slot_res[u]
                lnk_u = slot_lnk[u]
                d_chem_u = encode_donor_residue_chem(res_u)
                l_chem_u = int(lnk_u.chemical_int)
                has_pan_sig_u = (slot_u.cluster_id > 0 and slot_u.cluster_id in self.pan_prior.cluster_sig_full) or (
                    self.pan_prior.fine_to_med.get(slot_u.cluster_id, 0) in self.pan_prior.cluster_sig_full
                )

                clamped_target_slot: int | None = None
                clamped_target_chem: int | None = None
                if slot_u.is_clamped and reference_glycan is not None and slot_u.gene_index in clamped_by_gene:
                    ca_u = clamped_by_gene[slot_u.gene_index]
                    if ca_u.edge_index is not None and 0 <= ca_u.edge_index < len(reference_glycan.edges):
                        e_ref_u = reference_glycan.edges[ca_u.edge_index]
                        clamped_target_slot = ref_node_to_slot.get(e_ref_u.target)
                        clamped_target_chem = encode_donor_residue_chem(reference_glycan.nodes[e_ref_u.target])

                c_acc_u = min(max(int(lnk_u.acceptor), 0), 7)
                mpnn_u = slot_u.mpnn_node_idx
                mpnn_parent_logits: dict[int, float] = {}
                if (
                    self._last_mpnn_edge_scores is not None
                    and 0 <= mpnn_u < self._last_mpnn_edge_scores.shape[1]
                    and c_acc_u < self._last_mpnn_edge_scores.shape[2]
                ):
                    raw_p_logits: dict[int, float] = {}
                    for w_cand in range(m_nodes):
                        if w_cand == u:
                            continue
                        mpnn_w = 0 if w_cand == 0 else gt_slots[w_cand - 1].mpnn_node_idx
                        if 0 <= mpnn_w < self._last_mpnn_edge_scores.shape[0] and mpnn_w != mpnn_u:
                            raw_p_logits[w_cand] = float(self._last_mpnn_edge_scores[mpnn_w, mpnn_u, c_acc_u])
                    if raw_p_logits:
                        mean_p_l = sum(raw_p_logits.values()) / len(raw_p_logits)
                        for w_cand, l_val in raw_p_logits.items():
                            mpnn_parent_logits[w_cand] = 0.55 * float(np.tanh(0.25 * (l_val - mean_p_l)))

                for w in range(m_nodes):
                    if u == w:
                        continue
                    res_w = slot_res[w]
                    a_chem_w = encode_donor_residue_chem(res_w)
                    is_dup_sugar_w = sugar_counts.get(int(res_w.sugar), 0) >= 2

                    # 0. Hard/soft clamping of parent edge when u is clamped
                    if clamped_target_slot is not None:
                        if w == clamped_target_slot:
                            score_mat[u, w] = 30.0
                        else:
                            score_mat[u, w] = -30.0
                        continue

                    # 1. GNN / Classifier Acceptor-Sugar & Acceptor-Motif Potential
                    p_acc = slot_u.acceptor_probs.get(a_chem_w, 0.04)
                    s_uw = self.params.w_acceptor_prob * np.log(max(p_acc, 1e-3))
                    if clamped_target_chem is not None:
                        s_uw += 8.0 if a_chem_w == clamped_target_chem else -8.0

                    if slot_u.acceptor_motif_probs:
                        if w == 0:
                            p_am = max(
                                (
                                    slot_u.acceptor_motif_probs.get(
                                        encode_acceptor_motif_label(res_w, anom_cand, 1),
                                        0.0,
                                    )
                                    for anom_cand in (Anomer.BETA, Anomer.ALPHA, Anomer.UNKNOWN)
                                ),
                                default=0.0,
                            )
                        else:
                            pos_w = 2 if bb_flags[w] == 1 else 3
                            am_key = encode_acceptor_motif_label(res_w, slot_lnk[w].anomer, pos_w)
                            am_alt = encode_acceptor_motif_label(res_w, slot_lnk[w].anomer, 5 - pos_w)
                            p_am = max(
                                slot_u.acceptor_motif_probs.get(am_key, 0.0),
                                0.85 * slot_u.acceptor_motif_probs.get(am_alt, 0.0),
                            )
                        if p_am >= 0.02:
                            w_am = (
                                self.params.w_acceptor_motif
                                if (is_dup_sugar_w or not has_pan_sig_u)
                                else (0.14 * self.params.w_acceptor_motif)
                            )
                            s_uw += w_am * p_am

                    if w in mpnn_parent_logits:
                        w_mpnn_e = (
                            self.params.w_mpnn_edge
                            if (is_dup_sugar_w or not has_pan_sig_u)
                            else (0.20 * self.params.w_mpnn_edge)
                        )
                        s_uw += w_mpnn_e * mpnn_parent_logits[w]

                    # 1b. Direct PanGraphConsensusPrior joint (donor, linkage, acceptor_sugar) match for slot_u
                    has_exact_pan_acc_uw = False
                    for c_tier, tier_scale in (
                        (slot_u.cluster_id, 1.0),
                        (self.pan_prior.fine_to_med.get(slot_u.cluster_id, 0), 0.72),
                    ):
                        if c_tier > 0 and c_tier in self.pan_prior.cluster_sig_full:
                            sig_map_u = self.pan_prior.cluster_sig_full[c_tier]
                            tot_u = sum(sig_map_u.values())
                            exact_acc_w = 0.0
                            exact_dl_w = 0.0
                            acc_by_sugar: dict[int, float] = {}
                            for sig_u, cnt_u in sig_map_u.items():
                                if (
                                    sig_u[0] == int(res_u.sugar)
                                    and sig_u[1] == int(res_u.chirality)
                                    and sig_u[2] == int(res_u.is_furanose)
                                    and sig_u[3] == int(lnk_u.anomer)
                                    and sig_u[4] == int(lnk_u.donor)
                                    and sig_u[5] == int(lnk_u.acceptor)
                                ):
                                    exact_dl_w += cnt_u
                                    acc_s_u = int(sig_u[6])
                                    acc_by_sugar[acc_s_u] = acc_by_sugar.get(acc_s_u, 0.0) + cnt_u
                                    if acc_s_u == int(res_w.sugar):
                                        exact_acc_w += cnt_u
                            if exact_dl_w > 0:
                                max_acc_share = max(acc_by_sugar.values()) / exact_dl_w if acc_by_sugar else 0.0
                                if max_acc_share >= 0.75:
                                    if exact_acc_w > 0:
                                        has_exact_pan_acc_uw = True
                                        s_uw += self.params.w_pan_acc_match * tier_scale * (exact_acc_w / max(1e-6, tot_u))
                                    else:
                                        s_uw -= (0.49 * self.params.w_pan_acc_match) * tier_scale
                                elif exact_acc_w > 0:
                                    has_exact_pan_acc_uw = True
                                    s_uw += 0.20 * self.params.w_pan_acc_match * tier_scale * (exact_acc_w / max(1e-6, tot_u))
                            elif tot_u >= 1.8:
                                s_uw -= (0.49 * self.params.w_pan_acc_match) * tier_scale

                    # 1c. Terminal leaf / branch-point constraint on parent w
                    if w > 0 and not has_exact_pan_acc_uw:
                        slot_w = gt_slots[w - 1]
                        if bb_flags[w] == 0:
                            # Side-chain branch residue w rarely accepts further extensions
                            s_uw -= 3.5
                        if slot_w.cluster_id > 0 and slot_w.cluster_id in self.pan_prior.cluster_sig_full:
                            sig_map_w = self.pan_prior.cluster_sig_full[slot_w.cluster_id]
                            tot_w_s = sum(sig_map_w.values())
                            leaf_w = sum(cnt_w * float(sig_w[8]) for sig_w, cnt_w in sig_map_w.items()) / max(
                                1e-6, tot_w_s
                            )
                            if leaf_w >= 0.70:
                                s_uw -= 5.0 * leaf_w

                    # 2. Pan-Glycan Empirical Disaccharide Bigram Prior
                    c_exact = self.disaccharide_counts.get((d_chem_u, l_chem_u, a_chem_w), 0)
                    c_pair = self.donor_acc_counts.get((d_chem_u, a_chem_w), 0)
                    s_uw += self.params.w_disac_bigram * np.log1p(c_exact) + (0.38 * self.params.w_disac_bigram) * np.log1p(
                        c_pair
                    )

                    # 3. Pan-Locus Syntenic & Operon Progression Prior
                    if w == 0:
                        if init_gene_idx >= 0:
                            init_cid = int(rescued_locus.nodes[init_gene_idx].cluster_id)
                            if init_cid > 0 and slot_u.cluster_id > 0:
                                step1_map = self.pan_prior.init_adj_gt_step1.get((init_cid, slot_u.cluster_id), {})
                                c_init_gt = sum(step1_map.values()) + self.pan_prior.pan_locus_edge_counts.get(
                                    (init_cid, slot_u.cluster_id), 0
                                )
                                s_uw += (2.27 * self.params.w_syntenic_step) * np.log1p(c_init_gt)
                        # Step-1 GT is typically at one end of the GT cassette
                        if u == 1 or u == m_nodes - 1:
                            s_uw += 0.64 * self.params.w_syntenic_step
                    else:
                        slot_w = gt_slots[w - 1]
                        if slot_u.gene_index == slot_w.gene_index and u == w + 1:
                            s_uw += 12.0
                        elif abs(slot_u.gene_index - slot_w.gene_index) == 1:
                            s_uw += 0.82 * self.params.w_syntenic_step
                        elif abs(u - w) == 1:
                            s_uw += 0.45 * self.params.w_syntenic_step

                        # Processive GT pair disambiguation: main-chain extends distal unit, branch attaches to proximal unit
                        if slot_u.gene_index != slot_w.gene_index:
                            is_w_prox = w < m_nodes - 1 and gt_slots[w].gene_index == slot_w.gene_index
                            is_w_dist = w > 1 and gt_slots[w - 2].gene_index == slot_w.gene_index
                            if is_w_prox:
                                s_uw += 2.5 if bb_flags[u] == 0 else -2.5
                            elif is_w_dist:
                                s_uw += 2.5 if bb_flags[u] == 1 else -2.5
                        if slot_u.cluster_id > 0 and slot_w.cluster_id > 0:
                            c_bg = (
                                self.pan_prior.syntenic_pair_record_counts.get(
                                    (slot_w.cluster_id, slot_u.cluster_id), 0
                                )
                                + self.pan_prior.syntenic_pair_record_counts.get(
                                    (slot_u.cluster_id, slot_w.cluster_id), 0
                                )
                                + self.pan_prior.pan_locus_edge_counts.get(
                                    (slot_w.cluster_id, slot_u.cluster_id), 0
                                )
                                + self.pan_prior.pan_locus_edge_counts.get(
                                    (slot_u.cluster_id, slot_w.cluster_id), 0
                                )
                            )
                            s_uw += self.params.w_syntenic_step * np.log1p(c_bg)

                    score_mat[u, w] = s_uw

            best_parents, best_scores, n_found = _solve_constrained_arborescences_kernel(
                score_mat,
                acc_carbons,
                bb_flags,
                max_candidates=max(8, top_k * 3),
            )
            if n_found == 0:
                msa_res = maximum_spanning_arborescence(score_mat.T, root=0, slot_exclusivity=False)
                if msa_res.score > -1e15:
                    best_parents[0, :] = msa_res.parents
                    best_scores[0] = msa_res.score
                    n_found = 1

            for k_idx in range(n_found):
                parents = best_parents[k_idx]
                tree_sc = float(best_scores[k_idx])
                nodes_list: list[ResidueType] = list(slot_res)
                edges_list: list[LinkageEdge] = []
                assigns_list: list[GeneGlycanAssignment] = [
                    GeneGlycanAssignment(
                        gene_index=init_gene_idx,
                        role=GeneRole.INITIATOR,
                        bipartite_glycan_index=0,
                        target_type=TargetType.REDUCING_END,
                        residue=slot_res[0],
                        linkage=None,
                        edge_index=None,
                        evidence=r_ev,
                        score=float(r_prob),
                    )
                ]

                bb_children_of: dict[int, int] = {}
                used_carbons_on: dict[int, set[int]] = {i_n: set() for i_n in range(m_nodes)}

                for u in range(1, m_nodes):
                    w = int(parents[u])
                    lnk_u = slot_lnk[u]
                    e_idx = len(edges_list)
                    edges_list.append(LinkageEdge(source=u, target=w, edge_type=lnk_u))
                    if lnk_u.acceptor > 0:
                        used_carbons_on[w].add(int(lnk_u.acceptor))
                    if bb_flags[u] == 1:
                        bb_children_of[w] = u

                    slot_u = gt_slots[u - 1]
                    _, _, cand_prob, cand_ev = ch_list[u - 1]
                    assigns_list.append(
                        GeneGlycanAssignment(
                            gene_index=slot_u.gene_index,
                            role=slot_u.role,
                            bipartite_glycan_index=m_nodes + e_idx,
                            target_type=TargetType.LINKAGE,
                            residue=None,
                            linkage=lnk_u,
                            edge_index=e_idx,
                            evidence=cand_ev,
                            score=float(cand_prob),
                        )
                    )

                # Trace main-chain backbone tip from root 0
                tip_idx = 0
                visited_tip: set[int] = {0}
                while tip_idx in bb_children_of and bb_children_of[tip_idx] not in visited_tip:
                    tip_idx = bb_children_of[tip_idx]
                    visited_tip.add(tip_idx)

                # Close polymer repeat with Wzy linkage (`0 -> tip_idx`) when repeating
                has_wzy_gene = any(n.role == GeneRole.POLYMERASE for n in rescued_locus.nodes)
                if is_repeating and (
                    has_wzy_gene
                    or arch
                    in (
                        LocusArchetype.CPS_WZY,
                        LocusArchetype.LPS_WZY,
                        LocusArchetype.LPS_ABC,
                        LocusArchetype.CPS_ABC,
                        LocusArchetype.CPS_SYNTHASE,
                    )
                ) and tip_idx != 0:
                    wzy_g_idx, wzy_lt, wzy_conf, wzy_ev = self._predict_wzy_candidates(
                        rescued_locus,
                        nodes_list[0],
                        nodes_list[tip_idx],
                        used_carbons_on.get(tip_idx, set()),
                        clamped_by_gene,
                        reference_glycan=reference_glycan,
                    )
                    # Score Wzy acceptor sugar specificity against tip_idx
                    if wzy_g_idx in clamped_by_gene and reference_glycan is not None:
                        c_w = clamped_by_gene[wzy_g_idx]
                        if c_w.edge_index is not None and 0 <= c_w.edge_index < len(reference_glycan.edges):
                            e_ref_w = reference_glycan.edges[c_w.edge_index]
                            if ref_node_to_slot.get(e_ref_w.target) == tip_idx:
                                tree_sc += 25.0
                            elif encode_donor_residue_chem(nodes_list[tip_idx]) == encode_donor_residue_chem(
                                reference_glycan.nodes[e_ref_w.target]
                            ):
                                tree_sc += 8.0
                            else:
                                tree_sc -= 15.0
                    elif wzy_g_idx >= 0:
                        wzy_cid = int(rescued_locus.nodes[wzy_g_idx].cluster_id)
                        if wzy_cid > 0 and wzy_cid in self.pan_prior.cluster_sig_full:
                            w_sigs_all = self.pan_prior.cluster_sig_full[wzy_cid]
                            root_s_id = int(nodes_list[0].sugar)
                            w_sigs_root = {
                                sig_w: cnt_w for sig_w, cnt_w in w_sigs_all.items() if sig_w[0] == root_s_id
                            }
                            w_sigs = w_sigs_root if w_sigs_root else w_sigs_all
                            w_tot = sum(w_sigs.values())
                            w_tip_match = sum(
                                cnt_w
                                for sig_w, cnt_w in w_sigs.items()
                                if sig_w[6] == int(nodes_list[tip_idx].sugar)
                                and sig_w[5] == int(wzy_lt.acceptor)
                            )
                            if w_tip_match > 0:
                                tree_sc += 4.5 * (w_tip_match / max(1e-6, w_tot))
                            else:
                                tree_sc -= 2.5
                    tree_sc += 1.5 * float(np.log(max(wzy_conf, 0.1)))
                    wzy_e_idx = len(edges_list)
                    edges_list.append(LinkageEdge(source=0, target=tip_idx, edge_type=wzy_lt))
                    assigns_list.append(
                        GeneGlycanAssignment(
                            gene_index=wzy_g_idx,
                            role=GeneRole.POLYMERASE,
                            bipartite_glycan_index=m_nodes + wzy_e_idx,
                            target_type=TargetType.LINKAGE,
                            residue=None,
                            linkage=wzy_lt,
                            edge_index=wzy_e_idx,
                            evidence=wzy_ev,
                            score=float(wzy_conf),
                        )
                    )

                # Attach modifier substituents from tailing modifier ORFs
                gly_core = Glycan.from_topology(nodes=nodes_list, edges=edges_list)
                comp_gly, _, comp_map = self.assigner.complete_glycan_modifiers(
                    gly_core, rescued_locus, pan_prior=self.pan_prior
                )
                if comp_map:
                    gly_final = Glycan.from_topology(nodes=comp_gly.nodes, edges=comp_gly.edges)
                    mod_assigns = list(assigns_list)
                    has_obligate_mod = False
                    for mod_node_idx, mod_gene_idx in sorted(comp_map.items()):
                        mod_res = gly_final.nodes[mod_node_idx]
                        m_role = rescued_locus.nodes[mod_gene_idx].role
                        has_specific_mod_edge = any(
                            e_m.source == mod_node_idx and e_m.edge_type.acceptor > 0 for e_m in gly_final.edges
                        )
                        if has_specific_mod_edge or m_role in (
                            GeneRole.PYRUVYLTRANSFERASE,
                            GeneRole.FORMYLTRANSFERASE,
                            GeneRole.ALANINE_TRANSFERASE,
                        ):
                            has_obligate_mod = True
                        mod_assigns.append(
                            GeneGlycanAssignment(
                                gene_index=mod_gene_idx,
                                role=m_role,
                                bipartite_glycan_index=mod_node_idx,
                                target_type=TargetType.MODIFIED_MONO,
                                residue=mod_res,
                                linkage=None,
                                edge_index=None,
                                evidence=AssignmentEvidence.GRAPH_ALIGNED,
                                score=0.86,
                            )
                        )
                    if has_obligate_mod:
                        raw_candidates.append((gly_final, combo_sc + tree_sc + 0.25, tuple(mod_assigns)))
                        raw_candidates.append((gly_core, combo_sc + tree_sc, tuple(assigns_list)))
                    else:
                        raw_candidates.append((gly_core, combo_sc + tree_sc + 0.25, tuple(assigns_list)))
                        raw_candidates.append((gly_final, combo_sc + tree_sc, tuple(mod_assigns)))
                else:
                    raw_candidates.append((gly_core, combo_sc + tree_sc, tuple(assigns_list)))

        # Deduplicate candidates by canonical CSDB string (or topological isomorphism) and softmax-normalize
        unique_cands: list[tuple[Glycan, float, tuple[GeneGlycanAssignment, ...]]] = []
        seen_csdb: set[bytes] = set()
        raw_candidates.sort(key=lambda item: item[1], reverse=True)
        for cand_gly, cand_sc, cand_asg in raw_candidates:
            c_bytes = cand_gly.canonical or cand_gly.to_csdb()
            if c_bytes in seen_csdb:
                continue
            seen_csdb.add(c_bytes)
            unique_cands.append((cand_gly, cand_sc, cand_asg))
            if len(unique_cands) >= top_k:
                break

        if not unique_cands:
            return ()

        raw_scores = np.array([c[1] for c in unique_cands], dtype=np.float64)
        shifted = (raw_scores - np.max(raw_scores)) * 0.5
        exp_s = np.exp(np.clip(shifted, -30.0, 0.0))
        probs = exp_s / np.sum(exp_s)

        return tuple(
            GlycanPrediction(
                glycan=unique_cands[i][0],
                score=round(float(unique_cands[i][1]), 4),
                probability=round(float(probs[i]), 4),
                assignments=unique_cands[i][2],
                precursor_pool=named_pool,
            )
            for i in range(len(unique_cands))
        )
