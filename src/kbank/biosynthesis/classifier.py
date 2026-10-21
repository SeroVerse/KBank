"""Sequence-based 1D-CNN classifiers for glycosyltransferase and initiator specificity.

Provides `ClusterClassifier` (a subclass of `serogen.genes.GeneRoleClassifier`) and
`BiosyntheticClassifierSuite` for predicting donor monosaccharide, glycosidic linkage,
retaining/inverting catalytic mechanism, and initiating Und-PP sugar directly from
amino acid sequences, with homology-cluster probability pooling (`predict_cluster_proba`).
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from serogen.core.seq.seq import Sequences
from serogen.genes.cnn import GeneRoleClassifier
from serogen.genes.roles import GeneRole
from serogen.loci.models import Loci, Locus

from kbank.biosynthesis.align import AssignmentEvidence, TargetType, _edge_signatures
from kbank.glycan import Anomer, Chirality, Glycan, LinkageType, ResidueType
from kbank.sugars import Sugars

if TYPE_CHECKING:
    from kbank.db.records import GlycanRecords


VALID_TARGETS: tuple[str, ...] = (
    "linkage",
    "donor",
    "donor_linkage",
    "acceptor_sugar",
    "acceptor_motif",
    "is_backbone",
    "mechanism",
    "anomer",
    "initiator_sugar",
    "wzy_linkage",
    "modifier_linkage",
    "component_int",
)

MODIFIER_ROLES: frozenset[GeneRole] = frozenset(
    {
        GeneRole.ACETYLTRANSFERASE,
        GeneRole.PYRUVYLTRANSFERASE,
        GeneRole.METHYLTRANSFERASE,
        GeneRole.FORMYLTRANSFERASE,
        GeneRole.PHOSPHOTRANSFERASE,
        GeneRole.ALANINE_TRANSFERASE,
        GeneRole.DEACETYLASE,
    }
)


def encode_donor_residue_chem(res: ResidueType) -> int:
    """Return topology-invariant integer code for a donor `ResidueType` (`sugar | chirality | furanose`)."""
    return int(res) & ~ResidueType.TOPOLOGY_MASK


def encode_donor_linkage_label(res: ResidueType, link: LinkageType) -> int:
    """Pack `(donor_residue_chem, linkage_chem)` into a single positive 32-bit integer class label."""
    d_chem = encode_donor_residue_chem(res) & 0xFFF
    l_chem = link.chemical_int & 0x3FF
    return (d_chem << 10) | l_chem


def decode_donor_linkage_label(label: int) -> tuple[ResidueType, LinkageType]:
    """Unpack a `donor_linkage` integer class label into `(ResidueType, LinkageType)`."""
    d_chem = (int(label) >> 10) & 0xFFF
    l_chem = int(label) & 0x3FF
    return ResidueType(d_chem), LinkageType(l_chem | int(LinkageType.TAG_EDGE.mask))


def encode_acceptor_motif_label(res: ResidueType, acc_anomer: Anomer | int, acc_role: int) -> int:
    """Pack `(acceptor_residue_chem, acc_anomer, acc_role)` into a positive integer class label.

    `acc_role`: `1` = reducing-end root (`@root`), `2` = internal backbone (`@bb`), `3` = side-chain branch (`@br`).
    """
    a_chem = encode_donor_residue_chem(res) & 0xFFF
    anom_bits = int(acc_anomer) & 0x3
    role_bits = int(acc_role) & 0x3
    return (a_chem << 4) | (anom_bits << 2) | role_bits


def decode_acceptor_motif_label(label: int) -> tuple[ResidueType, Anomer, int]:
    """Unpack an `acceptor_motif` integer label into `(ResidueType, Anomer, acc_role)`."""
    val = int(label)
    a_chem = (val >> 4) & 0xFFF
    anom_bits = (val >> 2) & 0x3
    role_bits = val & 0x3
    anom = Anomer(anom_bits) if anom_bits in (0, 1, 2) else Anomer.UNKNOWN
    return ResidueType(a_chem), anom, role_bits


def extract_acceptor_motif_for_edge(glycan: Glycan, edge_idx: int) -> int:
    """Compute the `acceptor_motif` integer code (`{anomer}{sugar}@{root|bb|br}`) for the target residue of `edge_idx`."""
    edge = glycan.edges[edge_idx]
    t_idx = int(edge.target)
    t_res = glycan.nodes[t_idx]
    red_idx = glycan.reducing_end_index
    acc_anomer = Anomer.UNKNOWN
    is_bb_node = False
    for k_o, e_o in enumerate(glycan.edges):
        if int(e_o.source) == t_idx and not glycan.nodes[e_o.target].is_modifier:
            if e_o.edge_type.is_backbone:
                is_bb_node = True
                if e_o.edge_type.anomer in (Anomer.ALPHA, Anomer.BETA):
                    acc_anomer = e_o.edge_type.anomer
                    break
            elif acc_anomer == Anomer.UNKNOWN and e_o.edge_type.anomer in (Anomer.ALPHA, Anomer.BETA):
                acc_anomer = e_o.edge_type.anomer
    if t_idx == red_idx or t_res.is_reducing_end:
        acc_role = 1  # @root
    elif is_bb_node:
        acc_role = 2  # @bb
    else:
        acc_role = 3  # @br
    return encode_acceptor_motif_label(t_res, acc_anomer, acc_role)


def split_bifunctional_gt_domains(
    aa_seq: bytes,
    *,
    min_len: int = 550,
    overlap: int = 25,
) -> tuple[bytes, bytes] | None:
    """Split a multi-domain bifunctional GT (`len(aa_seq) >= min_len`) into `(N_term_seq, C_term_seq)`."""
    n = len(aa_seq)
    if n < min_len:
        return None
    mid = n // 2
    return aa_seq[: min(n, mid + overlap)], aa_seq[max(0, mid - overlap) :]


def encode_modifier_linkage_label(mod_sugar: int, acceptor_carbon: int) -> int:
    """Pack `(modifier_sugar_id, acceptor_carbon)` into a positive integer class label."""
    return ((int(mod_sugar) & 0xFF) << 4) | (int(acceptor_carbon) & 0xF)


def decode_modifier_linkage_label(label: int) -> tuple[int, int]:
    """Unpack a `modifier_linkage` integer class label into `(modifier_sugar_id, acceptor_carbon)`."""
    return (int(label) >> 4) & 0xFF, int(label) & 0xF


class ClusterClassifier(GeneRoleClassifier):
    """1D-CNN classifier predicting fine-grained glycan reaction specificity from ORF protein sequences.

    Subclasses `serogen.genes.GeneRoleClassifier` to support target tasks:
    - `"donor_linkage"`: Joint `(donor_residue, anomer, donor_c, acceptor_c)` glycosidic reaction.
    - `"donor"`: Transferred donor monosaccharide (`ResidueType` chemical bits).
    - `"linkage"`: Formed glycosidic bond (`LinkageType` chemical bits: `anomer`, `donor_c`, `acceptor_c`).
    - `"mechanism"`: Retaining (`1`) vs. inverting (`2`) GT stereochemical mechanism.
    - `"anomer"`: Alpha (`1`) vs. beta (`2`) anomeric configuration.
    - `"initiator_sugar"`: Und-PP initiating monosaccharide index (`Sugars` index) for `INITIATOR` genes.
    - `"component_int"`: Raw bit-packed `GeneGlycanAssignment.component_int`.
    """

    target: str

    def __init__(
        self,
        target: str = "donor_linkage",
        vocab_size: int = 23,
        embed_dim: int = 16,
        num_filters: int = 64,
        kernel_size: int | tuple[int, ...] = (5, 9, 13),
        learning_rate: float = 0.008,
        epochs: int = 25,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        random_state: int | None = 42,
        class_weight: str | None = None,
    ) -> None:
        """Initialize `ClusterClassifier` for a specific biosynthetic `target`."""
        if target not in VALID_TARGETS:
            raise ValueError(f"Unsupported target {target!r}. Expected one of {VALID_TARGETS}.")
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_size=kernel_size,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.target = target

    @staticmethod
    def format_target_label(target: str, label: int) -> str:
        """Convert an integer class label for `target` into a human-readable biochemical string."""
        val = int(label)
        if val < 0:
            return "UNKNOWN"
        if target == "donor_linkage":
            res, link = decode_donor_linkage_label(val)
            return f"{res!r}({link.name.decode('ascii', errors='replace')})"
        if target == "donor":
            return repr(ResidueType(val))
        if target == "acceptor_sugar":
            return repr(ResidueType(val))
        if target == "acceptor_motif":
            res, anom, role_id = decode_acceptor_motif_label(val)
            anom_s = "a" if anom == Anomer.ALPHA else ("b" if anom == Anomer.BETA else "")
            role_s = "root" if role_id == 1 else ("bb" if role_id == 2 else ("br" if role_id == 3 else "?"))
            return f"{anom_s}{res!r}@{role_s}"
        if target == "linkage" or target == "wzy_linkage":
            link = LinkageType(val | int(LinkageType.TAG_EDGE.mask))
            return link.name.decode("ascii", errors="replace")
        if target == "is_backbone":
            return "BACKBONE" if val == 1 else "BRANCH"
        if target == "modifier_linkage":
            mod_s, acc_c = decode_modifier_linkage_label(val)
            s_str = (
                Sugars.names[mod_s].decode("ascii", errors="replace")
                if 0 <= mod_s < len(Sugars.names)
                else f"Mod#{mod_s}"
            )
            c_str = str(acc_c) if acc_c > 0 else "?"
            return f"{s_str}(1-{c_str})"
        if target == "mechanism":
            return "RETAINING" if val == 1 else ("INVERTING" if val == 2 else "UNKNOWN")
        if target == "anomer":
            return "alpha" if val == int(Anomer.ALPHA) else ("beta" if val == int(Anomer.BETA) else "?")
        if target == "initiator_sugar":
            if 0 <= val < len(Sugars.names):
                return Sugars.names[val].decode("ascii", errors="replace")
            return f"Sugar#{val}"
        if target == "component_int":
            if val & int(LinkageType.TAG_EDGE.mask):
                return LinkageType(val).name.decode("ascii", errors="replace")
            return repr(ResidueType(val))
        return str(val)

    def decode_label(self, label: int) -> str:
        """Return a human-readable biochemical label for an integer class value."""
        return self.format_target_label(self.target, int(label))

    @property
    def class_names(self) -> tuple[str, ...]:
        """Human-readable names corresponding to `self.classes_`."""
        if not self.is_fitted:
            return ()
        return tuple(self.decode_label(int(c)) for c in self.classes_)

    def top_k_accuracy(
        self,
        X: Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        y: Sequence[int] | np.ndarray,
        k: int = 3,
    ) -> float:
        """Compute Top-`k` classification accuracy on `(X, y)`."""
        probs = self.predict_proba(X)
        y_arr = np.asarray(y, dtype=np.int64)
        if len(y_arr) == 0 or len(probs) == 0:
            return 0.0
        k_eff = min(max(1, int(k)), probs.shape[1])
        top_k_idx = np.argsort(probs, axis=1)[:, -k_eff:]
        top_k_classes = self.classes_[top_k_idx].astype(np.int64)
        hits = np.any(top_k_classes == y_arr[:, None], axis=1)
        return float(np.mean(hits))

    def predict_cluster_proba(
        self,
        X: Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        cluster_ids: Sequence[int] | np.ndarray,
        *,
        pool: str = "log_proba",
        ignore_non_positive: bool = True,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
        """Aggregate sequence class probabilities per homology `cluster_id` via geometric (`log_proba`) or mean pooling."""
        probs = self.predict_proba(X)
        c_arr = np.asarray(cluster_ids, dtype=np.int64)
        if len(c_arr) != len(probs):
            raise ValueError(f"cluster_ids length ({len(c_arr)}) does not match X length ({len(probs)}).")
        mask = (c_arr > 0) if ignore_non_positive else np.ones(len(c_arr), dtype=bool)
        if not np.any(mask):
            return np.empty(0, dtype=np.int64), np.empty((0, probs.shape[1]), dtype=np.float32)

        c_valid = c_arr[mask]
        p_valid = probs[mask]
        uniq_cids, inv = np.unique(c_valid, return_inverse=True)
        n_c = len(uniq_cids)
        n_cls = probs.shape[1]
        counts = np.bincount(inv, minlength=n_c).astype(np.float32)[:, None]

        if pool == "log_proba":
            log_p = np.log(np.clip(p_valid, 1e-8, 1.0)).astype(np.float32)
            agg_log = np.zeros((n_c, n_cls), dtype=np.float32)
            np.add.at(agg_log, inv, log_p)
            agg_log /= np.maximum(counts, 1.0)
            agg_log -= np.max(agg_log, axis=1, keepdims=True)
            exp_p = np.exp(agg_log)
            out_probs = (exp_p / np.maximum(np.sum(exp_p, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        else:
            agg_p = np.zeros((n_c, n_cls), dtype=np.float32)
            np.add.at(agg_p, inv, p_valid)
            agg_p /= np.maximum(counts, 1.0)
            out_probs = (agg_p / np.maximum(np.sum(agg_p, axis=1, keepdims=True), 1e-12)).astype(np.float32)

        return uniq_cids.astype(np.int64), out_probs

    @staticmethod
    def extract_training_data(
        records: GlycanRecords,
        *,
        target: str = "donor_linkage",
        min_score: float = 0.72,
        include_graph_aligned: bool = True,
        min_class_count: int = 2,
    ) -> tuple[list[bytes], np.ndarray, np.ndarray, np.ndarray]:
        """Extract `(sequences, y_labels, sample_weights, cluster_ids)` from `GlycanRecords`.

        Args:
            records: Curated `GlycanRecords` database with `.loci`, `.glycans`, and `.assignments`.
            target: Target task in `VALID_TARGETS`.
            min_score: Minimum assignment confidence score for `GRAPH_ALIGNED` examples.
            include_graph_aligned: Whether to include high-confidence `GRAPH_ALIGNED` assignments
                alongside `PAN_GRAPH_CONSENSUS` assignments.
            min_class_count: Minimum number of samples required for a class to be retained.

        Returns:
            tuple[list[bytes], np.ndarray, np.ndarray, np.ndarray]:
                `(sequences, y_labels, sample_weights, cluster_ids)`.
        """
        if target not in VALID_TARGETS:
            raise ValueError(f"Unsupported target {target!r}. Expected one of {VALID_TARGETS}.")

        seqs: list[bytes] = []
        labels: list[int] = []
        weights: list[float] = []
        cids: list[int] = []

        for i in range(len(records)):
            loc = records.loci[i]
            if len(loc.nodes) == 0 or len(loc.genes) == 0:
                continue
            gly = records.glycans[i]
            assigns = records.assignments[i]
            translations = loc.genes.translations

            # Identify bifunctional GTs assigned to 2+ distinct glycosidic edges so we can split N/C-terminal domains
            gene_edge_orders: dict[int, list[int]] = {}
            _, assembly_depths, _ = gly.polymer_edge_and_assembly_depths(gly.reducing_end_index)
            for a_pre in assigns:
                if (
                    a_pre.is_aligned
                    and a_pre.gene_index >= 0
                    and a_pre.target_type == TargetType.LINKAGE
                    and a_pre.edge_index is not None
                    and 0 <= a_pre.edge_index < len(gly.edges)
                ):
                    gene_edge_orders.setdefault(int(a_pre.gene_index), []).append(int(a_pre.edge_index))
            for g_k, e_list in gene_edge_orders.items():
                if len(e_list) >= 2:
                    gene_edge_orders[g_k] = sorted(
                        dict.fromkeys(e_list),
                        key=lambda ek: (assembly_depths.get(gly.edges[ek].source, 99), ek),
                    )

            for a in assigns:
                if not a.is_aligned or a.gene_index < 0 or a.gene_index >= len(loc.nodes):
                    continue
                if a.gene_index >= len(translations):
                    continue
                if a.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS:
                    w_base = 1.50 if float(a.score) >= 0.93 else 1.25
                elif include_graph_aligned and a.evidence == AssignmentEvidence.GRAPH_ALIGNED and a.score >= min_score:
                    w_base = 0.85
                else:
                    continue

                aa_seq = bytes(translations[a.gene_index].seq)
                if len(aa_seq) < 80:
                    continue

                g_node = loc.nodes[a.gene_index]
                cid = int(g_node.cluster_id)

                # Split multi-domain bifunctional GTs (>= 550 aa) assigned to 2+ reactions into N/C catalytic domains
                e_ord = gene_edge_orders.get(int(a.gene_index), [])
                if (
                    len(e_ord) >= 2
                    and a.edge_index is not None
                    and int(a.edge_index) in e_ord
                ):
                    doms = split_bifunctional_gt_domains(aa_seq, min_len=550, overlap=25)
                    if doms is not None:
                        pos_dom = e_ord.index(int(a.edge_index))
                        aa_seq = doms[0] if pos_dom == 0 else doms[1]
                        if pos_dom > 0 and cid > 0:
                            cid = cid + 500000

                if target == "initiator_sugar":
                    if a.role != GeneRole.INITIATOR or a.residue is None or a.residue.is_modifier:
                        continue
                    y_val = int(a.residue.sugar)
                elif target == "modifier_linkage":
                    if a.role not in MODIFIER_ROLES:
                        continue
                    mod_sugar_id = -1
                    acc_c = 0
                    if a.residue is not None and a.residue.is_modifier:
                        mod_sugar_id = int(a.residue.sugar)
                        if a.bipartite_glycan_index >= 0:
                            for e_m in gly.edges:
                                if e_m.source == a.bipartite_glycan_index and not gly.nodes[e_m.target].is_modifier:
                                    acc_c = int(e_m.edge_type.acceptor)
                                    break
                    elif a.target_type == TargetType.LINKAGE and a.edge_index is not None and 0 <= a.edge_index < len(gly.edges):
                        e_m = gly.edges[a.edge_index]
                        if gly.nodes[e_m.source].is_modifier:
                            mod_sugar_id = int(gly.nodes[e_m.source].sugar)
                            acc_c = int(e_m.edge_type.acceptor)
                    if mod_sugar_id < 0:
                        continue
                    y_val = encode_modifier_linkage_label(mod_sugar_id, acc_c)
                elif target == "wzy_linkage":
                    if a.role != GeneRole.POLYMERASE or a.target_type != TargetType.LINKAGE or a.edge_index is None:
                        continue
                    if not (0 <= a.edge_index < len(gly.edges)):
                        continue
                    w_link = gly.edges[a.edge_index].edge_type
                    if w_link.anomer == Anomer.UNKNOWN or w_link.acceptor == 0:
                        continue
                    y_val = int(w_link.chemical_int)
                elif target == "component_int":
                    y_val = int(a.component_int)
                    if y_val <= 0:
                        continue
                else:
                    if a.role not in (
                        GeneRole.GLYCOSYLTRANSFERASE,
                        GeneRole.POLYMERASE,
                        GeneRole.TRANSMEMBRANE_SYNTHASE,
                    ):
                        continue
                    if target == "is_backbone" and a.role == GeneRole.POLYMERASE:
                        continue
                    if a.target_type != TargetType.LINKAGE or a.edge_index is None:
                        continue
                    if not (0 <= a.edge_index < len(gly.edges)):
                        continue
                    edge = gly.edges[a.edge_index]
                    s_res = gly.nodes[edge.source]
                    t_res = gly.nodes[edge.target]
                    if s_res.is_modifier or t_res.is_modifier:
                        continue
                    link = edge.edge_type

                    if target == "donor":
                        y_val = encode_donor_residue_chem(s_res)
                    elif target == "acceptor_sugar":
                        y_val = encode_donor_residue_chem(t_res)
                    elif target == "acceptor_motif":
                        y_val = extract_acceptor_motif_for_edge(gly, a.edge_index)
                    elif target == "is_backbone":
                        y_val = 1 if link.is_backbone else 0
                    elif target == "linkage":
                        if link.anomer == Anomer.UNKNOWN or link.acceptor == 0:
                            continue
                        y_val = int(link.chemical_int)
                    elif target == "donor_linkage":
                        if link.anomer == Anomer.UNKNOWN or link.acceptor == 0:
                            continue
                        y_val = encode_donor_linkage_label(s_res, link)
                    elif target == "mechanism":
                        _, _, _, mech = _edge_signatures(gly, a.edge_index)
                        if mech not in (1, 2):
                            continue
                        y_val = int(mech)
                    elif target == "anomer":
                        if link.anomer not in (Anomer.ALPHA, Anomer.BETA):
                            continue
                        y_val = int(link.anomer)
                    else:
                        continue

                seqs.append(aa_seq)
                labels.append(y_val)
                weights.append(float(w_base * float(a.score)))
                cids.append(cid)

        if not seqs:
            return [], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

        y_arr = np.asarray(labels, dtype=np.int64)
        w_arr = np.asarray(weights, dtype=np.float32)
        c_arr = np.asarray(cids, dtype=np.int64)

        if min_class_count > 1:
            uniq, counts = np.unique(y_arr, return_counts=True)
            keep_classes = set(uniq[counts >= min_class_count].tolist())
            keep_mask = np.array([int(y_v) in keep_classes for y_v in y_arr], dtype=bool)
            seqs = [s for s, m in zip(seqs, keep_mask, strict=False) if m]
            y_arr = y_arr[keep_mask]
            w_arr = w_arr[keep_mask]
            c_arr = c_arr[keep_mask]

        return seqs, y_arr, w_arr, c_arr

    def fit_from_records(
        self,
        records: GlycanRecords,
        *,
        min_score: float = 0.72,
        include_graph_aligned: bool = True,
        min_class_count: int = 2,
        init_from: GeneRoleClassifier | None = None,
        backbone_lr_scale: float = 1.0,
    ) -> Self:
        """Extract training examples for `self.target` from `records` and fit the 1D-CNN."""
        seqs, y_arr, w_arr, _ = self.extract_training_data(
            records,
            target=self.target,
            min_score=min_score,
            include_graph_aligned=include_graph_aligned,
            min_class_count=min_class_count,
        )
        if len(seqs) == 0 or len(np.unique(y_arr)) < 2:
            raise ValueError(f"Insufficient training data for target={self.target!r} (n={len(seqs)}).")
        return self.fit(
            seqs,
            y_arr,
            sample_weight=w_arr,
            init_from=init_from,
            backbone_lr_scale=backbone_lr_scale,
        )


class BiosyntheticClassifierSuite:
    """Factorized suite of `ClusterClassifier` models for hybrid pan-graph / sequence inference.

    Combines five specialized 1D-CNN heads:
    - `donor_linkage_clf`: Joint `(donor, linkage)` reaction classifier.
    - `donor_clf`: Donor monosaccharide classifier (`D-Glcp`, `L-Rhap`, `D-Galp`, etc.).
    - `linkage_clf`: Glycosidic linkage classifier (`a1-3`, `b1-4`, etc.).
    - `mechanism_clf`: Stereochemical mechanism classifier (`RETAINING` vs. `INVERTING`).
    - `initiator_clf`: `INITIATOR` Und-PP sugar classifier (`D-Galp`, `D-Glcp`, `D-GlcNAcp`, etc.).
    """

    SUITE_TARGETS: tuple[str, ...] = (
        "donor_linkage",
        "donor",
        "linkage",
        "acceptor_sugar",
        "acceptor_motif",
        "is_backbone",
        "mechanism",
        "initiator_sugar",
        "wzy_linkage",
        "modifier_linkage",
    )

    def __init__(
        self,
        classifiers: dict[str, ClusterClassifier] | None = None,
    ) -> None:
        """Initialize `BiosyntheticClassifierSuite` with a dictionary of fitted `ClusterClassifier`s."""
        self.classifiers: dict[str, ClusterClassifier] = dict(classifiers) if classifiers else {}
        self._cluster_probs: dict[str, dict[int, dict[int, float]]] = {}
        self._gene_probs: dict[str, dict[tuple[int, int], dict[int, float]]] = {}
        self._seq_probs: dict[str, dict[bytes, dict[int, float]]] = {}

    @property
    def is_fitted(self) -> bool:
        """True if at least the core GT classifiers (`donor`, `linkage`, `mechanism`) are fitted."""
        return all(
            t in self.classifiers and self.classifiers[t].is_fitted
            for t in ("donor", "linkage", "mechanism")
        )

    def __getitem__(self, target: str) -> ClusterClassifier:
        """Return the `ClusterClassifier` for `target`."""
        return self.classifiers[target]

    @classmethod
    def fit_from_records(
        cls,
        records: GlycanRecords,
        *,
        embed_dim: int = 16,
        num_filters: int = 64,
        kernel_size: int | tuple[int, ...] = (5, 9, 13),
        learning_rate: float = 0.008,
        epochs: int = 25,
        batch_size: int = 32,
        random_state: int = 42,
        min_score: float = 0.72,
        warm_start_backbone: bool = True,
    ) -> Self:
        """Train all factorized `ClusterClassifier` heads on `records`."""
        base_clf: GeneRoleClassifier | None = None
        if warm_start_backbone:
            try:
                pretrained = GeneRoleClassifier.pretrained()
                if (
                    pretrained.embed_dim == embed_dim
                    and pretrained.num_filters == num_filters
                ):
                    base_clf = pretrained
            except Exception:
                base_clf = None

        suite: dict[str, ClusterClassifier] = {}
        for idx_t, target in enumerate(cls.SUITE_TARGETS):
            clf = ClusterClassifier(
                target=target,
                embed_dim=embed_dim,
                num_filters=num_filters,
                kernel_size=kernel_size,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                random_state=random_state + idx_t,
                class_weight=None,
            )
            min_cnt = (
                4
                if target in ("donor_linkage", "donor", "acceptor_sugar", "acceptor_motif")
                else (3 if target in ("linkage", "wzy_linkage") else 2)
            )
            try:
                clf.fit_from_records(
                    records,
                    min_score=min_score,
                    include_graph_aligned=True,
                    min_class_count=min_cnt,
                    init_from=base_clf,
                    backbone_lr_scale=0.5 if base_clf is not None else 1.0,
                )
                suite[target] = clf
            except ValueError:
                continue
        return cls(classifiers=suite)

    def save(self, path: str | Path) -> None:
        """Save all fitted `ClusterClassifier` heads into a single compressed `.npz` archive (zero pickle)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        present_targets: list[str] = []
        for target, clf in self.classifiers.items():
            if not clf.is_fitted:
                continue
            present_targets.append(target)
            prefix = f"{target}__"
            arrays[f"{prefix}classes"] = clf.classes_
            arrays[f"{prefix}embedding"] = clf.embedding_
            arrays[f"{prefix}conv_w"] = clf.conv_w_
            arrays[f"{prefix}conv_b"] = clf.conv_b_
            arrays[f"{prefix}dense_w"] = clf.dense_w_
            arrays[f"{prefix}dense_b"] = clf.dense_b_
            arrays[f"{prefix}k_sizes"] = clf.k_sizes_
            arrays[f"{prefix}loss_history"] = np.asarray(getattr(clf, "loss_history_", []), dtype=np.float32)
            arrays[f"{prefix}hyperparams"] = np.array(
                [clf.vocab_size, clf.embed_dim, clf.num_filters], dtype=np.int32
            )
        arrays["targets"] = np.array([t.encode("ascii") for t in present_targets], dtype="S32")
        import os

        tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, out_path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Self:
        """Load a `BiosyntheticClassifierSuite` from a `.npz` archive (zero pickle)."""
        from serogen.genes.cnn import _compute_lut

        resolved_path = (
            Path(path)
            if path is not None
            else (Path(__file__).resolve().parent.parent / "data" / "cluster_classifiers.npz")
        )
        with np.load(resolved_path, allow_pickle=False) as data:
            target_bytes = data["targets"]
            targets = [bytes(t).decode("ascii") for t in target_bytes]
            suite: dict[str, ClusterClassifier] = {}
            for target in targets:
                prefix = f"{target}__"
                hp = data[f"{prefix}hyperparams"]
                k_sizes = data[f"{prefix}k_sizes"].astype(np.int32)
                unique_ks = np.unique(k_sizes).astype(np.int32)
                clf = ClusterClassifier(
                    target=target,
                    vocab_size=int(hp[0]),
                    embed_dim=int(hp[1]),
                    num_filters=int(hp[2]),
                    kernel_size=tuple(int(k) for k in unique_ks),
                )
                clf.classes_ = data[f"{prefix}classes"]
                clf.embedding_ = data[f"{prefix}embedding"].astype(np.float32)
                clf.conv_w_ = data[f"{prefix}conv_w"].astype(np.float32)
                clf.conv_b_ = data[f"{prefix}conv_b"].astype(np.float32)
                clf.dense_w_ = data[f"{prefix}dense_w"].astype(np.float32)
                clf.dense_b_ = data[f"{prefix}dense_b"].astype(np.float32)
                clf.k_sizes_ = k_sizes
                clf.unique_ks_ = unique_ks
                clf.k_indices_ = np.array(
                    [np.where(unique_ks == k)[0][0] for k in k_sizes],
                    dtype=np.int32,
                )
                clf.loss_history_ = (
                    data[f"{prefix}loss_history"].astype(np.float32).tolist()
                    if f"{prefix}loss_history" in data
                    else []
                )
                clf.lut_ = _compute_lut(clf.embedding_, clf.conv_w_, clf.k_sizes_)
                suite[target] = clf
        return cls(classifiers=suite)

    @classmethod
    @cache
    def pretrained(cls) -> Self | None:
        """Load the packaged `src/kbank/data/cluster_classifiers.npz` suite if available, else `None`."""
        try:
            res = files("kbank.data").joinpath("cluster_classifiers.npz")
            with as_file(res) as p:
                if p.exists():
                    return cls.load(p)
        except Exception:
            pass
        fallback_path = Path(__file__).resolve().parent.parent / "data" / "cluster_classifiers.npz"
        if fallback_path.exists():
            return cls.load(fallback_path)
        return None

    def precompute_for_loci(self, loci: Loci | Sequence[Locus]) -> None:
        """Precompute cluster-pooled (`cluster_id > 0`) and single-ORF (`cluster_id == 0`) probabilities across `loci`."""
        if not self.is_fitted:
            return

        gt_seqs: list[bytes] = []
        gt_cids: list[int] = []
        gt_keys: list[tuple[int, int]] = []

        wzy_seqs: list[bytes] = []
        wzy_cids: list[int] = []
        wzy_keys: list[tuple[int, int]] = []

        init_seqs: list[bytes] = []
        init_cids: list[int] = []
        init_keys: list[tuple[int, int]] = []

        mod_seqs: list[bytes] = []
        mod_cids: list[int] = []
        mod_keys: list[tuple[int, int]] = []

        n_loci = len(loci)
        for loc_i in range(n_loci):
            loc = loci[loc_i]
            if len(loc.nodes) == 0 or len(loc.genes) == 0:
                continue
            translations = loc.genes.translations
            n_tr = len(translations)
            for g_i, g_node in enumerate(loc.nodes):
                if g_i >= n_tr:
                    continue
                aa_seq = bytes(translations[g_i].seq)
                if len(aa_seq) < 60:
                    continue
                cid = int(g_node.cluster_id)
                if g_node.role in (
                    GeneRole.GLYCOSYLTRANSFERASE,
                    GeneRole.POLYMERASE,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                ):
                    gt_seqs.append(aa_seq)
                    gt_cids.append(cid)
                    gt_keys.append((loc_i, g_i))
                    if len(aa_seq) >= 550:
                        doms = split_bifunctional_gt_domains(aa_seq, min_len=550, overlap=25)
                        if doms is not None:
                            gt_seqs.append(doms[0])
                            gt_cids.append((cid + 400000) if cid > 0 else 0)
                            gt_keys.append((loc_i, -(g_i + 1) * 2))
                            gt_seqs.append(doms[1])
                            gt_cids.append((cid + 500000) if cid > 0 else 0)
                            gt_keys.append((loc_i, -(g_i + 1) * 2 - 1))
                    if g_node.role == GeneRole.POLYMERASE:
                        wzy_seqs.append(aa_seq)
                        wzy_cids.append(cid)
                        wzy_keys.append((loc_i, g_i))
                elif g_node.role == GeneRole.INITIATOR:
                    init_seqs.append(aa_seq)
                    init_cids.append(cid)
                    init_keys.append((loc_i, g_i))
                elif g_node.role in MODIFIER_ROLES:
                    mod_seqs.append(aa_seq)
                    mod_cids.append(cid)
                    mod_keys.append((loc_i, g_i))

        self._cluster_probs.clear()
        self._gene_probs.clear()
        self._seq_probs.clear()

        def _populate_group(
            seqs_list: list[bytes],
            cids_list: list[int],
            keys_list: list[tuple[int, int]],
            targets: tuple[str, ...],
        ) -> None:
            if not seqs_list:
                return
            s_batch = Sequences.from_bytes(seqs_list)
            c_arr = np.asarray(cids_list, dtype=np.int64)
            for target in targets:
                clf = self.classifiers.get(target)
                if clf is None or not clf.is_fitted:
                    continue
                classes = [int(c) for c in clf.classes_]
                raw_probs = clf.predict_proba(s_batch)
                gene_map: dict[tuple[int, int], dict[int, float]] = {}
                seq_map = self._seq_probs.setdefault(target, {})
                for idx_s, key in enumerate(keys_list):
                    row = raw_probs[idx_s]
                    dist_s = {
                        classes[c_i]: float(row[c_i]) for c_i in range(len(classes)) if row[c_i] >= 0.01
                    }
                    gene_map[key] = dist_s
                    seq_map[seqs_list[idx_s]] = dist_s
                self._gene_probs[target] = gene_map

                uniq_cids, c_probs = clf.predict_cluster_proba(
                    s_batch,
                    c_arr,
                    pool="log_proba",
                    ignore_non_positive=True,
                )
                c_map: dict[int, dict[int, float]] = {}
                for idx_c, cid_val in enumerate(uniq_cids):
                    row = c_probs[idx_c]
                    c_map[int(cid_val)] = {
                        classes[c_i]: float(row[c_i]) for c_i in range(len(classes)) if row[c_i] >= 0.01
                    }
                self._cluster_probs[target] = c_map

        _populate_group(
            gt_seqs,
            gt_cids,
            gt_keys,
            ("donor_linkage", "donor", "linkage", "acceptor_sugar", "acceptor_motif", "is_backbone", "mechanism"),
        )
        _populate_group(wzy_seqs, wzy_cids, wzy_keys, ("wzy_linkage",))
        _populate_group(init_seqs, init_cids, init_keys, ("initiator_sugar",))
        _populate_group(mod_seqs, mod_cids, mod_keys, ("modifier_linkage",))

    def _get_target_dist(
        self,
        target: str,
        cluster_id: int,
        aa_seq: bytes | None = None,
    ) -> dict[int, float]:
        """Retrieve precomputed cluster distribution or cached single-sequence distribution from `aa_seq`."""
        if cluster_id > 0:
            c_dist = self._cluster_probs.get(target, {}).get(int(cluster_id))
            if c_dist is not None:
                return c_dist
        if aa_seq is not None and len(aa_seq) >= 60:
            s_map = self._seq_probs.setdefault(target, {})
            cached_s = s_map.get(aa_seq)
            if cached_s is not None:
                return cached_s
            clf = self.classifiers.get(target)
            if clf is not None and clf.is_fitted:
                probs = clf.predict_proba([aa_seq])[0]
                dist_val = {
                    int(clf.classes_[i]): float(probs[i])
                    for i in range(len(clf.classes_))
                    if probs[i] >= 0.01
                }
                s_map[aa_seq] = dist_val
                return dist_val
        return {}

    def preferred_init_sugar(
        self,
        cluster_id: int,
        candidate_sugars: set[int],
        aa_seq: bytes | None = None,
    ) -> tuple[int | None, float]:
        """Return `(best_sugar_idx, probability)` from `initiator_sugar` classifier among `candidate_sugars`."""
        dist = self._get_target_dist("initiator_sugar", cluster_id, aa_seq=aa_seq)
        if not dist:
            return None, 0.0
        best_s: int | None = None
        best_p = 0.0
        for s_idx in candidate_sugars:
            p = dist.get(int(s_idx), 0.0)
            if p > best_p:
                best_p = p
                best_s = int(s_idx)
        return (best_s, best_p) if best_p >= 0.35 else (None, 0.0)

    def gt_edge_score(
        self,
        cluster_id: int,
        glycan: Glycan,
        edge_idx: int,
        aa_seq: bytes | None = None,
    ) -> tuple[float, bool]:
        """Compute factorized 1D-CNN cost bonus in `[-0.25, +0.06]` and consensus flag for `(cluster_id, edge_idx)`."""
        if not (0 <= edge_idx < len(glycan.edges)):
            return 0.0, False
        edge = glycan.edges[edge_idx]
        s_res = glycan.nodes[edge.source]
        t_res = glycan.nodes[edge.target]
        if s_res.is_modifier or t_res.is_modifier:
            return 0.0, False

        link = edge.edge_type
        donor_code = encode_donor_residue_chem(s_res)
        acc_code = encode_donor_residue_chem(t_res)
        acc_motif_code = extract_acceptor_motif_for_edge(glycan, edge_idx)
        link_code = int(link.chemical_int)
        dl_code = encode_donor_linkage_label(s_res, link)
        bb_code = 1 if link.is_backbone else 0
        _, _, _, mech = _edge_signatures(glycan, edge_idx)

        def _score_single_seq(seq_cand: bytes | None, cid_cand: int) -> tuple[float, bool]:
            dist_dl = self._get_target_dist("donor_linkage", cid_cand, aa_seq=seq_cand)
            dist_d = self._get_target_dist("donor", cid_cand, aa_seq=seq_cand)
            dist_l = self._get_target_dist("linkage", cid_cand, aa_seq=seq_cand)
            dist_acc = self._get_target_dist("acceptor_sugar", cid_cand, aa_seq=seq_cand)
            dist_am = self._get_target_dist("acceptor_motif", cid_cand, aa_seq=seq_cand)
            dist_bb = self._get_target_dist("is_backbone", cid_cand, aa_seq=seq_cand)
            dist_m = self._get_target_dist("mechanism", cid_cand, aa_seq=seq_cand)

            if not dist_d and not dist_l and not dist_dl:
                return 0.0, False

            p_dl = dist_dl.get(dl_code, 0.0)
            p_d = dist_d.get(donor_code, 0.0)
            p_l = dist_l.get(link_code, 0.0)
            p_acc = dist_acc.get(acc_code, 0.0)
            p_am = dist_am.get(acc_motif_code, 0.0)
            p_bb = dist_bb.get(bb_code, 0.5)
            p_m = dist_m.get(int(mech), 0.0) if mech in (1, 2) else 0.5

            bonus_val = -(
                0.10 * p_dl
                + 0.05 * p_d
                + 0.05 * p_l
                + 0.025 * p_acc
                + 0.025 * p_am
                + 0.02 * max(0.0, p_bb - 0.5)
                + 0.03 * max(0.0, p_m - 0.5)
            )
            if dist_m and mech in (1, 2) and p_m < 0.25:
                bonus_val += 0.04 * (1.0 - p_m)

            strong_flag = False
            if p_dl >= 0.42 or (p_d >= 0.50 and p_l >= 0.45 and p_m >= 0.65):
                competing = 0
                for k_o, e_o in enumerate(glycan.edges):
                    if k_o == edge_idx:
                        continue
                    so = glycan.nodes[e_o.source]
                    to = glycan.nodes[e_o.target]
                    if so.is_modifier or to.is_modifier:
                        continue
                    dl_o = encode_donor_linkage_label(so, e_o.edge_type)
                    am_o = extract_acceptor_motif_for_edge(glycan, k_o)
                    if dl_o == dl_code and am_o == acc_motif_code:
                        continue
                    p_dl_o = dist_dl.get(dl_o, 0.0)
                    p_d_o = dist_d.get(encode_donor_residue_chem(so), 0.0)
                    p_l_o = dist_l.get(int(e_o.edge_type.chemical_int), 0.0)
                    p_am_o = dist_am.get(am_o, 0.0)
                    if (
                        (p_dl_o >= 0.75 * max(p_dl, 1e-6))
                        and (p_d_o * p_l_o >= 0.75 * max(p_d * p_l, 1e-6))
                        and (p_am <= 0.15 or p_am_o >= 0.80 * p_am)
                    ):
                        competing += 1
                if competing == 0:
                    strong_flag = True
            return float(np.clip(bonus_val, -0.25, 0.06)), strong_flag

        best_bonus, has_strong_clf = _score_single_seq(aa_seq, cluster_id)
        if aa_seq is not None and len(aa_seq) >= 550:
            doms = split_bifunctional_gt_domains(aa_seq, min_len=550, overlap=25)
            if doms is not None:
                for d_idx, d_seq in enumerate(doms):
                    d_cid = (cluster_id + (400000 if d_idx == 0 else 500000)) if cluster_id > 0 else 0
                    b_d, s_d = _score_single_seq(d_seq, d_cid)
                    if b_d < best_bonus:
                        best_bonus = b_d
                    if s_d:
                        has_strong_clf = True

        return best_bonus, has_strong_clf
