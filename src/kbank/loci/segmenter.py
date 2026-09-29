"""Architectural Change-Point Dynamic Programming (ACP-DP) locus segmentation engine.

Disambiguates bacterial chromosome gene sequences into discrete, non-overlapping,
functionally complete biosynthetic gene clusters (BGCs).
Cleanly resolves adjacent and tandem loci (e.g. K-locus and O-locus separated by <= 3 kb)
without chimeric merging by enforcing mutual exclusivity and architectural invariants.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kbank.genes.roles import GeneRoles
from kbank.loci.archetypes import Archetypes, LocusArchetype, score_archetype_likelihood


@dataclass(frozen=True, slots=True)
class LocusSegment:
    """A segmented, discrete biosynthetic gene cluster (BGC) locus.

    Attributes:
        start: Genomic start coordinate on chromosome (0-based inclusive).
        end: Genomic end coordinate on chromosome (0-based exclusive).
        strand: Predominant transcriptional strand (+1 for forward, -1 for reverse).
        archetype: Canonical machinery archetype classification.
        completeness: Continuous probabilistic completeness score in [0.0, 1.0].
        gene_indices: 1D int32 array of ORF indices belonging to this locus.
        roles: 1D int32 array of predicted role indices for each gene.
        confidences: 1D float32 array of prediction confidences for each gene.
        is_tandem_split: True if this locus was disambiguated from an adjacent tandem locus.
        score: Dynamic programming segment score.
    """

    start: int
    end: int
    strand: int
    archetype: LocusArchetype
    completeness: float
    gene_indices: np.ndarray
    roles: np.ndarray
    confidences: np.ndarray
    is_tandem_split: bool
    score: float

    @property
    def num_genes(self) -> int:
        """Number of genes in this locus."""
        return len(self.gene_indices)

    @property
    def span(self) -> int:
        """Total genomic span in base pairs."""
        return self.end - self.start

    def __repr__(self) -> str:
        """String representation of the locus segment."""
        return (
            f"LocusSegment(archetype={self.archetype.name}, span={self.start:,}..{self.end:,} bp "
            f"({self.span:,} bp), strand={self.strand:+d}, genes={self.num_genes}, "
            f"completeness={self.completeness:.1%}, tandem_split={self.is_tandem_split})"
        )


class LocusSegmenter:
    """Architectural Change-Point Dynamic Programming (ACP-DP) segmentation engine."""

    def __init__(
        self,
        min_completeness: float = 0.65,
        w_comp: float = 12.0,
        w_incomp: float = 100.0,
        locus_init_penalty: float = 8.0,
        null_gene_llr: float = -1.5,
        strand_switch_penalty: float = 8.0,
        min_strand_purity: float = 0.85,
        max_genes_per_locus: int = 45,
        max_intergenic_gap: int = 4000,
    ) -> None:
        """Initializes the LocusSegmenter with optimization hyperparameters.

        Args:
            min_completeness: Minimum completeness score [0.0, 1.0] to emit a locus.
            w_comp: Reward multiplier for functional machinery completeness.
            w_incomp: Severe penalty multiplier for conflicting machinery.
            locus_init_penalty: Occam's razor penalty lambda for initiating a new locus.
            null_gene_llr: Log-likelihood penalty for non-biosynthetic spacer genes.
            strand_switch_penalty: Penalty for each transcriptional strand reversal within a locus.
            min_strand_purity: Minimum fraction of genes sharing the dominant strand (default 0.85).
            max_genes_per_locus: Maximum search window in genes for candidate loci.
            max_intergenic_gap: Intergenic distance threshold in bp triggering an automatic cut.
        """
        self.min_completeness = min_completeness
        self.w_comp = w_comp
        self.w_incomp = w_incomp
        self.locus_init_penalty = locus_init_penalty
        self.null_gene_llr = null_gene_llr
        self.strand_switch_penalty = strand_switch_penalty
        self.min_strand_purity = min_strand_purity
        self.max_genes_per_locus = max_genes_per_locus
        self.max_intergenic_gap = max_intergenic_gap

    def segment(
        self,
        starts: Sequence[int] | np.ndarray,
        ends: Sequence[int] | np.ndarray,
        strands: Sequence[int] | np.ndarray,
        probs: np.ndarray,
    ) -> list[LocusSegment]:
        """Segments chromosomal genes into non-overlapping, architecturally consistent loci.

        Args:
            starts: 1D array of gene start coordinates.
            ends: 1D array of gene end coordinates.
            strands: 1D array of gene strands (+1 or -1).
            probs: (N, C) posterior probabilities from GeneRoleClassifier.

        Returns:
            list[LocusSegment]: Disambiguated and classified BGC loci.
        """
        n_genes = len(starts)
        if n_genes == 0:
            return []

        starts_arr = np.asarray(starts, dtype=np.int64)
        ends_arr = np.asarray(ends, dtype=np.int64)
        strands_arr = np.asarray(strands, dtype=np.int32)
        top_roles = np.argmax(probs, axis=1).astype(np.int32)
        top_conf = np.max(probs, axis=1).astype(np.float32)

        # 1. Identify candidate biosynthetic gene clusters / islands
        # An island is a region where multiple biosynthetic genes appear in proximity
        candidate_islands = self._find_candidate_islands(
            starts_arr, ends_arr, top_roles, top_conf
        )

        all_loci: list[LocusSegment] = []

        # 2. Run ACP-DP on each candidate island
        for island_start_idx, island_end_idx in candidate_islands:
            sub_loci = self._segment_island(
                island_start_idx,
                island_end_idx,
                starts_arr,
                ends_arr,
                strands_arr,
                probs,
                top_roles,
                top_conf,
            )
            all_loci.extend(sub_loci)

        # 3. Mark tandem splits (loci separated by <= 8 intervening genes or <= 10 kb)
        if len(all_loci) > 1:
            marked_loci: list[LocusSegment] = []
            for i, loc in enumerate(all_loci):
                is_tandem = False
                if i > 0:
                    prev_loc = all_loci[i - 1]
                    gene_gap = loc.gene_indices[0] - prev_loc.gene_indices[-1] - 1
                    bp_gap = loc.start - prev_loc.end
                    if 0 <= gene_gap <= 8 and 0 <= bp_gap <= 10_000:
                        is_tandem = True
                if i < len(all_loci) - 1:
                    next_loc = all_loci[i + 1]
                    gene_gap = next_loc.gene_indices[0] - loc.gene_indices[-1] - 1
                    bp_gap = next_loc.start - loc.end
                    if 0 <= gene_gap <= 8 and 0 <= bp_gap <= 10_000:
                        is_tandem = True

                if is_tandem != loc.is_tandem_split:
                    loc = LocusSegment(
                        start=loc.start,
                        end=loc.end,
                        strand=loc.strand,
                        archetype=loc.archetype,
                        completeness=loc.completeness,
                        gene_indices=loc.gene_indices,
                        roles=loc.roles,
                        confidences=loc.confidences,
                        is_tandem_split=is_tandem,
                        score=loc.score,
                    )
                marked_loci.append(loc)
            all_loci = marked_loci

        return all_loci

    def _find_candidate_islands(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        roles: np.ndarray,
        confidences: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Identifies chromosomal windows with significant biosynthetic gene density."""
        n = len(starts)
        machinery_roles = {
            GeneRoles[b"FLIPPASE"], GeneRoles[b"POLYMERASE"], GeneRoles[b"COPOLYMERASE"],
            GeneRoles[b"KINASE"], GeneRoles[b"PHOSPHATASE"], GeneRoles[b"ABC_PERMEASE"],
            GeneRoles[b"ABC_ATPASE"], GeneRoles[b"TRANSLOCON"], GeneRoles[b"CAPSULE_ANCHOR"],
            GeneRoles[b"TRANSMEMBRANE_SYNTHASE"], GeneRoles[b"INITIATOR"],
        }

        # 1. Identify core machinery seeds
        seed_idx = [
            i for i in range(n)
            if roles[i] in machinery_roles and confidences[i] > 0.40
        ]
        if not seed_idx:
            return []

        # 2. Group adjacent machinery seeds within 7 genes and 10 kb
        linked: list[list[int]] = []
        curr_group: list[int] = []
        for i in range(len(seed_idx)):
            idx_i = seed_idx[i]
            if not curr_group:
                curr_group = [idx_i]
            else:
                prev_idx = curr_group[-1]
                if idx_i - prev_idx <= 7 and starts[idx_i] - ends[prev_idx] <= 10_000:
                    curr_group.append(idx_i)
                else:
                    if len(curr_group) >= 2:
                        linked.append(curr_group)
                    curr_group = [idx_i]
        if len(curr_group) >= 2:
            linked.append(curr_group)

        # Allow solitary machinery seeds if flanked by multiple biosynthetic genes (>= 4)
        for s in seed_idx:
            left_s = max(0, s - 3)
            right_s = min(n, s + 4)
            bio_count = sum(
                1 for r, c in zip(roles[left_s:right_s], confidences[left_s:right_s])
                if r < 27 and c >= 0.25
            )
            if bio_count >= 4 and not any(s in g for g in linked):
                linked.append([s])

        linked.sort(key=lambda g: g[0])

        # 3. Expand each core with flanking biosynthetic cargo genes
        islands: list[list[int]] = []
        for g in linked:
            s_idx = g[0]
            e_idx = g[-1]
            left = s_idx
            right = e_idx + 1

            # Walk right through contiguous biosynthetic genes
            while (
                right < n
                and roles[right] < 27
                and confidences[right] >= 0.25
                and starts[right] - ends[right - 1] <= self.max_intergenic_gap
                and right - e_idx <= 25
            ):
                right += 1

            # Walk left through contiguous biosynthetic genes
            while (
                left > 0
                and roles[left - 1] < 27
                and confidences[left - 1] >= 0.25
                and starts[left] - ends[left - 1] <= self.max_intergenic_gap
                and s_idx - left <= 25
            ):
                left -= 1

            if not islands or left > islands[-1][1]:
                islands.append([left, right])
            else:
                islands[-1][1] = max(islands[-1][1], right)

        return [(isl[0], isl[1]) for isl in islands]

    def _segment_island(
        self,
        start_idx: int,
        end_idx: int,
        starts: np.ndarray,
        ends: np.ndarray,
        strands: np.ndarray,
        probs: np.ndarray,
        top_roles: np.ndarray,
        top_conf: np.ndarray,
    ) -> list[LocusSegment]:
        """Runs Bellman dynamic programming change-point optimization over an island."""
        M = end_idx - start_idx
        if M < 3:
            return []

        sub_starts = starts[start_idx:end_idx]
        sub_ends = ends[start_idx:end_idx]
        sub_strands = strands[start_idx:end_idx]
        sub_probs = probs[start_idx:end_idx]

        # Precompute boundary relevance matrix for the whole island: shape (M, 6)
        boundary_relevant = Archetypes.is_boundary_relevant(sub_probs)

        # DP tables
        # D[i]: max score for prefix of length i (genes 0..i-1)
        D = np.full(M + 1, -float("inf"), dtype=np.float64)
        D[0] = 0.0

        # Backtrack: backtrack[i] = (best_j, best_archetype, best_comp, best_score)
        backtrack: list[tuple[int, LocusArchetype | None, float, float]] = [
            (0, None, 0.0, 0.0) for _ in range(M + 1)
        ]

        for i in range(1, M + 1):
            # Option 1: Gene i-1 is a spacer (background / intergenic gene)
            spacer_score = D[i - 1] + self.null_gene_llr
            if spacer_score > D[i]:
                D[i] = spacer_score
                backtrack[i] = (i - 1, None, 0.0, 0.0)

            # Option 2: Candidate locus segment from j to i (genes j..i-1)
            min_j = max(0, i - self.max_genes_per_locus)
            for arch_idx in range(len(Archetypes)):
                # Boundary invariant: end boundary gene i-1 must be relevant to archetype
                if not boundary_relevant[i - 1, arch_idx]:
                    continue

                arch = LocusArchetype(arch_idx)

                for j in range(i - 1, min_j - 1, -1):
                    # Boundary invariant: start boundary gene j must be relevant to archetype
                    if not boundary_relevant[j, arch_idx]:
                        continue

                    # Check for physical gap break between j and j+1
                    if j < i - 1:
                        gap = sub_starts[j + 1] - sub_ends[j]
                        if gap > self.max_intergenic_gap:
                            break  # Cannot extend segment backwards across large physical gap

                    seg_len = i - j
                    # Check strand purity and count switches
                    cand_strands = sub_strands[j:i]
                    n_fwd = int(np.sum(cand_strands == 1))
                    n_rev = int(np.sum(cand_strands == -1))
                    dom_count = max(n_fwd, n_rev)
                    strand_purity = dom_count / seg_len

                    # Disallow segments with poor strand consistency
                    if strand_purity < self.min_strand_purity:
                        continue

                    num_switches = int(np.sum(cand_strands[1:] != cand_strands[:-1]))
                    strand_pen = num_switches * self.strand_switch_penalty

                    cand_probs = sub_probs[j:i]
                    seg_score, comp, incomp = score_archetype_likelihood(
                        cand_probs, arch, w_comp=self.w_comp, w_incomp=self.w_incomp
                    )

                    # Total score adding to D[j]
                    total = D[j] + seg_score - self.locus_init_penalty - strand_pen
                    if total > D[i]:
                        D[i] = total
                        backtrack[i] = (j, arch, comp, seg_score)

        # Backtrack to reconstruct optimal segmentation
        curr = M
        segments: list[tuple[int, int, LocusArchetype, float, float]] = []

        while curr > 0:
            prev_j, arch, comp, seg_score = backtrack[curr]
            if arch is not None and comp >= self.min_completeness:
                segments.append((prev_j, curr, arch, comp, seg_score))
            curr = prev_j

        segments.reverse()

        # Build LocusSegment objects
        result: list[LocusSegment] = []
        for local_j, local_i, arch, comp, seg_score in segments:
            global_indices = np.arange(start_idx + local_j, start_idx + local_i, dtype=np.int32)
            seg_starts = sub_starts[local_j:local_i]
            seg_ends = sub_ends[local_j:local_i]
            seg_strands = sub_strands[local_j:local_i]

            # Determine dominant strand
            strand_sum = np.sum(seg_strands)
            dom_strand = 1 if strand_sum >= 0 else -1

            loc_start = int(seg_starts[0])
            loc_end = int(seg_ends[-1])

            seg_roles = top_roles[global_indices]
            seg_conf = top_conf[global_indices]

            result.append(
                LocusSegment(
                    start=loc_start,
                    end=loc_end,
                    strand=dom_strand,
                    archetype=arch,
                    completeness=comp,
                    gene_indices=global_indices,
                    roles=seg_roles,
                    confidences=seg_conf,
                    is_tandem_split=False,
                    score=seg_score,
                )
            )

        return result
