r"""Sequence clustering algorithms, data structures, and Numba kernels.

Provides high-performance sequence clustering ([`SequenceClusterer`][kbank.loci.cluster.SequenceClusterer])
and structured cluster results ([`SequenceClusters`][kbank.loci.cluster.SequenceClusters]) combining
sparse syncmer-linked randstrobe pre-filtering, parallel banded Smith-Waterman-Gotoh alignment,
and sub-millisecond Numba greedy set-cover / CD-HIT clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit

from kaptive.core.kmers import RandstrobeIndex
from kaptive.core.pairwise import PairwiseAligner
from kaptive.core.seq import Sequences


# Data Containers ------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class SequenceClusters:
    r"""Data-oriented Structure-of-Arrays (SoA) container for sequence clustering results.

    Stores sequence cluster assignments, centroid representative indices, and cluster sizes
    as contiguous 1D NumPy arrays.

    Attributes:
        cluster_ids: 1D array mapping sequence index $i \in [0, N-1]$ to cluster ID $c \in [0, C-1]$.
        representative_indices: 1D array of sequence indices representing the centroid of each cluster.
        sizes: 1D array of member counts for each cluster.
        n_clusters: Total number of clusters $C$.
        n_seqs: Total number of clustered sequences $N$.
    """

    cluster_ids: np.ndarray[tuple[int], np.dtype[np.int32]]
    representative_indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    sizes: np.ndarray[tuple[int], np.dtype[np.int32]]
    n_clusters: int
    n_seqs: int

    def __len__(self) -> int:
        r"""Return the total number of clusters."""
        return self.n_clusters

    def get_members(self, cluster_id: int) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Return 1D array of original sequence indices assigned to the specified cluster ID.

        Args:
            cluster_id: 0-based integer cluster identifier.

        Returns:
            np.ndarray: Indices of sequences belonging to this cluster.
        """
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise IndexError(f"Cluster ID {cluster_id} out of bounds [0, {self.n_clusters - 1}]")
        return np.flatnonzero(self.cluster_ids == cluster_id).astype(np.int32)

    def representatives(self, seqs: Sequences) -> Sequences:
        r"""Extract the representative centroid sequences from the source collection.

        Args:
            seqs: Source [`Sequences`][kaptive.core.seq.Sequences] collection.

        Returns:
            Sequences: Sliced collection containing only cluster centroids.
        """
        if len(seqs) != self.n_seqs:
            raise ValueError(f"Sequence length mismatch: expected {self.n_seqs}, got {len(seqs)}")
        return seqs[self.representative_indices]  # type: ignore[no-any-return]

    @classmethod
    def empty(cls) -> SequenceClusters:
        r"""Construct an empty [`SequenceClusters`][kbank.loci.cluster.SequenceClusters] instance."""
        return cls(
            cluster_ids=np.empty(0, dtype=np.int32),
            representative_indices=np.empty(0, dtype=np.int32),
            sizes=np.empty(0, dtype=np.int32),
            n_clusters=0,
            n_seqs=0,
        )


# Clustering Engine ---------------------------------------------------------------------------------------------------
class SequenceClusterer:
    r"""High-performance sequence clustering engine.

    Combines sparse syncmer-linked randstrobe pre-filtering ([`RandstrobeIndex`][kaptive.core.kmers.RandstrobeIndex]),
    SIMD-parallel banded Smith-Waterman-Gotoh alignment ([`PairwiseAligner`][kaptive.core.pairwise.PairwiseAligner]),
    and compiled Numba greedy clustering algorithms to deduplicate biological sequences.

    Modes:
        - `"cdhit"` (default): Length-descending greedy incremental clustering (CD-HIT style).
          Guarantees that full-length proteins are selected as cluster representatives rather than fragments.
        - `"greedy_cover"`: Iterative maximum-uncovered-degree greedy set cover.
        - `"scipy_average"`: Hierarchical average-linkage (UPGMA) clustering via SciPy.
        - `"connected_components"`: Transitive closure core components.
    """

    def __init__(
        self,
        min_identity: float = 80.0,
        min_coverage: float = 0.0,
        min_score: int = 2,
        k: int = 10,
        s: int = 5,
        aligner_k: int = 20,
        mode: Literal["cdhit", "greedy_cover", "scipy_average", "connected_components"] = "cdhit",
    ) -> None:
        r"""Initialise sequence clusterer parameters.

        Args:
            min_identity: Minimum pairwise sequence percent identity threshold $[0.0, 100.0]$. Defaults to 80.0.
            min_coverage: Minimum alignment coverage fraction $[0.0, 1.0]$ on both query and target. Defaults to 0.0.
            min_score: Minimum shared randstrobe hash hits for alignment candidate seed filtering. Defaults to 2.
            k: Strobemer seed length. Defaults to 10.
            s: Strobemer sampling stride. Defaults to 5.
            aligner_k: Bandwidth parameter ($2k + 1$ diagonals) for banded Gotoh alignment. Defaults to 20.
            mode: Clustering algorithm mode. Defaults to `"cdhit"`.
        """
        valid_modes = ("cdhit", "greedy_cover", "scipy_average", "connected_components")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of {valid_modes}")

        self.min_identity = float(min_identity)
        self.min_coverage = float(min_coverage)
        self.min_score = int(min_score)
        self.k = int(k)
        self.s = int(s)
        self.aligner_k = int(aligner_k)
        self.mode = mode
        self._aligner = PairwiseAligner(k=self.aligner_k)

    def cluster(self, seqs: Sequences) -> SequenceClusters:
        r"""Cluster an input batch of sequences into non-redundant representative groups.

        Args:
            seqs: Input collection of sequences to cluster.

        Returns:
            SequenceClusters: Structured cluster assignments and centroid representatives.
        """
        n = len(seqs)
        if n == 0:
            return SequenceClusters.empty()

        if n == 1:
            return SequenceClusters(
                cluster_ids=np.zeros(1, dtype=np.int32),
                representative_indices=np.zeros(1, dtype=np.int32),
                sizes=np.ones(1, dtype=np.int32),
                n_clusters=1,
                n_seqs=1,
            )

        # 1. Sparse Candidate Pre-Filtering via RandstrobeIndex
        target_idx = RandstrobeIndex.build(seqs, k=self.k, s=self.s, sort_by_hash=True)
        query_idx = RandstrobeIndex.build(seqs, k=self.k, s=self.s, sort_by_hash=False)

        seeds = target_idx.hits(query_idx, min_score=self.min_score)
        if len(seeds) == 0:
            # All sequences are isolated singletons
            return SequenceClusters(
                cluster_ids=np.arange(n, dtype=np.int32),
                representative_indices=np.arange(n, dtype=np.int32),
                sizes=np.ones(n, dtype=np.int32),
                n_clusters=n,
                n_seqs=n,
            )

        # Retain upper triangle only (query_idx < target_idx) to avoid redundant alignments
        upper_mask = seeds.query_indices < seeds.target_indices
        candidate_seeds = seeds.filter(upper_mask)

        if len(candidate_seeds) == 0:
            return SequenceClusters(
                cluster_ids=np.arange(n, dtype=np.int32),
                representative_indices=np.arange(n, dtype=np.int32),
                sizes=np.ones(n, dtype=np.int32),
                n_clusters=n,
                n_seqs=n,
            )

        # 2. Parallel Banded Smith-Waterman-Gotoh Alignment
        alignments = self._aligner.align_seeds(seqs, seqs, candidate_seeds)
        pidents = alignments.pidents

        # 3. Filter Matches by Identity and Coverage Thresholds
        valid = pidents >= self.min_identity
        if self.min_coverage > 0.0:
            q_lens = seqs.lengths[candidate_seeds.query_indices]
            t_lens = seqs.lengths[candidate_seeds.target_indices]
            q_cov = (alignments.q_ends - alignments.q_starts) / np.maximum(q_lens, 1)
            t_cov = (alignments.t_ends - alignments.t_starts) / np.maximum(t_lens, 1)
            valid &= (q_cov >= self.min_coverage) & (t_cov >= self.min_coverage)

        q_valid = candidate_seeds.query_indices[valid].astype(np.int32)
        t_valid = candidate_seeds.target_indices[valid].astype(np.int32)
        p_valid = pidents[valid]

        # 4. Clustering Execution
        if self.mode == "cdhit":
            degree = np.zeros(n, dtype=np.int32)
            np.add.at(degree, q_valid, 1)
            np.add.at(degree, t_valid, 1)
            # Sort descending by sequence length, break ties by degree descending
            order = np.lexsort((-degree, -seqs.lengths)).astype(np.int32)
            c_ids, reps, sizes = _numba_cdhit_cluster(n, order, q_valid, t_valid)

        elif self.mode == "greedy_cover":
            c_ids, reps, sizes = _numba_greedy_cover_cluster(n, q_valid, t_valid)

        elif self.mode == "scipy_average":
            c_ids, reps, sizes = _scipy_average_cluster(n, seqs.lengths, q_valid, t_valid, p_valid, self.min_identity)

        elif self.mode == "connected_components":
            c_ids, reps, sizes = _connected_components_cluster(n, seqs.lengths, q_valid, t_valid)

        else:
            raise RuntimeError(f"Unhandled mode: {self.mode}")

        return SequenceClusters(
            cluster_ids=c_ids,
            representative_indices=reps,
            sizes=sizes,
            n_clusters=len(reps),
            n_seqs=n,
        )


# Kernels & Algorithms -------------------------------------------------------------------------------------------------
@njit(nogil=True)
def _numba_cdhit_cluster(
    n_seqs: int,
    order: np.ndarray,
    q_indices: np.ndarray,
    t_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Numba kernel for length-prioritized greedy incremental clustering (CD-HIT style)."""
    n_edges = len(q_indices)
    degree = np.zeros(n_seqs, dtype=np.int32)
    for p in range(n_edges):
        degree[q_indices[p]] += 1
        degree[t_indices[p]] += 1

    offsets = np.zeros(n_seqs + 1, dtype=np.int32)
    for i in range(n_seqs):
        offsets[i + 1] = offsets[i] + degree[i]

    adj = np.empty(offsets[n_seqs], dtype=np.int32)
    cur_pos = offsets[:-1].copy()
    for p in range(n_edges):
        u = q_indices[p]
        v = t_indices[p]
        adj[cur_pos[u]] = v
        cur_pos[u] += 1
        adj[cur_pos[v]] = u
        cur_pos[v] += 1

    cluster_ids = np.full(n_seqs, -1, dtype=np.int32)
    reps_list: list[int] = []
    sizes_list: list[int] = []

    for idx in range(n_seqs):
        i = order[idx]
        if cluster_ids[i] != -1:
            continue
        c_id = len(reps_list)
        reps_list.append(i)
        cluster_ids[i] = c_id
        cluster_size = 1

        for edge_idx in range(offsets[i], offsets[i + 1]):
            neighbor = adj[edge_idx]
            if cluster_ids[neighbor] == -1:
                cluster_ids[neighbor] = c_id
                cluster_size += 1

        sizes_list.append(cluster_size)

    reps_arr = np.empty(len(reps_list), dtype=np.int32)
    for k in range(len(reps_list)):
        reps_arr[k] = reps_list[k]

    sizes_arr = np.empty(len(sizes_list), dtype=np.int32)
    for k in range(len(sizes_list)):
        sizes_arr[k] = sizes_list[k]

    return cluster_ids, reps_arr, sizes_arr


@njit(nogil=True)
def _numba_greedy_cover_cluster(
    n_seqs: int,
    q_indices: np.ndarray,
    t_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Numba kernel for iterative maximum uncovered degree greedy set cover."""
    n_edges = len(q_indices)
    degree = np.zeros(n_seqs, dtype=np.int32)
    for p in range(n_edges):
        degree[q_indices[p]] += 1
        degree[t_indices[p]] += 1

    offsets = np.zeros(n_seqs + 1, dtype=np.int32)
    for i in range(n_seqs):
        offsets[i + 1] = offsets[i] + degree[i]

    adj = np.empty(offsets[n_seqs], dtype=np.int32)
    cur_pos = offsets[:-1].copy()
    for p in range(n_edges):
        u = q_indices[p]
        v = t_indices[p]
        adj[cur_pos[u]] = v
        cur_pos[u] += 1
        adj[cur_pos[v]] = u
        cur_pos[v] += 1

    cluster_ids = np.full(n_seqs, -1, dtype=np.int32)
    reps_list: list[int] = []
    sizes_list: list[int] = []

    uncovered = np.ones(n_seqs, dtype=np.int32) + degree
    unassigned_count = n_seqs

    while unassigned_count > 0:
        best_u = -1
        max_cov = -1
        for i in range(n_seqs):
            if cluster_ids[i] == -1 and uncovered[i] > max_cov:
                max_cov = uncovered[i]
                best_u = i

        if best_u == -1:
            break

        c_id = len(reps_list)
        reps_list.append(best_u)
        cluster_ids[best_u] = c_id
        cluster_size = 1
        unassigned_count -= 1

        for edge_idx in range(offsets[best_u], offsets[best_u + 1]):
            nbr = adj[edge_idx]
            uncovered[nbr] -= 1

        for edge_idx in range(offsets[best_u], offsets[best_u + 1]):
            nbr = adj[edge_idx]
            if cluster_ids[nbr] == -1:
                cluster_ids[nbr] = c_id
                cluster_size += 1
                unassigned_count -= 1
                for e2 in range(offsets[nbr], offsets[nbr + 1]):
                    w = adj[e2]
                    uncovered[w] -= 1

        sizes_list.append(cluster_size)

    reps_arr = np.empty(len(reps_list), dtype=np.int32)
    for k in range(len(reps_list)):
        reps_arr[k] = reps_list[k]

    sizes_arr = np.empty(len(sizes_list), dtype=np.int32)
    for k in range(len(sizes_list)):
        sizes_arr[k] = sizes_list[k]

    return cluster_ids, reps_arr, sizes_arr


def _scipy_average_cluster(
    n_seqs: int,
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    pidents: np.ndarray[tuple[int], np.dtype[np.float64]],
    min_identity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Hierarchical average-linkage (UPGMA) clustering via SciPy."""
    from scipy.cluster.hierarchy import fcluster, linkage

    if n_seqs <= 1:
        return np.zeros(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.array([n_seqs], dtype=np.int32)

    total_pairs = n_seqs * (n_seqs - 1) // 2
    condensed_d = np.ones(total_pairs, dtype=np.float64)

    if len(q_indices) > 0:
        q = q_indices.astype(np.int64)
        t = t_indices.astype(np.int64)
        swap = q > t
        if np.any(swap):
            q_copy = q.copy()
            q[swap] = t[swap]
            t[swap] = q_copy[swap]

        indices = n_seqs * q - (q * (q + 1)) // 2 + t - q - 1
        dists = np.clip(1.0 - (pidents / 100.0), 0.0, 1.0)
        condensed_d[indices] = dists

    z = linkage(condensed_d, method="average")
    cutoff = max(1e-6, 1.0 - (min_identity / 100.0))
    raw_clusters = fcluster(z, t=cutoff, criterion="distance")

    unique_c, inv = np.unique(raw_clusters, return_inverse=True)
    cluster_ids = inv.astype(np.int32)
    n_clusters = len(unique_c)

    reps = np.empty(n_clusters, dtype=np.int32)
    sizes = np.bincount(cluster_ids, minlength=n_clusters).astype(np.int32)

    order = np.argsort(-lengths)
    seen = np.zeros(n_clusters, dtype=np.bool_)
    for idx in order:
        c = cluster_ids[idx]
        if not seen[c]:
            seen[c] = True
            reps[c] = idx

    return cluster_ids, reps, sizes


def _connected_components_cluster(
    n_seqs: int,
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Transitive closure connected components clustering via SciPy."""
    from scipy.sparse import csgraph, csr_array

    if n_seqs <= 1:
        return np.zeros(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.array([n_seqs], dtype=np.int32)

    if len(q_indices) == 0:
        return np.arange(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.ones(n_seqs, dtype=np.int32)

    data = np.ones(len(q_indices), dtype=np.int8)
    adj = csr_array((data, (q_indices, t_indices)), shape=(n_seqs, n_seqs))
    n_components, labels = csgraph.connected_components(adj, directed=False)

    cluster_ids = labels.astype(np.int32)
    sizes = np.bincount(cluster_ids, minlength=n_components).astype(np.int32)

    order = np.argsort(-lengths)
    reps = np.empty(n_components, dtype=np.int32)
    seen = np.zeros(n_components, dtype=np.bool_)
    for idx in order:
        c = cluster_ids[idx]
        if not seen[c]:
            seen[c] = True
            reps[c] = idx

    return cluster_ids, reps, sizes
