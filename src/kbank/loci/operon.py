"""Syntenic operon block partitioning for bacterial genomic loci.

Groups contiguous, co-directional genes into atomic operon blocks based on transcriptional
polarity (strand) and intergenic distance constraints (tau_gap <= 3.5 kb).
These atomic blocks serve as the fundamental indivisible units or change-point candidates
for locus boundary disambiguation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OperonBlock:
    """A contiguous, co-directional syntenic operon block of genes.

    Attributes:
        start: Genomic start coordinate of the first gene in the block (inclusive).
        end: Genomic end coordinate of the last gene in the block (exclusive).
        strand: Transcriptional strand (+1 for forward, -1 for reverse).
        gene_indices: 1D int32 array of ORF indices belonging to this block.
        intergenic_gaps: 1D int32 array of intergenic distances between adjacent genes (length = N - 1).
    """

    start: int
    end: int
    strand: int
    gene_indices: np.ndarray
    intergenic_gaps: np.ndarray

    @property
    def num_genes(self) -> int:
        """Number of genes in this operon block."""
        return len(self.gene_indices)

    @property
    def span(self) -> int:
        """Total genomic span of this operon in base pairs."""
        return self.end - self.start

    def __repr__(self) -> str:
        """String representation of the operon block."""
        return (
            f"OperonBlock(start={self.start:,}, end={self.end:,}, strand={self.strand:+d}, "
            f"genes={self.num_genes}, span={self.span:,} bp)"
        )


def partition_into_operon_blocks(
    starts: Sequence[int] | np.ndarray,
    ends: Sequence[int] | np.ndarray,
    strands: Sequence[int] | np.ndarray,
    max_gap: int = 3500,
) -> list[OperonBlock]:
    """Partitions a chromosomal sequence of genes into contiguous syntenic operon blocks.

    A new operon block is initiated whenever:
    1. Transcriptional strand changes (e.g. +1 to -1).
    2. Intergenic distance between successive genes exceeds max_gap (default 3,500 bp).
    3. Contig coordinate wraps or resets.

    Args:
        starts: 1D array-like of gene start coordinates.
        ends: 1D array-like of gene end coordinates.
        strands: 1D array-like of gene strands (+1 or -1).
        max_gap: Maximum allowed intergenic distance in base pairs before breaking an operon.

    Returns:
        list[OperonBlock]: Sequence of atomic operon blocks across the genome.
    """
    n_genes = len(starts)
    if n_genes == 0:
        return []

    starts_arr = np.asarray(starts, dtype=np.int64)
    ends_arr = np.asarray(ends, dtype=np.int64)
    strands_arr = np.asarray(strands, dtype=np.int32)

    blocks: list[OperonBlock] = []

    curr_indices = [0]
    curr_strand = int(strands_arr[0])
    curr_gaps: list[int] = []

    for i in range(1, n_genes):
        prev_end = ends_arr[i - 1]
        this_start = starts_arr[i]
        this_strand = int(strands_arr[i])

        gap = int(this_start - prev_end)

        # Condition for breaking operon: strand change or gap exceeds threshold
        is_strand_switch = this_strand != curr_strand
        is_gap_exceeded = gap > max_gap or gap < -100_000  # negative gap > 100kb implies new contig

        if is_strand_switch or is_gap_exceeded:
            # Finalize current block
            idx_arr = np.array(curr_indices, dtype=np.int32)
            gap_arr = np.array(curr_gaps, dtype=np.int32)
            block_start = int(starts_arr[curr_indices[0]])
            block_end = int(ends_arr[curr_indices[-1]])

            blocks.append(
                OperonBlock(
                    start=block_start,
                    end=block_end,
                    strand=curr_strand,
                    gene_indices=idx_arr,
                    intergenic_gaps=gap_arr,
                )
            )

            # Start new block
            curr_indices = [i]
            curr_strand = this_strand
            curr_gaps = []
        else:
            curr_indices.append(i)
            curr_gaps.append(gap)

    # Finalize last block
    if curr_indices:
        idx_arr = np.array(curr_indices, dtype=np.int32)
        gap_arr = np.array(curr_gaps, dtype=np.int32)
        block_start = int(starts_arr[curr_indices[0]])
        block_end = int(ends_arr[curr_indices[-1]])

        blocks.append(
            OperonBlock(
                start=block_start,
                end=block_end,
                strand=curr_strand,
                gene_indices=idx_arr,
                intergenic_gaps=gap_arr,
            )
        )

    return blocks
