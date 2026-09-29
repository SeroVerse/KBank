r"""OrfPredictor high-level execution engine for sequences and assemblies."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kaptive.core.genome import GenomeAssembly
from kaptive.core.interval import Intervals
from kaptive.core.seq import Sequence, Sequences

from ._constants import LUT_ENC, LUT_RC
from .extract import _extract_genes
from .models import Genes, HMM, _build_indels, _build_sequences, get_hmm_model
from .viterbi import _viterbi_backtrack, _viterbi_forward


class OrfPredictor:
    r"""High-performance Open Reading Frame (ORF) predictor.

    Executes Numba JIT-compiled FragGeneScan HMM decoding over single sequences,
    [`Sequences`][kaptive.core.seq.Sequences], or [`GenomeAssembly`][kaptive.core.genome.GenomeAssembly]
    containers, returning batched [`Genes`][kaptive.bgc.annotate.Genes].

    Attributes:
        whole_genome (bool): Flag indicating complete genome mode (with start codon refinement)
            or fragmented / short-read mode.
        model (HMMModel): Baked HMM model parameter container.
    """

    def __init__(
        self,
        whole_genome: bool = False,
        model: HMM | None = None,
        window_size: int = 50_000,
        overlap: int = 5_000,
    ) -> None:
        r"""Initialize an [`OrfPredictor`][kaptive.bgc.orf.OrfPredictor] instance.

        Args:
            whole_genome (bool): If True, enables whole genome gene finder mode. Defaults to False.
            model (HMMModel | None): Optional custom HMM parameters. If None, uses default baked model.
            window_size (int): Size in base pairs for local GC-adaptive sliding windows on long contigs.
                Set to 0 to disable windowing. Defaults to 50_000.
            overlap (int): Overlap size in base pairs between consecutive windows to ensure seamless
                gene boundary detection. Defaults to 5_000.
        """
        self.whole_genome = whole_genome
        self.model = model if model is not None else get_hmm_model()
        self.window_size = window_size
        self.overlap = overlap

    def _predict_single_chunk(
        self,
        raw_arr: np.ndarray,  # type: ignore[type-arg]
    ) -> tuple[list[int], list[int], list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:  # type: ignore[type-arg]
        if len(raw_arr) < (120 if self.whole_genome else 60):
            return [], [], [], [], [], []

        encoded = LUT_ENC[raw_arr]

        # Calculate GC content bin (0..43)
        c_count = np.sum((encoded == 1) | (encoded == 2))
        cg_pct = (int(c_count) * 100) // len(encoded)
        cg_bin = min(43, max(0, cg_pct - 26))

        m = self.model
        best_st, path = _viterbi_forward(
            encoded,
            cg_bin,
            self.whole_genome,
            m.tr,
            m.tr_mi,
            m.tr_ii,
            m.pi,
            m.tr_rr,
            m.e_match,
            m.tr_start,
            m.tr_stop,
            m.dist_start,
            m.dist_stop,
        )
        vpath = _viterbi_backtrack(best_st, path)

        return _extract_genes(
            raw_arr,
            encoded,
            vpath,
            cg_bin,
            self.whole_genome,
            m.tr_start,
            m.tr_stop,
            LUT_RC,
        )

    def _predict_windowed(
        self,
        raw_arr: np.ndarray,  # type: ignore[type-arg]
        parallel: bool = False,
    ) -> tuple[list[int], list[int], list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:  # type: ignore[type-arg]
        total_len = len(raw_arr)
        w_size = self.window_size
        overlap = self.overlap
        step = w_size - overlap
        margin = overlap // 2

        windows: list[tuple[int, int]] = []
        for c_start in range(0, total_len, step):
            c_end = min(c_start + w_size, total_len)
            windows.append((c_start, c_end))

        def _process_window(
            win: tuple[int, int],
        ) -> tuple[list[int], list[int], list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
            c_start, c_end = win
            sub = raw_arr[c_start:c_end]
            st, en, strd, cds, ipos, ikind = self._predict_single_chunk(sub)
            left_cutoff = 0 if c_start == 0 else c_start + margin
            right_cutoff = total_len if c_end == total_len else c_end - margin

            w_starts: list[int] = []
            w_ends: list[int] = []
            w_strands: list[int] = []
            w_cds: list[np.ndarray] = []
            w_ipos: list[np.ndarray] = []
            w_ikind: list[np.ndarray] = []

            for s, e, strand, seq, ip, ik in zip(st, en, strd, cds, ipos, ikind):
                g_s = c_start + s
                g_e = c_start + e
                mid = (g_s + g_e) // 2
                if left_cutoff <= mid < right_cutoff:
                    w_starts.append(g_s)
                    w_ends.append(g_e)
                    w_strands.append(strand)
                    w_cds.append(seq)
                    if len(ip) > 0:
                        w_ipos.append(ip + c_start)
                        w_ikind.append(ik)
                    else:
                        w_ipos.append(ip)
                        w_ikind.append(ik)

            return w_starts, w_ends, w_strands, w_cds, w_ipos, w_ikind

        if parallel and len(windows) > 1:
            n_workers = min(len(windows), min(32, (os.cpu_count() or 1) + 4))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                chunk_results = list(executor.map(_process_window, windows))
        else:
            chunk_results = [_process_window(w) for w in windows]

        all_starts: list[int] = []
        all_ends: list[int] = []
        all_strands: list[int] = []
        all_cds: list[np.ndarray] = []
        all_ipos: list[np.ndarray] = []
        all_ikind: list[np.ndarray] = []

        for st, en, strd, cds, ip, ik in chunk_results:
            all_starts.extend(st)
            all_ends.extend(en)
            all_strands.extend(strd)
            all_cds.extend(cds)
            all_ipos.extend(ip)
            all_ikind.extend(ik)

        return all_starts, all_ends, all_strands, all_cds, all_ipos, all_ikind

    def _predict_single(
        self,
        raw_arr: np.ndarray,  # type: ignore[type-arg]
        parallel: bool = False,
    ) -> tuple[list[int], list[int], list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:  # type: ignore[type-arg]
        if 0 < self.window_size < len(raw_arr):
            return self._predict_windowed(raw_arr, parallel=parallel)
        return self._predict_single_chunk(raw_arr)

    def predict(
        self,
        target: bytes | str | Sequence | Sequences | GenomeAssembly,
    ) -> Genes:
        r"""Predict Open Reading Frames across target sequences.

        Args:
            target (bytes | str | Sequence | Sequences | GenomeAssembly): Input genomic sequence(s).

        Returns:
            Genes: Batched [`Genes`][kaptive.bgc.annotate.Genes] container containing interval coordinates,
                in-frame coding sequences (5' -> 3'), and contig assignments.

        Raises:
            TypeError: If target is not a supported sequence type.
        """
        if isinstance(target, (str, bytes)):
            raw_arr = np.frombuffer(target.encode("ascii") if isinstance(target, str) else target, dtype=np.uint8)
            starts, ends, strands, cds_list, ipos, ikind = self._predict_single(raw_arr, parallel=True)
            if not starts:
                return Genes.empty()
            return Genes(
                intervals=Intervals(
                    starts=np.array(starts, dtype=np.int32),
                    ends=np.array(ends, dtype=np.int32),
                    strands=np.array(strands, dtype=np.int8),
                ),
                seqs=_build_sequences(cds_list),
                contig_indices=np.zeros(len(starts), dtype=np.uint32),
                indels=_build_indels(ipos, ikind),
            )

        if isinstance(target, Sequence):
            raw_arr = np.frombuffer(target.seq, dtype=np.uint8)
            starts, ends, strands, cds_list, ipos, ikind = self._predict_single(raw_arr, parallel=True)
            if not starts:
                return Genes.empty()
            return Genes(
                intervals=Intervals(
                    starts=np.array(starts, dtype=np.int32),
                    ends=np.array(ends, dtype=np.int32),
                    strands=np.array(strands, dtype=np.int8),
                ),
                seqs=_build_sequences(cds_list),
                contig_indices=np.zeros(len(starts), dtype=np.uint32),
                indels=_build_indels(ipos, ikind),
            )

        if isinstance(target, (Sequences, GenomeAssembly)):
            num_contigs = len(target)
            if num_contigs == 0:
                return Genes.empty()

            offsets = target.offsets
            lengths = target.lengths
            flat_seqs = target.seqs

            # Sequential fast-path for a single contig
            if num_contigs == 1:
                c_len = int(lengths[0])
                if c_len < 3:
                    return Genes.empty()
                c_start = int(offsets[0])
                raw_contig = flat_seqs[c_start : c_start + c_len]
                st, en, strd, cds, ipos, ikind = self._predict_single(raw_contig, parallel=True)
                if not st:
                    return Genes.empty()
                return Genes(
                    intervals=Intervals(
                        starts=np.array(st, dtype=np.int32),
                        ends=np.array(en, dtype=np.int32),
                        strands=np.array(strd, dtype=np.int8),
                    ),
                    seqs=_build_sequences(cds),
                    contig_indices=np.zeros(len(st), dtype=np.uint32),
                    indels=_build_indels(ipos, ikind),
                )

            # Multi-contig batch: partition into balanced chunks
            n_workers = min(num_contigs, min(32, (os.cpu_count() or 1) + 4))
            chunks = np.array_split(np.arange(num_contigs, dtype=np.int32), n_workers)

            def _chunk_worker(
                chunk_indices: np.ndarray,  # type: ignore[type-arg]
            ) -> tuple[
                list[int],
                list[int],
                list[int],
                list[int],
                list[np.ndarray],  # type: ignore[type-arg]
                list[np.ndarray],  # type: ignore[type-arg]
                list[np.ndarray],  # type: ignore[type-arg]
            ]:
                c_starts: list[int] = []
                c_ends: list[int] = []
                c_strands: list[int] = []
                c_contig_idx: list[int] = []
                c_cds: list[np.ndarray] = []  # type: ignore[type-arg]
                c_ipos: list[np.ndarray] = []  # type: ignore[type-arg]
                c_ikind: list[np.ndarray] = []  # type: ignore[type-arg]

                for idx in chunk_indices:
                    i = int(idx)
                    c_len = int(lengths[i])
                    if c_len < 3:
                        continue
                    c_start = int(offsets[i])
                    raw_contig = flat_seqs[c_start : c_start + c_len]
                    st, en, strd, cds, ipos, ikind = self._predict_single(raw_contig)
                    n_genes = len(st)
                    if n_genes > 0:
                        c_starts.extend(st)
                        c_ends.extend(en)
                        c_strands.extend(strd)
                        c_contig_idx.extend([i] * n_genes)
                        c_cds.extend(cds)
                        c_ipos.extend(ipos)
                        c_ikind.extend(ikind)

                return c_starts, c_ends, c_strands, c_contig_idx, c_cds, c_ipos, c_ikind

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                chunk_results = list(executor.map(_chunk_worker, chunks))

            # Flatten chunk results
            all_starts = []
            all_ends = []
            all_strands = []
            all_contig_indices = []
            all_cds = []
            all_ipos = []
            all_ikind = []

            for st, en, strd, c_idx, cds, ipos, ikind in chunk_results:
                if st:
                    all_starts.extend(st)
                    all_ends.extend(en)
                    all_strands.extend(strd)
                    all_contig_indices.extend(c_idx)
                    all_cds.extend(cds)
                    all_ipos.extend(ipos)
                    all_ikind.extend(ikind)

            if not all_starts:
                return Genes.empty()

            return Genes(
                intervals=Intervals(
                    starts=np.array(all_starts, dtype=np.int32),
                    ends=np.array(all_ends, dtype=np.int32),
                    strands=np.array(all_strands, dtype=np.int8),
                ),
                seqs=_build_sequences(all_cds),
                contig_indices=np.array(all_contig_indices, dtype=np.uint32),
                indels=_build_indels(all_ipos, all_ikind),
            )

        raise TypeError(f"Unsupported target type for ORF prediction: {type(target)}")