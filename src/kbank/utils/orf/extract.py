r"""Numba-accelerated gene coordinate extraction and frameshift indel tracking."""

from __future__ import annotations

import numpy as np
from numba import njit

from ._constants import (
    I1,
    I6,
    M1,
    M4,
    M6,
    E,
    Er,
    I1r,
    I6r,
    M1r,
    M4r,
    M6r,
    R,
    S,
    Sr,
)
from .viterbi import _trinucleotide


@njit(fastmath=True, nogil=True, cache=True)
def _extract_genes(
    raw_seq: np.ndarray,  # type: ignore[type-arg]
    seq: np.ndarray,  # type: ignore[type-arg]
    vpath: np.ndarray,  # type: ignore[type-arg]
    cg_bin: int,
    whole_genome: bool,
    tr_start: np.ndarray,  # type: ignore[type-arg]
    tr_stop: np.ndarray,  # type: ignore[type-arg]
    lut_rc: np.ndarray,  # type: ignore[type-arg]
) -> tuple[list[int], list[int], list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:  # type: ignore[type-arg]
    seq_len = len(seq)
    gene_len = 120 if whole_genome else 60

    starts: list[int] = []
    ends: list[int] = []
    strands: list[int] = []
    cds_list: list[np.ndarray] = [np.empty(0, dtype=np.uint8)][:0]  # type: ignore[type-arg]
    indel_pos_list: list[np.ndarray] = [np.empty(0, dtype=np.int32)][:0]  # type: ignore[type-arg]
    indel_kind_list: list[np.ndarray] = [np.empty(0, dtype=np.int8)][:0]  # type: ignore[type-arg]

    curr_indel_pos: list[int] = [0][:0]
    curr_indel_kind: list[int] = [0][:0]

    codon_start = 0
    start_t = -1
    dna_start_t_withstop = 0
    dna_start_t = 0

    # Temporary flat buffer for gene nucleotides
    dna_buf = np.empty(seq_len * 2 + 100, dtype=np.uint8)
    dna_len = 0
    prev_match = S

    for t in range(seq_len):
        st = vpath[t]

        if codon_start == 0 and start_t < 0 and ((M1 <= st <= M6) or (M1r <= st <= M6r) or st == S or st == Sr):
            dna_start_t_withstop = t + 1
            dna_start_t = t + 1
            start_t = t + 1

        if codon_start == 0 and (st == M1 or st == M4 or st == M1r or st == M4r):
            dna_len = 0
            dna_buf[dna_len] = raw_seq[t]
            dna_len += 1
            dna_start_t_withstop = t + 1
            dna_start_t = t + 1
            curr_indel_pos.clear()
            curr_indel_kind.clear()

            if st == M1r or st == M4r:
                if t > 2:
                    dna_start_t_withstop = t - 2
                    dna_buf[3 : dna_len + 3] = dna_buf[:dna_len]
                    dna_buf[0] = raw_seq[t - 3]
                    dna_buf[1] = raw_seq[t - 2]
                    dna_buf[2] = raw_seq[t - 1]
                    dna_len += 3

            prev_match = st
            codon_start = 1 if st <= M6 else -1

        elif codon_start != 0 and (st == E or st == Er or t == seq_len - 1):
            if st == E or st == Er:
                end_t = t + 3
            else:
                temp_t = t
                while (
                    vpath[temp_t] != M1
                    and vpath[temp_t] != M4
                    and vpath[temp_t] != M1r
                    and vpath[temp_t] != M4r
                    and temp_t > 0
                ):
                    if dna_len > 0:
                        dna_len -= 1
                    temp_t -= 1
                end_t = temp_t

            threshold = gene_len + (3 if whole_genome and codon_start == -1 else 0)
            if dna_len > threshold:
                if codon_start == 1:
                    if start_t == dna_start_t - 3:
                        dna_start_t -= 3
                        dna_buf[3 : dna_len + 3] = dna_buf[:dna_len]
                        dna_buf[0] = raw_seq[dna_start_t - 1]
                        dna_buf[1] = raw_seq[dna_start_t]
                        dna_buf[2] = raw_seq[dna_start_t + 1]
                        dna_len += 3

                    if whole_genome:
                        start_old = start_t
                        s = 0
                        e_save = 0.0
                        s_save = 0
                        while start_old >= 1 + s + 35:
                            c0 = seq[start_old - 1 - s]
                            c1 = seq[start_old - s]
                            c2 = seq[start_old + 1 - s]
                            if (
                                (c0 == 3 and c1 == 0 and c2 == 0)
                                or (c0 == 3 and c1 == 0 and c2 == 2)
                                or (c0 == 3 and c1 == 2 and c2 == 0)
                            ):
                                break
                            if (
                                (c0 == 0 and c1 == 3 and c2 == 2)
                                or (c0 == 2 and c1 == 3 and c2 == 2)
                                or (c0 == 3 and c1 == 3 and c2 == 2)
                            ):
                                utr_start = start_old - 1 - s - 30
                                freq_sum = 0.0
                                for j in range(61):
                                    tr_idx = _trinucleotide(seq, utr_start + j)
                                    freq_sum -= tr_start[0, cg_bin, j, tr_idx]
                                if s == 0 or freq_sum < e_save:
                                    e_save = freq_sum
                                    s_save = s
                            s += 3

                        dna_start_t -= s_save
                        if s_save > 0:
                            dna_buf[s_save : dna_len + s_save] = dna_buf[:dna_len]
                            for idx_s in range(s_save):
                                dna_buf[idx_s] = raw_seq[dna_start_t - 1 + idx_s]
                            dna_len += s_save

                        dna_buf[dna_len] = raw_seq[end_t - 3]
                        dna_buf[dna_len + 1] = raw_seq[end_t - 2]
                        dna_buf[dna_len + 2] = raw_seq[end_t - 1]
                        dna_len += 3

                    starts.append(dna_start_t - 1)  # 0-based coordinate
                    ends.append(end_t)
                    strands.append(1)
                    cds_list.append(dna_buf[:dna_len].copy())
                    if len(curr_indel_pos) > 0:
                        indel_pos_list.append(np.array(curr_indel_pos, dtype=np.int32))
                        indel_kind_list.append(np.array(curr_indel_kind, dtype=np.int8))
                    else:
                        indel_pos_list.append(np.empty(0, dtype=np.int32))
                        indel_kind_list.append(np.empty(0, dtype=np.int8))

                elif codon_start == -1:
                    if whole_genome:
                        end_old = end_t
                        s = 0
                        e_save = 0.0
                        s_save = 0
                        while end_old - 2 + s + 35 < seq_len:
                            c0 = seq[end_old - 3 + s]
                            c1 = seq[end_old - 2 + s]
                            c2 = seq[end_old - 1 + s]
                            if (
                                (c0 == 3 and c1 == 3 and c2 == 0)
                                or (c0 == 1 and c1 == 3 and c2 == 0)
                                or (c0 == 3 and c1 == 1 and c2 == 0)
                            ):
                                break
                            if (
                                (c0 == 1 and c1 == 0 and c2 == 3)
                                or (c0 == 1 and c1 == 0 and c2 == 1)
                                or (c0 == 1 and c1 == 0 and c2 == 0)
                            ):
                                utr_start = end_old - 3 + s - 30
                                freq_sum = 0.0
                                for j in range(61):
                                    tr_idx = _trinucleotide(seq, utr_start + j)
                                    freq_sum -= tr_stop[1, cg_bin, j, tr_idx]
                                if s == 0 or freq_sum < e_save:
                                    e_save = freq_sum
                                    s_save = s
                            s += 3

                        end_t = end_old + s_save
                        dna_buf[dna_len] = raw_seq[end_t - 3]
                        dna_buf[dna_len + 1] = raw_seq[end_t - 2]
                        dna_buf[dna_len + 2] = raw_seq[end_t - 1]
                        dna_len += 3

                    # Reverse complement for reverse strand CDS (5' -> 3')
                    rc_arr = np.empty(dna_len, dtype=np.uint8)
                    for k in range(dna_len):
                        rc_arr[k] = lut_rc[dna_buf[dna_len - 1 - k]]

                    starts.append(dna_start_t_withstop - 1)  # 0-based coordinate
                    ends.append(end_t)
                    strands.append(-1)
                    cds_list.append(rc_arr)
                    if len(curr_indel_pos) > 0:
                        indel_pos_list.append(np.array(curr_indel_pos, dtype=np.int32))
                        indel_kind_list.append(np.array(curr_indel_kind, dtype=np.int8))
                    else:
                        indel_pos_list.append(np.empty(0, dtype=np.int32))
                        indel_kind_list.append(np.empty(0, dtype=np.int8))

            codon_start = 0
            start_t = -1
            dna_len = 0
            curr_indel_pos.clear()
            curr_indel_kind.clear()

        elif codon_start != 0 and (
            ((M1 <= st <= M6) and prev_match >= M1) or ((M1r <= st <= M6r) and prev_match >= M1r)
        ):
            if st < prev_match:
                out_nt = st + 6 - prev_match
            else:
                out_nt = st - prev_match
            for kk in range(out_nt):
                dna_buf[dna_len] = ord("N")
                dna_len += 1
                if kk > 0:
                    curr_indel_pos.append(t)
                    curr_indel_kind.append(-1)
            if dna_len > 0:
                dna_len -= 1
            dna_buf[dna_len] = raw_seq[t]
            dna_len += 1
            prev_match = st

        elif codon_start != 0 and ((I1 <= st <= I6) or (I1r <= st <= I6r)):
            curr_indel_pos.append(t)
            curr_indel_kind.append(1)

        elif codon_start != 0 and st == R:
            codon_start = 0
            start_t = -1
            dna_len = 0
            curr_indel_pos.clear()
            curr_indel_kind.clear()

    return starts, ends, strands, cds_list, indel_pos_list, indel_kind_list
