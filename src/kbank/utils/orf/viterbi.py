r"""Numba-accelerated 29-state Viterbi forward dynamic programming and backtracking."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._constants import (
    I1,
    I2,
    I3,
    I4,
    I5,
    I6,
    LN_007,
    LN_010,
    LN_016,
    LN_025,
    LN_030,
    LN_053,
    LN_054,
    LN_083,
    LN_095,
    M1,
    M2,
    M3,
    M4,
    M5,
    M6,
    NUM_STATES,
    TR_DD,
    TR_DM,
    TR_ER,
    TR_ES,
    TR_ES1,
    TR_GE,
    TR_GG,
    TR_II,
    TR_IM,
    TR_MD,
    TR_MI,
    TR_MM,
    TR_RR,
    TR_RS,
    E,
    Er,
    I1r,
    I2r,
    I3r,
    I4r,
    I5r,
    I6r,
    M1r,
    M2r,
    M3r,
    M4r,
    M5r,
    M6r,
    R,
    S,
    Sr,
)


@njit(fastmath=True, nogil=True, cache=True)
def _modify_border_dist(
    cell_val: float,
    values: np.ndarray,  # type: ignore[type-arg]
    start_freq: float,
) -> float:
    h_kd = values[2] * math.exp(-1.0 * (start_freq - values[1]) ** 2 / (values[0] ** 2) / 2.0)
    r_kd = values[5] * math.exp(-1.0 * (start_freq - values[4]) ** 2 / (values[3] ** 2) / 2.0)
    denom = h_kd + r_kd
    if denom == 0.0:
        ratio = 0.01
    else:
        ratio = h_kd / denom
    if ratio < 0.01:
        ratio = 0.01
    elif ratio > 0.99:
        ratio = 0.99
    return cell_val - math.log(ratio)


@njit(fastmath=True, nogil=True, cache=True)
def _trinucleotide(seq: np.ndarray, i: int) -> int:  # type: ignore[type-arg]
    n0 = seq[i]
    n1 = seq[i + 1]
    n2 = seq[i + 2]
    if n0 > 3 or n1 > 3 or n2 > 3:
        return 0
    return int(n0) * 16 + int(n1) * 4 + int(n2)


@njit(fastmath=True, nogil=True, cache=True)
def _viterbi_forward(
    seq: np.ndarray,  # type: ignore[type-arg]
    cg_bin: int,
    whole_genome: bool,
    tr: np.ndarray,  # type: ignore[type-arg]
    tr_mi: np.ndarray,  # type: ignore[type-arg]
    tr_ii: np.ndarray,  # type: ignore[type-arg]
    pi: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_start: np.ndarray,  # type: ignore[type-arg]
    tr_stop: np.ndarray,  # type: ignore[type-arg]
    dist_start: np.ndarray,  # type: ignore[type-arg]
    dist_stop: np.ndarray,  # type: ignore[type-arg]
) -> tuple[int, np.ndarray]:  # type: ignore[type-arg]
    seq_len = len(seq)
    alpha = np.zeros((4, NUM_STATES), dtype=np.float64)
    path = np.zeros((seq_len, NUM_STATES), dtype=np.int8)

    temp_i = np.zeros(6, dtype=np.int64)
    temp_i_1 = np.zeros(6, dtype=np.int64)

    # Initial state costs
    for i in range(NUM_STATES):
        alpha[0, i] = -pi[i]

    # Stop codon at start
    if seq[0] == 3 and (
        (seq[1] == 0 and seq[2] == 0) or (seq[1] == 0 and seq[2] == 2) or (seq[1] == 2 and seq[2] == 0)
    ):
        alpha[0, E] = np.inf
        alpha[1, E] = np.inf
        path[1, E] = E
        path[2, E] = E

        alpha[2, M6] = np.inf
        alpha[1, M5] = np.inf
        alpha[0, M4] = np.inf
        alpha[2, M3] = np.inf
        alpha[1, M2] = np.inf
        alpha[0, M1] = np.inf

        if seq[1] == 0 and seq[2] == 0:
            alpha[2, E] -= LN_053
        elif seq[1] == 0 and seq[2] == 2:
            alpha[2, E] -= LN_016
        else:
            alpha[2, E] -= LN_030

    # Reverse stop codon at start
    if seq[2] == 0 and (
        (seq[1] == 3 and seq[0] == 3) or (seq[1] == 3 and seq[0] == 1) or (seq[1] == 1 and seq[0] == 3)
    ):
        alpha[0, Sr] = np.inf
        alpha[1, Sr] = np.inf
        alpha[2, Sr] = alpha[0, S]
        path[1, Sr] = Sr
        path[2, Sr] = Sr

        alpha[2, M3r] = np.inf
        alpha[2, M6r] = np.inf

        if seq[1] == 3 and seq[0] == 3:
            alpha[2, Sr] = LN_053
        elif seq[1] == 3 and seq[0] == 1:
            alpha[2, Sr] = LN_016
        else:
            alpha[2, Sr] = LN_030

    num_noncoding = 0
    for t in range(1, seq_len):
        r_curr = t & 3
        r_prev = (t - 1) & 3
        r_next1 = (t + 1) & 3
        r_next2 = (t + 2) & 3
        if t + 2 < seq_len:
            alpha[r_next2, :] = 0.0
        from_nt = seq[t - 1]
        if from_nt > 3:
            from_nt = 2
        if t > 1:
            from0 = seq[t - 2]
            if from0 > 3:
                from0 = 2
        else:
            from0 = 2

        to_nt = seq[t]
        if to_nt > 3:
            num_noncoding += 1
            to_nt = 2

        from2 = from0 * 4 + from_nt

        # Match State M1
        if np.isfinite(alpha[r_curr, M1]):
            alpha[r_curr, M1] = alpha[r_prev, M6] - tr[TR_GG] - tr[TR_MM] - e_match[0, cg_bin, 0, from2, to_nt]
            path[t, M1] = M6

            if not whole_genome:
                em = e_match[0, cg_bin, 0, from2, to_nt]
                for state_from, num_d in (
                    (M5, 2.0),
                    (M4, 3.0),
                    (M3, 4.0),
                    (M2, 5.0),
                    (M1, 6.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M1]:
                        alpha[r_curr, M1] = temp_a
                        path[t, M1] = state_from

            temp_a = alpha[r_prev, S] - e_match[0, cg_bin, 0, from2, to_nt]
            if temp_a <= alpha[r_curr, M1]:
                alpha[r_curr, M1] = temp_a
                path[t, M1] = S

            temp_idx = temp_i[5]
            avoid = False
            if t >= 2 and temp_idx > 0 and seq[temp_idx - 1] == 3:
                if (
                    (seq[temp_idx] == 0 and seq[t] == 0)
                    or (seq[temp_idx] == 0 and seq[t] == 2)
                    or (seq[temp_idx] == 2 and seq[t] == 0)
                ):
                    avoid = True
            if not avoid:
                temp_a = alpha[r_prev, I6] - tr[TR_IM] - LN_025
                if temp_a < alpha[r_curr, M1]:
                    alpha[r_curr, M1] = temp_a
                    path[t, M1] = I6

        # Match State M2
        if np.isfinite(alpha[r_curr, M2]):
            alpha[r_curr, M2] = alpha[r_prev, M1] - tr[TR_MM] - e_match[0, cg_bin, 1, from2, to_nt]
            path[t, M2] = M1

            if not whole_genome:
                em = e_match[0, cg_bin, 1, from2, to_nt]
                for state_from, num_d in (
                    (M6, 2.0),
                    (M5, 3.0),
                    (M4, 4.0),
                    (M3, 5.0),
                    (M2, 6.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M2]:
                        alpha[r_curr, M2] = temp_a
                        path[t, M2] = state_from

            temp_idx = temp_i[0]
            avoid = False
            if (
                t >= 2
                and t + 1 < seq_len
                and seq[temp_idx] == 3
                and (
                    (seq[t] == 0 and seq[t + 1] == 0)
                    or (seq[t] == 0 and seq[t + 1] == 2)
                    or (seq[t] == 2 and seq[t + 1] == 0)
                )
            ):
                avoid = True
            if not avoid:
                temp_a = alpha[r_prev, I1] - tr[TR_IM] - LN_025
                if temp_a < alpha[r_curr, M2]:
                    alpha[r_curr, M2] = temp_a
                    path[t, M2] = I1

        # Match State M3
        if np.isfinite(alpha[r_curr, M3]):
            alpha[r_curr, M3] = alpha[r_prev, M2] - tr[TR_MM] - e_match[0, cg_bin, 2, from2, to_nt]
            path[t, M3] = M2

            if not whole_genome:
                em = e_match[0, cg_bin, 2, from2, to_nt]
                for state_from, num_d in (
                    (M6, 3.0),
                    (M5, 4.0),
                    (M4, 5.0),
                    (M3, 6.0),
                    (M1, 2.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M3]:
                        alpha[r_curr, M3] = temp_a
                        path[t, M3] = state_from

            temp_idx = temp_i[1]
            avoid = False
            if (
                t >= 2
                and temp_idx > 0
                and seq[temp_idx - 1] == 3
                and (
                    (seq[temp_idx] == 0 and seq[t] == 0)
                    or (seq[temp_idx] == 0 and seq[t] == 2)
                    or (seq[temp_idx] == 2 and seq[t] == 0)
                )
            ):
                avoid = True
            if not avoid:
                temp_a = alpha[r_prev, I2] - tr[TR_IM] - LN_025
                if temp_a < alpha[r_curr, M3]:
                    alpha[r_curr, M3] = temp_a
                    path[t, M3] = I2

        # Match State M4
        if np.isfinite(alpha[r_curr, M4]):
            alpha[r_curr, M4] = alpha[r_prev, M3] - tr[TR_MM] - e_match[0, cg_bin, 3, from2, to_nt]
            path[t, M4] = M3

            if not whole_genome:
                em = e_match[0, cg_bin, 3, from2, to_nt]
                for state_from, num_d in (
                    (M6, 4.0),
                    (M5, 5.0),
                    (M4, 6.0),
                    (M2, 2.0),
                    (M1, 3.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M4]:
                        alpha[r_curr, M4] = temp_a
                        path[t, M4] = state_from

            temp_a = alpha[r_prev, I3] - tr[TR_IM] - LN_025
            if temp_a < alpha[r_curr, M4]:
                alpha[r_curr, M4] = temp_a
                path[t, M4] = I3

        # Match State M5
        if np.isfinite(alpha[r_curr, M5]):
            alpha[r_curr, M5] = alpha[r_prev, M4] - tr[TR_MM] - e_match[0, cg_bin, 4, from2, to_nt]
            path[t, M5] = M4

            if not whole_genome:
                em = e_match[0, cg_bin, 4, from2, to_nt]
                for state_from, num_d in (
                    (M6, 5.0),
                    (M5, 6.0),
                    (M3, 2.0),
                    (M2, 3.0),
                    (M1, 4.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M5]:
                        alpha[r_curr, M5] = temp_a
                        path[t, M5] = state_from

            temp_idx = temp_i[3]
            avoid = False
            if (
                t >= 2
                and t + 1 < seq_len
                and seq[temp_idx] == 3
                and (
                    (seq[t] == 0 and seq[t + 1] == 0)
                    or (seq[t] == 0 and seq[t + 1] == 2)
                    or (seq[t] == 2 and seq[t + 1] == 0)
                )
            ):
                avoid = True
            if not avoid:
                temp_a = alpha[r_prev, I4] - tr[TR_IM] - LN_025
                if temp_a < alpha[r_curr, M5]:
                    alpha[r_curr, M5] = temp_a
                    path[t, M5] = I4

        # Match State M6
        if np.isfinite(alpha[r_curr, M6]):
            alpha[r_curr, M6] = alpha[r_prev, M5] - tr[TR_MM] - e_match[0, cg_bin, 5, from2, to_nt]
            path[t, M6] = M5

            if not whole_genome:
                em = e_match[0, cg_bin, 5, from2, to_nt]
                for state_from, num_d in (
                    (M6, 6.0),
                    (M4, 2.0),
                    (M3, 3.0),
                    (M2, 4.0),
                    (M1, 5.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M6]:
                        alpha[r_curr, M6] = temp_a
                        path[t, M6] = state_from

            temp_idx = temp_i[4]
            avoid = False
            if (
                t >= 2
                and temp_idx > 0
                and seq[temp_idx - 1] == 3
                and (
                    (seq[temp_idx] == 0 and seq[t] == 0)
                    or (seq[temp_idx] == 0 and seq[t] == 2)
                    or (seq[temp_idx] == 2 and seq[t] == 0)
                )
            ):
                avoid = True
            if not avoid:
                temp_a = alpha[r_prev, I5] - tr[TR_IM] - LN_025
                if temp_a < alpha[r_curr, M6]:
                    alpha[r_curr, M6] = temp_a
                    path[t, M6] = I5

        # Forward Insertion States I1..I6
        for st_idx in range(6):
            istate = I1 + st_idx
            alpha[r_curr, istate] = alpha[r_prev, istate] - tr[TR_II] - tr_ii[from_nt, to_nt]
            path[t, istate] = istate

        for st_idx in range(6):
            mstate = M1 + st_idx
            istate = I1 + st_idx
            last_i = tr[TR_GG] if st_idx == 5 else 0.0
            temp_a = alpha[r_prev, mstate] - tr[TR_MI] - tr_mi[from_nt, to_nt] - last_i
            if temp_a < alpha[r_curr, istate]:
                alpha[r_curr, istate] = temp_a
                path[t, istate] = mstate
                temp_i[st_idx] = t - 1

        # Reverse Match State M1r
        if (
            t >= 3
            and seq[t - 1] == 0
            and (
                (seq[t - 2] == 3 and seq[t - 3] == 3)
                or (seq[t - 2] == 3 and seq[t - 3] == 1)
                or (seq[t - 2] == 1 and seq[t - 3] == 3)
            )
        ):
            alpha[r_curr, M1r] = alpha[r_prev, Sr] - e_match[1, cg_bin, 0, from2, to_nt]
            path[t, M1r] = Sr
        else:
            alpha[r_curr, M1r] = alpha[r_prev, M6r] - tr[TR_GG] - tr[TR_MM] - e_match[1, cg_bin, 0, from2, to_nt]
            path[t, M1r] = M6r

            if not whole_genome:
                em = e_match[1, cg_bin, 0, from2, to_nt]
                for state_from, num_d in (
                    (M5r, 2.0),
                    (M4r, 3.0),
                    (M3r, 4.0),
                    (M2r, 5.0),
                    (M1r, 6.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M1r]:
                        alpha[r_curr, M1r] = temp_a
                        path[t, M1r] = state_from

            temp_idx = temp_i_1[5]
            avoid = False
            if (
                t >= 2
                and seq[t] == 0
                and temp_idx > 1
                and (
                    (seq[temp_idx] == 3 and seq[temp_idx - 1] == 3)
                    or (seq[temp_idx] == 3 and seq[temp_idx - 1] == 1)
                    or (seq[temp_idx] == 1 and seq[temp_idx - 1] == 3)
                )
            ):
                avoid = True
            if not avoid:
                temp_a = alpha[r_prev, I6r] - tr[TR_IM] - LN_025
                if temp_a < alpha[r_curr, M1r]:
                    alpha[r_curr, M1r] = temp_a
                    path[t, M1r] = I6r

        # Reverse Match State M2r
        alpha[r_curr, M2r] = alpha[r_prev, M1r] - tr[TR_MM] - e_match[1, cg_bin, 1, from2, to_nt]
        path[t, M2r] = M1r

        if not whole_genome:
            em = e_match[1, cg_bin, 1, from2, to_nt]
            for state_from, num_d in (
                (M6r, 2.0),
                (M5r, 3.0),
                (M4r, 4.0),
                (M3r, 5.0),
                (M2r, 6.0),
            ):
                temp_a = (
                    alpha[r_prev, state_from]
                    - tr[TR_MD]
                    - em
                    - LN_025 * (num_d - 1.0)
                    - tr[TR_DD] * (num_d - 2.0)
                    - tr[TR_DM]
                )
                if temp_a < alpha[r_curr, M2r]:
                    alpha[r_curr, M2r] = temp_a
                    path[t, M2r] = state_from

        temp_idx = temp_i_1[0]
        avoid = False
        if (
            t >= 2
            and t + 1 < seq_len
            and seq[t + 1] == 0
            and (
                (seq[t] == 3 and seq[temp_idx] == 3)
                or (seq[t] == 3 and seq[temp_idx] == 1)
                or (seq[t] == 0 and seq[temp_idx] == 3)
            )
        ):
            avoid = True
        if not avoid:
            temp_a = alpha[r_prev, I1r] - tr[TR_IM] - LN_025
            if temp_a < alpha[r_curr, M2r]:
                alpha[r_curr, M2r] = temp_a
                path[t, M2r] = I1r

        # Reverse Match State M3r
        alpha[r_curr, M3r] = alpha[r_prev, M2r] - tr[TR_MM] - e_match[1, cg_bin, 2, from2, to_nt]
        path[t, M3r] = M2r

        if not whole_genome:
            em = e_match[1, cg_bin, 2, from2, to_nt]
            for state_from, num_d in (
                (M6r, 3.0),
                (M5r, 4.0),
                (M4r, 5.0),
                (M3r, 6.0),
                (M1r, 2.0),
            ):
                temp_a = (
                    alpha[r_prev, state_from]
                    - tr[TR_MD]
                    - em
                    - LN_025 * (num_d - 1.0)
                    - tr[TR_DD] * (num_d - 2.0)
                    - tr[TR_DM]
                )
                if temp_a < alpha[r_curr, M3r]:
                    alpha[r_curr, M3r] = temp_a
                    path[t, M3r] = state_from

        temp_idx = temp_i_1[1]
        avoid = False
        if (
            t >= 2
            and seq[t] == 0
            and temp_idx > 1
            and (
                (seq[temp_idx] == 3 and seq[temp_idx - 1] == 3)
                or (seq[temp_idx] == 3 and seq[temp_idx - 1] == 1)
                or (seq[temp_idx] == 1 and seq[temp_idx - 1] == 3)
            )
        ):
            avoid = True
        if not avoid:
            temp_a = alpha[r_prev, I2r] - tr[TR_IM] - LN_025
            if temp_a < alpha[r_curr, M3r]:
                alpha[r_curr, M3r] = temp_a
                path[t, M3r] = I2r

        # Reverse Match State M4r
        if (
            t >= 3
            and seq[t - 1] == 0
            and (
                (seq[t - 2] == 3 and seq[t - 3] == 3)
                or (seq[t - 2] == 3 and seq[t - 3] == 1)
                or (seq[t - 2] == 1 and seq[t - 3] == 3)
            )
        ):
            alpha[r_curr, M4r] = alpha[r_prev, Sr] - e_match[1, cg_bin, 3, from2, to_nt]
            path[t, M4r] = Sr
        else:
            alpha[r_curr, M4r] = alpha[r_prev, M3r] - tr[TR_MM] - e_match[1, cg_bin, 3, from2, to_nt]
            path[t, M4r] = M3r

            if not whole_genome:
                em = e_match[1, cg_bin, 3, from2, to_nt]
                for state_from, num_d in (
                    (M6r, 4.0),
                    (M5r, 5.0),
                    (M4r, 6.0),
                    (M2r, 2.0),
                    (M1r, 3.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < alpha[r_curr, M4r]:
                        alpha[r_curr, M4r] = temp_a
                        path[t, M4r] = state_from

            temp_a = alpha[r_prev, I3r] - tr[TR_IM] - LN_025
            if temp_a < alpha[r_curr, M4r]:
                alpha[r_curr, M4r] = temp_a
                path[t, M4r] = I3r

        # Reverse Match State M5r
        alpha[r_curr, M5r] = alpha[r_prev, M4r] - tr[TR_MM] - e_match[1, cg_bin, 4, from2, to_nt]
        path[t, M5r] = M4r

        if not whole_genome:
            em = e_match[1, cg_bin, 4, from2, to_nt]
            for state_from, num_d in (
                (M6r, 5.0),
                (M5r, 6.0),
                (M3r, 2.0),
                (M2r, 3.0),
                (M1r, 4.0),
            ):
                temp_a = (
                    alpha[r_prev, state_from]
                    - tr[TR_MD]
                    - em
                    - LN_025 * (num_d - 1.0)
                    - tr[TR_DD] * (num_d - 2.0)
                    - tr[TR_DM]
                )
                if temp_a < alpha[r_curr, M5r]:
                    alpha[r_curr, M5r] = temp_a
                    path[t, M5r] = state_from

        temp_idx = temp_i_1[3]
        avoid = False
        if (
            t >= 2
            and t + 1 < seq_len
            and seq[t + 1] == 0
            and (
                (seq[t] == 3 and seq[temp_idx] == 3)
                or (seq[t] == 3 and seq[temp_idx] == 1)
                or (seq[t] == 0 and seq[temp_idx] == 3)
            )
        ):
            avoid = True
        if not avoid:
            temp_a = alpha[r_prev, I4r] - tr[TR_IM] - LN_025
            if temp_a < alpha[r_curr, M5r]:
                alpha[r_curr, M5r] = temp_a
                path[t, M5r] = I4r

        # Reverse Match State M6r
        alpha[r_curr, M6r] = alpha[r_prev, M5r] - tr[TR_MM] - e_match[1, cg_bin, 5, from2, to_nt]
        path[t, M6r] = M5r

        if not whole_genome:
            em = e_match[1, cg_bin, 5, from2, to_nt]
            for state_from, num_d in (
                (M6r, 6.0),
                (M4r, 2.0),
                (M3r, 3.0),
                (M2r, 4.0),
                (M1r, 5.0),
            ):
                temp_a = (
                    alpha[r_prev, state_from]
                    - tr[TR_MD]
                    - em
                    - LN_025 * (num_d - 1.0)
                    - tr[TR_DD] * (num_d - 2.0)
                    - tr[TR_DM]
                )
                if temp_a < alpha[r_curr, M6r]:
                    alpha[r_curr, M6r] = temp_a
                    path[t, M6r] = state_from

        temp_idx = temp_i_1[4]
        avoid = False
        if (
            t >= 2
            and seq[t] == 0
            and temp_idx > 1
            and (
                (seq[temp_idx] == 3 and seq[temp_idx - 1] == 3)
                or (seq[temp_idx] == 3 and seq[temp_idx - 1] == 1)
                or (seq[temp_idx] == 1 and seq[temp_idx - 1] == 3)
            )
        ):
            avoid = True
        if not avoid:
            temp_a = alpha[r_prev, I5r] - tr[TR_IM] - LN_025
            if temp_a < alpha[r_curr, M6r]:
                alpha[r_curr, M6r] = temp_a
                path[t, M6r] = I5r

        # Reverse Insertion States I1r..I6r
        for st_idx in range(6):
            istate = I1r + st_idx
            alpha[r_curr, istate] = alpha[r_prev, istate] - tr[TR_II] - tr_ii[from_nt, to_nt]
            path[t, istate] = istate

        if (t < 3 or path[t - 3, Sr] != R) and (t < 4 or path[t - 4, Sr] != R) and (t < 5 or path[t - 5, Sr] != R):
            for st_idx in range(6):
                mstate = M1r + st_idx
                istate = I1r + st_idx
                last_i = tr[TR_GG] if st_idx == 5 else 0.0
                temp_a = alpha[r_prev, mstate] - tr[TR_MI] - tr_mi[from_nt, to_nt] - last_i
                if temp_a < alpha[r_curr, istate]:
                    alpha[r_curr, istate] = temp_a
                    path[t, istate] = mstate
                    temp_i_1[st_idx] = t - 1

        # Non-coding State R
        alpha[r_curr, R] = alpha[r_prev, R] - tr_rr[cg_bin, from_nt, to_nt] - tr[TR_RR] - LN_095
        path[t, R] = R

        temp_a = alpha[r_prev, E] - tr[TR_ER] - LN_095
        if temp_a < alpha[r_curr, R]:
            alpha[r_curr, R] = temp_a
            path[t, R] = E

        temp_a = alpha[r_prev, Er] - tr[TR_ER] - LN_095
        if temp_a < alpha[r_curr, R]:
            alpha[r_curr, R] = temp_a
            path[t, R] = Er

        # End State E (Forward Stop)
        if alpha[r_curr, E] == 0.0:
            alpha[r_curr, E] = np.inf
            path[t, E] = -1

            if (
                t < seq_len - 2
                and seq[t] == 3
                and (
                    (seq[t + 1] == 0 and seq[t + 2] == 0)
                    or (seq[t + 1] == 0 and seq[t + 2] == 2)
                    or (seq[t + 1] == 2 and seq[t + 2] == 0)
                )
            ):
                alpha[r_next2, E] = np.inf

                temp_a = alpha[r_prev, M6] - tr[TR_GE]
                if temp_a < alpha[r_next2, E]:
                    alpha[r_next2, E] = temp_a
                    path[t, E] = M6

                temp_a = alpha[r_prev, M3] - tr[TR_GE]
                if temp_a < alpha[r_next2, E]:
                    alpha[r_next2, E] = temp_a
                    path[t, E] = M3

                alpha[r_curr, E] = np.inf
                alpha[r_next1, E] = np.inf
                path[t + 1, E] = E
                path[t + 2, E] = E

                alpha[r_next2, M6] = np.inf
                alpha[r_next1, M5] = np.inf
                alpha[r_curr, M4] = np.inf
                alpha[r_next2, M3] = np.inf
                alpha[r_next1, M2] = np.inf
                alpha[r_curr, M1] = np.inf

                if seq[t + 1] == 0 and seq[t + 2] == 0:
                    alpha[r_next2, E] -= LN_054
                elif seq[t + 1] == 0 and seq[t + 2] == 2:
                    alpha[r_next2, E] -= LN_016
                else:
                    alpha[r_next2, E] -= LN_030

                start_freq = 0.0
                start_i = max(0, t - 60)
                end_i = max(0, t - 3)
                for i in range(start_i, end_i + 1):
                    tr_idx = _trinucleotide(seq, i)
                    start_freq -= tr_stop[0, cg_bin, i + 60 - t, tr_idx]
                if t < 60:
                    denom = max(0, t - 3) + 1
                    start_freq *= 58.0 / denom

                alpha[r_next2, E] = _modify_border_dist(alpha[r_next2, E], dist_stop[0, cg_bin], start_freq)

        # Start State Sr (Reverse Stop)
        if alpha[r_curr, Sr] == 0.0:
            alpha[r_curr, Sr] = np.inf
            path[t, Sr] = -1

            if (
                t < seq_len - 2
                and seq[t + 2] == 0
                and (
                    (seq[t + 1] == 3 and seq[t] == 3)
                    or (seq[t + 1] == 3 and seq[t] == 1)
                    or (seq[t + 1] == 1 and seq[t] == 3)
                )
            ):
                alpha[r_curr, Sr] = np.inf
                alpha[r_next1, Sr] = np.inf
                alpha[r_next2, Sr] = alpha[r_prev, R] - tr[TR_RS]
                path[t, Sr] = R
                path[t + 1, Sr] = Sr
                path[t + 2, Sr] = Sr

                temp_a = alpha[r_prev, Er] - tr[TR_ES]
                if temp_a < alpha[r_next2, Sr]:
                    alpha[r_next2, Sr] = temp_a
                    path[t, Sr] = Er

                temp_a = alpha[r_prev, E] - tr[TR_ES1]
                if temp_a < alpha[r_next2, Sr]:
                    alpha[r_next2, Sr] = temp_a
                    path[t, Sr] = E

                alpha[r_next2, M3r] = np.inf
                alpha[r_next2, M6r] = np.inf

                if seq[t + 1] == 3 and seq[t] == 3:
                    alpha[r_next2, Sr] -= LN_054
                elif seq[t + 1] == 3 and seq[t] == 1:
                    alpha[r_next2, Sr] -= LN_016
                else:
                    alpha[r_next2, Sr] -= LN_030

                start_freq = 0.0
                if t + 5 < seq_len:
                    start_i = t + 3
                    end_i = min(t + 60, seq_len - 3)
                    for i in range(start_i, end_i + 1):
                        tr_idx = _trinucleotide(seq, i)
                        start_freq -= tr_start[1, cg_bin, i - 3 - t, tr_idx]

                alpha[r_next2, Sr] = _modify_border_dist(alpha[r_next2, Sr], dist_start[1, cg_bin], start_freq)

        # Start State S (Forward Start)
        if alpha[r_curr, S] == 0.0:
            alpha[r_curr, S] = np.inf
            path[t, S] = -1

            if t < seq_len - 2 and (seq[t] == 0 or seq[t] == 2 or seq[t] == 3) and seq[t + 1] == 3 and seq[t + 2] == 2:
                alpha[r_curr, S] = np.inf
                alpha[r_next1, S] = np.inf
                alpha[r_next2, S] = alpha[r_prev, R] - tr[TR_RS]
                path[t, S] = R
                path[t + 1, S] = S
                path[t + 2, S] = S

                temp_a = alpha[r_prev, E] - tr[TR_ES]
                if temp_a < alpha[r_next2, S]:
                    alpha[r_next2, S] = temp_a
                    path[t, S] = E

                temp_a = alpha[r_prev, Er] - tr[TR_ES1]
                if temp_a < alpha[r_next2, S]:
                    alpha[r_next2, S] = temp_a
                    path[t, S] = Er

                if seq[t] == 0:
                    alpha[r_next2, S] -= LN_083
                elif seq[t] == 2:
                    alpha[r_next2, S] -= LN_010
                else:
                    alpha[r_next2, S] -= LN_007

                start_freq = 0.0
                start_i = max(0, t - 30)
                end_i = min(t + 30, seq_len - 3)
                for i in range(start_i, end_i + 1):
                    tr_idx = _trinucleotide(seq, i)
                    start_freq -= tr_start[0, cg_bin, i + 30 - t, tr_idx]
                if t < 30:
                    start_freq *= 61.0 / (t + 30 + 1)

                alpha[r_next2, S] = _modify_border_dist(alpha[r_next2, S], dist_start[0, cg_bin], start_freq)

        # End State Er (Reverse Start)
        if alpha[r_curr, Er] == 0.0:
            alpha[r_curr, Er] = np.inf
            path[t, Er] = -1

            if (
                t < seq_len - 2
                and seq[t] == 1
                and seq[t + 1] == 0
                and (seq[t + 2] == 3 or seq[t + 2] == 1 or seq[t + 2] == 0)
            ):
                alpha[r_curr, Er] = np.inf
                alpha[r_next1, Er] = np.inf
                alpha[r_next2, Er] = alpha[r_prev, M6r] - tr[TR_GE]
                path[t, Er] = M6r
                path[t + 1, Er] = Er
                path[t + 2, Er] = Er

                if seq[t + 2] == 3:
                    alpha[r_next2, Er] -= LN_083
                elif seq[t + 2] == 1:
                    alpha[r_next2, Er] -= LN_010
                else:
                    alpha[r_next2, Er] -= LN_007

                start_freq = 0.0
                start_i = max(0, t - 30)
                end_i = min(t + 30, seq_len - 3)
                for i in range(start_i, end_i + 1):
                    tr_idx = _trinucleotide(seq, i)
                    start_freq -= tr_stop[1, cg_bin, i + 30 - t, tr_idx]
                if t < 30:
                    start_freq *= 61.0 / (t + 30 + 1)

                alpha[r_next2, Er] = _modify_border_dist(alpha[r_next2, Er], dist_stop[1, cg_bin], start_freq)

        if num_noncoding > 9:
            for i in range(NUM_STATES):
                if i != R:
                    alpha[r_curr, i] = np.inf
                    path[t, i] = R

    last_r = (seq_len - 1) & 3
    best_st = 0
    best_prob = np.inf
    for i in range(NUM_STATES):
        if alpha[last_r, i] < best_prob:
            best_prob = alpha[last_r, i]
            best_st = i
    return best_st, path


@njit(fastmath=True, nogil=True, cache=True)
def _viterbi_backtrack(
    best_st: int,
    path: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    seq_len = len(path)
    vpath = np.zeros(seq_len, dtype=np.int8)

    vpath[seq_len - 1] = best_st
    for t in range(seq_len - 2, -1, -1):
        prev = path[t + 1, vpath[t + 1]]
        if prev < 0:
            prev = R
        vpath[t] = prev

    return vpath
