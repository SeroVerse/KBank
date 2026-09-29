r"""HMM states, transition indices, mathematical constants, and lookup tables."""

import math

import numpy as np

# HMM States -----------------------------------------------------------------------------------------------------------
S = 0
E = 1
R = 2
Sr = 3
Er = 4
M1 = 5
M2 = 6
M3 = 7
M4 = 8
M5 = 9
M6 = 10
M1r = 11
M2r = 12
M3r = 13
M4r = 14
M5r = 15
M6r = 16
I1 = 17
I2 = 18
I3 = 19
I4 = 20
I5 = 21
I6 = 22
I1r = 23
I2r = 24
I3r = 25
I4r = 26
I5r = 27
I6r = 28
NUM_STATES = 29

# Global Transition Penalty Indices ------------------------------------------------------------------------------------
TR_MM = 0
TR_MI = 1
TR_MD = 2
TR_II = 3
TR_IM = 4
TR_DD = 5
TR_DM = 6
TR_GE = 7
TR_GG = 8
TR_ER = 9
TR_RS = 10
TR_RR = 11
TR_ES = 12
TR_ES1 = 13

# Mathematical Precomputed Constants -----------------------------------------------------------------------------------
LN_025 = math.log(0.25)
LN_095 = math.log(0.95)
LN_053 = math.log(0.53)
LN_016 = math.log(0.16)
LN_030 = math.log(0.30)
LN_054 = math.log(0.54)
LN_083 = math.log(0.83)
LN_010 = math.log(0.10)
LN_007 = math.log(0.07)

# Nucleotide Encoding LUT (A=0, C=1, G=2, T/U=3, unknown=4) -----------------------------------------------------------
LUT_ENC = np.full(256, 4, dtype=np.int8)
LUT_ENC[ord("A")] = 0
LUT_ENC[ord("a")] = 0
LUT_ENC[ord("C")] = 1
LUT_ENC[ord("c")] = 1
LUT_ENC[ord("G")] = 2
LUT_ENC[ord("g")] = 2
LUT_ENC[ord("T")] = 3
LUT_ENC[ord("t")] = 3
LUT_ENC[ord("U")] = 3
LUT_ENC[ord("u")] = 3

# Reverse Complement ASCII LUT -----------------------------------------------------------------------------------------
LUT_RC = np.full(256, ord("N"), dtype=np.uint8)
LUT_RC[ord("A")] = ord("T")
LUT_RC[ord("a")] = ord("t")
LUT_RC[ord("C")] = ord("G")
LUT_RC[ord("c")] = ord("g")
LUT_RC[ord("G")] = ord("C")
LUT_RC[ord("g")] = ord("c")
LUT_RC[ord("T")] = ord("A")
LUT_RC[ord("t")] = ord("a")
LUT_RC[ord("U")] = ord("A")
LUT_RC[ord("u")] = ord("a")
