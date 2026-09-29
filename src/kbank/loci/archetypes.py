"""Canonical machinery archetypes and probabilistic completeness scoring for BGC loci.

This module formalizes the six universal bacterial surface glycan archetypes:
1. CPS_WZY (Group 1/4 Capsule: Wza/Wzb/Wzc pore/kinase + Wzx/Wzy)
2. CPS_ABC (Group 2/3 Capsule: KpsM/KpsT ABC transporter + KpsD/E/C translocon/anchor)
3. LPS_WZY (Wzy-Dependent O-Antigen: Wzx/Wzy/Wzz, ligated to Lipid A)
4. LPS_ABC (ABC-Dependent O-Antigen: Wzm/Wzt, no capsule pore, ligated to Lipid A)
5. ECA_WZY (Enterobacterial Common Antigen: WecA/WzxE/WzyE/WzzE housekeeping cluster)
6. BIOFILM_SYNTHASE (Processive Synthase EPS: Alg8/Alg44/PelF, Pga, Cellulose)

Each archetype enforces required functional modules (AND across modules, OR within set),
strictly incompatible roles (discordant machinery), and probabilistic geometric-mean completeness.
All archetypes, masks, and rules are encapsulated in the Archetypes Catalog singleton,
backed by 32-bit integer bitmasks and vectorized boolean operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np

from kbank.core.models import BaseIndexEnum, Catalog
from kbank.genes.roles import GeneRoles


class LocusArchetype(BaseIndexEnum):
    """Universal bacterial surface polysaccharide machinery archetypes."""

    CPS_WZY = 0  # Group 1/4 Capsule (Wzy-dependent, outer pore wza/wzb/wzc)
    CPS_ABC = 1  # Group 2/3 Capsule (ABC transporter kpsM/T + outer translocon kpsD/E/C)
    LPS_WZY = 2  # O-Antigen (Wzy-dependent, copolymerase wzz, ligated to Lipid A)
    LPS_ABC = 3  # O-Antigen (ABC transporter wzm/wzt, no capsule translocon, ligated to Lipid A)
    ECA_WZY = 4  # Enterobacterial Common Antigen (conserved housekeeping wec cluster)
    BIOFILM_SYNTHASE = 5  # Biofilm Exopolysaccharide (alg8/alg44/pelF, pga, cellulose)


# (name, category, required_module_masks, forbidden_mask, cargo_mask, description, aliases)
_CANONICAL_ARCHETYPES = [
    # 1. Group 1/4 Capsule (e.g. Klebsiella K-loci, E. coli K30, A. baumannii KL)
    (
        b"CPS_WZY",
        b"Capsule",
        (
            (1 << GeneRoles[b"TRANSLOCON"]) | (1 << GeneRoles[b"CAPSULE_ANCHOR"]),
            (1 << GeneRoles[b"KINASE"]) | (1 << GeneRoles[b"PHOSPHATASE"]),
            (1 << GeneRoles[b"FLIPPASE"]),
            (1 << GeneRoles[b"POLYMERASE"]) | (1 << GeneRoles[b"LIGASE"]),
            (1 << GeneRoles[b"INITIATOR"]) | (1 << GeneRoles[b"PHOSPHOTRANSFERASE"]),
            (1 << GeneRoles[b"GLYCOSYLTRANSFERASE"]),
        ),
        (
            (1 << GeneRoles[b"ABC_PERMEASE"])
            | (1 << GeneRoles[b"ABC_ATPASE"])
            | (1 << GeneRoles[b"TRANSMEMBRANE_SYNTHASE"])
        ),
        (
            (1 << GeneRoles[b"EPIMERASE"])
            | (1 << GeneRoles[b"OXIDOREDUCTASE"])
            | (1 << GeneRoles[b"ACETYLTRANSFERASE"])
            | (1 << GeneRoles[b"PYRUVYLTRANSFERASE"])
            | (1 << GeneRoles[b"NUCLEOTIDYLTRANSFERASE"])
            | (1 << GeneRoles[b"NUCLEOTIDASE"])
            | (1 << GeneRoles[b"AMINOTRANSFERASE"])
            | (1 << GeneRoles[b"ALDOLASE"])
        ),
        b"Group 1/4 Capsule: Wza/Wzb/Wzc pore/kinase + Wzx/Wzy repeat unit assembly.",
        (b"Capsule_Wzy", b"K_locus", b"Group1_Capsule", b"Group4_Capsule"),
    ),
    # 2. Group 2/3 Capsule (e.g. E. coli K1, K5; N. meningitidis; H. influenzae)
    (
        b"CPS_ABC",
        b"Capsule",
        (
            (1 << GeneRoles[b"ABC_PERMEASE"]),
            (1 << GeneRoles[b"ABC_ATPASE"]),
            (1 << GeneRoles[b"TRANSLOCON"]),
            (1 << GeneRoles[b"CAPSULE_ANCHOR"]),
            (1 << GeneRoles[b"GLYCOSYLTRANSFERASE"]) | (1 << GeneRoles[b"TRANSMEMBRANE_SYNTHASE"]),
        ),
        (
            (1 << GeneRoles[b"FLIPPASE"])
            | (1 << GeneRoles[b"POLYMERASE"])
        ),
        (
            (1 << GeneRoles[b"EPIMERASE"])
            | (1 << GeneRoles[b"OXIDOREDUCTASE"])
            | (1 << GeneRoles[b"ACETYLTRANSFERASE"])
            | (1 << GeneRoles[b"ALDOLASE"])
            | (1 << GeneRoles[b"AMINOTRANSFERASE"])
            | (1 << GeneRoles[b"NUCLEOTIDYLTRANSFERASE"])
        ),
        b"Group 2/3 Capsule: KpsM/KpsT ABC transporter + KpsD/E/C translocon/anchor.",
        (b"Capsule_ABC", b"Group2_Capsule", b"Group3_Capsule"),
    ),
    # 3. Wzy-Dependent O-Antigen (e.g. E. coli O-antigens, Salmonella, Pseudomonas O5)
    (
        b"LPS_WZY",
        b"OAntigen",
        (
            (1 << GeneRoles[b"FLIPPASE"]),
            (1 << GeneRoles[b"POLYMERASE"]) | (1 << GeneRoles[b"LIGASE"]),
            (1 << GeneRoles[b"COPOLYMERASE"]),
            (1 << GeneRoles[b"INITIATOR"]) | (1 << GeneRoles[b"PHOSPHOTRANSFERASE"]),
            (
                (1 << GeneRoles[b"GLYCOSYLTRANSFERASE"])
                | (1 << GeneRoles[b"METHYLTRANSFERASE"])
                | (1 << GeneRoles[b"ACETYLTRANSFERASE"])
            ),
        ),
        (
            (1 << GeneRoles[b"TRANSLOCON"])
            | (1 << GeneRoles[b"KINASE"])
            | (1 << GeneRoles[b"PHOSPHATASE"])
            | (1 << GeneRoles[b"ABC_PERMEASE"])
            | (1 << GeneRoles[b"ABC_ATPASE"])
            | (1 << GeneRoles[b"TRANSMEMBRANE_SYNTHASE"])
        ),
        (
            (1 << GeneRoles[b"EPIMERASE"])
            | (1 << GeneRoles[b"OXIDOREDUCTASE"])
            | (1 << GeneRoles[b"ACETYLTRANSFERASE"])
            | (1 << GeneRoles[b"AMINOTRANSFERASE"])
            | (1 << GeneRoles[b"NUCLEOTIDYLTRANSFERASE"])
            | (1 << GeneRoles[b"NUCLEOTIDASE"])
            | (1 << GeneRoles[b"FORMYLTRANSFERASE"])
        ),
        b"Wzy-Dependent O-Antigen: Wzx/Wzy/Wzz repeat unit assembly, ligated to Lipid A.",
        (b"OAntigen_Wzy", b"O_locus", b"O_antigen_Wzy"),
    ),
    # 4. ABC-Transporter O-Antigen (e.g. Klebsiella OL1-OL4, Pseudomonas A-band)
    (
        b"LPS_ABC",
        b"OAntigen",
        (
            (1 << GeneRoles[b"ABC_PERMEASE"]),
            (1 << GeneRoles[b"ABC_ATPASE"]),
            (1 << GeneRoles[b"GLYCOSYLTRANSFERASE"]) | (1 << GeneRoles[b"TRANSMEMBRANE_SYNTHASE"]),
        ),
        (
            (1 << GeneRoles[b"FLIPPASE"])
            | (1 << GeneRoles[b"POLYMERASE"])
            | (1 << GeneRoles[b"TRANSLOCON"])
            | (1 << GeneRoles[b"KINASE"])
            | (1 << GeneRoles[b"PHOSPHATASE"])
            | (1 << GeneRoles[b"CAPSULE_ANCHOR"])
        ),
        (
            (1 << GeneRoles[b"EPIMERASE"])
            | (1 << GeneRoles[b"OXIDOREDUCTASE"])
            | (1 << GeneRoles[b"ACETYLTRANSFERASE"])
            | (1 << GeneRoles[b"METHYLTRANSFERASE"])
            | (1 << GeneRoles[b"NUCLEOTIDYLTRANSFERASE"])
        ),
        b"ABC-Dependent O-Antigen: Wzm/Wzt transporter, no outer pore, ligated to Lipid A.",
        (b"OAntigen_ABC", b"O_antigen_ABC"),
    ),
    # 5. Enterobacterial Common Antigen (conserved housekeeping wec cluster)
    (
        b"ECA_WZY",
        b"Housekeeping",
        (
            (1 << GeneRoles[b"INITIATOR"]),
            (1 << GeneRoles[b"FLIPPASE"]),
            (1 << GeneRoles[b"POLYMERASE"]),
            (1 << GeneRoles[b"COPOLYMERASE"]),
            (1 << GeneRoles[b"GLYCOSYLTRANSFERASE"]),
            (
                (1 << GeneRoles[b"AMINOTRANSFERASE"])
                | (1 << GeneRoles[b"ACETYLTRANSFERASE"])
                | (1 << GeneRoles[b"EPIMERASE"])
                | (1 << GeneRoles[b"OXIDOREDUCTASE"])
            ),
        ),
        (
            (1 << GeneRoles[b"TRANSLOCON"])
            | (1 << GeneRoles[b"KINASE"])
            | (1 << GeneRoles[b"PHOSPHATASE"])
            | (1 << GeneRoles[b"ABC_PERMEASE"])
            | (1 << GeneRoles[b"ABC_ATPASE"])
            | (1 << GeneRoles[b"TRANSMEMBRANE_SYNTHASE"])
        ),
        (
            (1 << GeneRoles[b"AMINOTRANSFERASE"])
            | (1 << GeneRoles[b"ACETYLTRANSFERASE"])
            | (1 << GeneRoles[b"EPIMERASE"])
            | (1 << GeneRoles[b"OXIDOREDUCTASE"])
        ),
        b"Enterobacterial Common Antigen: conserved housekeeping wec cluster (WecA/WzxE/WzyE/WzzE).",
        (b"ECA_Wzy", b"ECA", b"EnterobacterialCommonAntigen"),
    ),
    # 6. Extracellular Biofilm / Secreted Synthase (e.g. Alginate, Pel, Pga, Cellulose)
    (
        b"BIOFILM_SYNTHASE",
        b"Biofilm",
        (
            (1 << GeneRoles[b"TRANSMEMBRANE_SYNTHASE"]),
            (1 << GeneRoles[b"COPOLYMERASE"]),
            (1 << GeneRoles[b"CAPSULE_ANCHOR"]) | (1 << GeneRoles[b"TRANSLOCON"]),
        ),
        (
            (1 << GeneRoles[b"FLIPPASE"])
            | (1 << GeneRoles[b"POLYMERASE"])
            | (1 << GeneRoles[b"ABC_PERMEASE"])
            | (1 << GeneRoles[b"ABC_ATPASE"])
        ),
        (
            (1 << GeneRoles[b"DEACETYLASE"])
            | (1 << GeneRoles[b"GLYCOSYLHYDROLASE"])
            | (1 << GeneRoles[b"EPIMERASE"])
            | (1 << GeneRoles[b"OXIDOREDUCTASE"])
            | (1 << GeneRoles[b"GLYCOSYLTRANSFERASE"])
        ),
        b"Biofilm Exopolysaccharide: processive transmembrane synthase (Alg8/PelF/PgaC) + c-di-GMP regulator.",
        (b"Biofilm_Synthase", b"Biofilm", b"Synthase_EPS"),
    ),
]


# ----------------------------------------------------------------------------------------------------------------------
# Bitmask Helper Functions
# ----------------------------------------------------------------------------------------------------------------------

def roles_to_mask(roles: Sequence[int] | np.ndarray) -> np.uint32:
    """Converts a collection or array of role integer indices into a single uint32 bitmask."""
    if len(roles) == 0:
        return np.uint32(0)
    arr = np.asarray(roles, dtype=np.int64)
    valid = arr[(arr >= 0) & (arr < 32)]
    if len(valid) == 0:
        return np.uint32(0)
    return np.bitwise_or.reduce((np.uint32(1) << valid).astype(np.uint32))


def mask_to_roles(mask: int | np.uint32) -> list[int]:
    """Unpacks a uint32 bitmask into a list of role integer indices."""
    m = int(mask)
    roles: list[int] = []
    idx = 0
    while m > 0:
        if m & 1:
            roles.append(idx)
        m >>= 1
        idx += 1
    return roles


class Archetypes(Catalog):
    """Structure-of-Arrays (SoA) catalog and boolean rule engine for locus archetypes."""

    _SOA_ARRAYS: tuple[str, ...] = (
        "_names",
        "_categories",
        "_forbidden_masks",
        "_cargo_masks",
        "_required_masks",
        "_all_relevant_masks",
        "_num_modules",
    )
    _CANONICAL_MANIFEST = _CANONICAL_ARCHETYPES
    Index: type[LocusArchetype] = LocusArchetype

    _names: np.ndarray
    _categories: np.ndarray
    _forbidden_masks: np.ndarray
    _cargo_masks: np.ndarray
    _required_masks: np.ndarray
    _all_relevant_masks: np.ndarray
    _module_masks: np.ndarray
    _num_modules: np.ndarray
    _descriptions: tuple[bytes, ...]
    _aliases: tuple[tuple[bytes, ...], ...]
    _raw_modules: tuple[tuple[int, ...], ...]

    # Precomputed tensors for continuous probabilistic scoring
    _module_role_indicators: np.ndarray
    _forbidden_role_indicators: np.ndarray
    _relevant_role_indicators: np.ndarray
    _null_priors: np.ndarray
    _active_module_mask: np.ndarray

    def init_default(self) -> None:
        """Initializes the catalog from the canonical default manifest."""
        self._init_from_manifest(self._CANONICAL_MANIFEST)

    def _init_from_manifest(
        self,
        manifest: Sequence[
            tuple[
                bytes,
                bytes,
                tuple[int, ...],
                int,
                int,
                bytes,
                tuple[bytes, ...],
            ]
        ],
        alias_map: dict[bytes, bytes] | None = None,
    ) -> None:
        N = len(manifest)
        self._names = np.empty(N, dtype="S24")
        self._categories = np.empty(N, dtype="S16")
        self._forbidden_masks = np.empty(N, dtype=np.uint32)
        self._cargo_masks = np.empty(N, dtype=np.uint32)
        self._required_masks = np.empty(N, dtype=np.uint32)
        self._all_relevant_masks = np.empty(N, dtype=np.uint32)
        self._module_masks = np.zeros((N, 8), dtype=np.uint32)
        self._num_modules = np.zeros(N, dtype=np.uint8)

        self.name_to_index = {}
        self.alias_map = dict(alias_map) if alias_map is not None else {}

        descriptions_list: list[bytes] = []
        aliases_list: list[tuple[bytes, ...]] = []
        raw_modules_list: list[tuple[int, ...]] = []

        for i, item in enumerate(manifest):
            (
                name,
                category,
                req_modules,
                forbid_mask,
                cargo_mask,
                desc,
                aliases,
            ) = item

            self._names[i] = name
            self._categories[i] = category
            self._forbidden_masks[i] = np.uint32(forbid_mask)
            self._cargo_masks[i] = np.uint32(cargo_mask)

            num_mods = len(req_modules)
            self._num_modules[i] = num_mods
            req_mask = np.uint32(0)
            for k, mod_mask in enumerate(req_modules):
                u_mod = np.uint32(mod_mask)
                self._module_masks[i, k] = u_mod
                req_mask |= u_mod

            self._required_masks[i] = req_mask
            self._all_relevant_masks[i] = req_mask | np.uint32(cargo_mask)

            self.name_to_index[name] = i

            for al in aliases:
                bal = bytes(al)
                self.alias_map[bal] = name

            descriptions_list.append(desc)
            aliases_list.append(tuple(aliases))
            raw_modules_list.append(tuple(req_modules))

        self._descriptions = tuple(descriptions_list)
        self._aliases = tuple(aliases_list)
        self._raw_modules = tuple(raw_modules_list)
        self._feature_matrix = None

        self._build_indicator_tensors()
        self.Index = LocusArchetype
        LocusArchetype._catalog = self

    def _build_indicator_tensors(self) -> None:
        """Precomputes boolean indicator tensors for continuous SIMD/BLAS scoring."""
        N = len(self._names)
        C = 30  # GeneRoles has 30 canonical roles (0..29)

        # 1. Module role indicators: shape (N, 8, 30)
        self._module_role_indicators = np.zeros((N, 8, C), dtype=np.bool_)
        for a in range(N):
            for k in range(int(self._num_modules[a])):
                mod_mask = int(self._module_masks[a, k])
                for r in range(C):
                    if (mod_mask >> r) & 1:
                        self._module_role_indicators[a, k, r] = True

        # 2. Forbidden role indicators: shape (N, 30)
        self._forbidden_role_indicators = np.zeros((N, C), dtype=np.bool_)
        for a in range(N):
            forbid_mask = int(self._forbidden_masks[a])
            for r in range(C):
                if (forbid_mask >> r) & 1:
                    self._forbidden_role_indicators[a, r] = True

        # 3. Relevant role indicators: shape (N, 30)
        self._relevant_role_indicators = np.zeros((N, C), dtype=np.bool_)
        for a in range(N):
            rel_mask = int(self._all_relevant_masks[a])
            for r in range(C):
                if (rel_mask >> r) & 1:
                    self._relevant_role_indicators[a, r] = True

        # 4. Null priors: shape (N,)
        self._null_priors = np.sum(self._relevant_role_indicators, axis=1) / float(C)

        # 5. Active module mask: shape (N, 8)
        self._active_module_mask = np.arange(8)[None, :] < self._num_modules[:, None]

    def _slice(self, key: slice | np.ndarray | list[int]) -> Self:
        """Slices the SoA arrays to return a new sub-catalog instance."""
        sub = super()._slice(key)
        sub._module_masks = self._module_masks[key]
        n_sub = len(sub)
        sub._descriptions = tuple(self._descriptions[i] for i in range(n_sub))
        sub._aliases = tuple(self._aliases[i] for i in range(n_sub))
        sub._raw_modules = tuple(self._raw_modules[i] for i in range(n_sub))
        sub._build_indicator_tensors()
        return sub

    # ------------------------------------------------------------------------------------------------------------------
    # Vectorized Array Properties
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def names(self) -> np.ndarray:
        """Contiguous 1D array of canonical archetype names as bytes."""
        return self._names

    @property
    def categories(self) -> np.ndarray:
        """Contiguous 1D array of archetype functional categories as bytes."""
        return self._categories

    @property
    def forbidden_masks(self) -> np.ndarray:
        """Contiguous 1D uint32 array of incompatible role bitmasks per archetype."""
        return self._forbidden_masks

    @property
    def cargo_masks(self) -> np.ndarray:
        """Contiguous 1D uint32 array of biosynthetic cargo role bitmasks per archetype."""
        return self._cargo_masks

    @property
    def required_masks(self) -> np.ndarray:
        """Contiguous 1D uint32 array of mandatory machinery role bitmasks per archetype."""
        return self._required_masks

    @property
    def all_relevant_masks(self) -> np.ndarray:
        """Contiguous 1D uint32 array of all relevant role bitmasks per archetype."""
        return self._all_relevant_masks

    @property
    def module_masks(self) -> np.ndarray:
        """2D uint32 array of shape (N, 8) containing alternative role bitmasks for each module."""
        return self._module_masks

    @property
    def num_modules(self) -> np.ndarray:
        """Contiguous 1D uint8 array indicating the count of mandatory modules per archetype."""
        return self._num_modules

    @property
    def descriptions(self) -> tuple[bytes, ...]:
        """Tuple of descriptive byte strings for each canonical archetype."""
        return self._descriptions

    @property
    def aliases(self) -> tuple[tuple[bytes, ...], ...]:
        """Tuple of alternative name aliases for each canonical archetype."""
        return self._aliases

    def __getattr__(self, name: str) -> LocusArchetype:
        """Delegates attribute lookups to the dynamic Index BaseIndexEnum."""
        if hasattr(LocusArchetype, name):
            res = getattr(LocusArchetype, name)
            if isinstance(res, LocusArchetype):
                return res
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # ------------------------------------------------------------------------------------------------------------------
    # Vectorized Boolean Rule Engine
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def check_forbidden(cls, presence_mask: int | np.uint32 | np.ndarray) -> np.ndarray:
        """Evaluates whether gene presence mask(s) violate forbidden role constraints.

        Args:
            presence_mask: Scalar uint32 bitmask or ndarray of shape (...).

        Returns:
            np.ndarray[bool]: Boolean array of shape (N,) or (..., N) where True indicates
            forbidden role contamination.
        """
        inst = cls._get_instance()
        if isinstance(presence_mask, (int, np.integer)):
            return (np.uint32(presence_mask) & inst._forbidden_masks) != 0
        p_arr = np.asarray(presence_mask, dtype=np.uint32)[..., None]
        return (p_arr & inst._forbidden_masks) != 0

    @classmethod
    def check_modules(cls, presence_mask: int | np.uint32 | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates module satisfaction and completeness for given gene presence mask(s).

        Args:
            presence_mask: Scalar uint32 bitmask or ndarray of shape (...).

        Returns:
            tuple[np.ndarray[bool], np.ndarray[float64]]:
                - satisfied_modules: boolean array of shape (N, 8) or (..., N, 8)
                - completeness: float array in [0.0, 1.0] of shape (N,) or (..., N)
        """
        inst = cls._get_instance()
        if isinstance(presence_mask, (int, np.integer)):
            p = np.uint32(presence_mask)
            sat = (p & inst._module_masks) != 0
            sat_active = sat & inst._active_module_mask
            sat_count = np.sum(sat_active, axis=1)
            completeness = sat_count / inst._num_modules.astype(np.float64)
            return sat_active, completeness

        p_arr = np.asarray(presence_mask, dtype=np.uint32)[..., None, None]
        sat = (p_arr & inst._module_masks) != 0
        sat_active = sat & inst._active_module_mask
        sat_count = np.sum(sat_active, axis=-1)
        completeness = sat_count / inst._num_modules.astype(np.float64)
        return sat_active, completeness

    @classmethod
    def is_complete(cls, presence_mask: int | np.uint32 | np.ndarray) -> np.ndarray:
        """Checks if presence mask satisfies all required modules with zero forbidden roles.

        Args:
            presence_mask: Scalar uint32 bitmask or ndarray of shape (...).

        Returns:
            np.ndarray[bool]: Boolean array of shape (N,) or (..., N).
        """
        _, comp = cls.check_modules(presence_mask)
        has_forbid = cls.check_forbidden(presence_mask)
        return (comp == 1.0) & (~has_forbid)

    @classmethod
    def classify_mask(cls, presence_mask: int | np.uint32) -> tuple[LocusArchetype | None, float]:
        """Selects best matching archetype for a discrete gene presence mask.

        Requires 0 forbidden violations. Selects highest completeness score.

        Args:
            presence_mask: Scalar uint32 bitmask.

        Returns:
            tuple[LocusArchetype | None, float]: (best_archetype, completeness)
        """
        inst = cls._get_instance()
        sat_active, comp = cls.check_modules(presence_mask)
        has_forbid = cls.check_forbidden(presence_mask)
        valid = [i for i in range(len(inst)) if not has_forbid[i] and comp[i] > 0.0]
        if not valid:
            return None, 0.0
        sat_count = np.sum(sat_active, axis=1)
        best_i = max(valid, key=lambda i: (comp[i], sat_count[i]))
        return LocusArchetype(best_i), float(comp[best_i])

    @classmethod
    def is_boundary_relevant(cls, role_or_probs: int | np.ndarray) -> np.ndarray:
        """Checks whether a gene role or probability vector is relevant to each archetype.

        Used for instant boundary filtering in segmentation dynamic programming.

        Args:
            role_or_probs: Integer role index, 1D probability vector (C,), or 2D matrix (M, C).

        Returns:
            np.ndarray[bool]: Boolean array of shape (N,) or (M, N).
        """
        inst = cls._get_instance()
        if isinstance(role_or_probs, (int, np.integer)):
            r = int(role_or_probs)
            if 0 <= r < 32:
                return (np.uint32(1 << r) & inst._all_relevant_masks) != 0
            return np.zeros(len(inst), dtype=np.bool_)

        arr = np.asarray(role_or_probs, dtype=np.float64)
        if arr.ndim == 1:
            if len(arr) < 30:
                p = np.pad(arr, (0, 30 - len(arr)), mode="constant")
            else:
                p = arr[:30]
            p_cluster = p @ inst._relevant_role_indicators.T
            return p_cluster > inst._null_priors
        if arr.ndim == 2:
            if arr.shape[1] < 30:
                p = np.pad(arr, ((0, 0), (0, 30 - arr.shape[1])), mode="constant")
            else:
                p = arr[:, :30]
            p_cluster = p @ inst._relevant_role_indicators.T
            return p_cluster > inst._null_priors[None, :]
        raise ValueError("role_or_probs must be an integer, 1D array, or 2D array")

    @classmethod
    def score_segment_vectorized(
        cls,
        probs: np.ndarray,
        w_comp: float = 12.0,
        w_incomp: float = 100.0,
        null_gene_llr: float = -1.5,
        noise_gate: float = 0.30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scores candidate segment across all archetypes in parallel using SIMD tensor operations.

        Args:
            probs: (M, C) array of posterior probabilities for M genes.
            w_comp: Completeness reward weight.
            w_incomp: Incompatibility penalty weight.
            null_gene_llr: Log-likelihood penalty for non-cluster genes.
            noise_gate: Softmax background probability threshold below which forbidden
                evidence is ignored.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - composite_scores: (N,) float64 composite DP segment scores
                - completeness: (N,) float64 geometric mean completeness in [0.0, 1.0]
                - incompatibility: (N,) float64 forbidden evidence in [0.0, 1.0]
        """
        inst = cls._get_instance()
        N = len(inst)
        if len(probs) == 0:
            return (
                np.full(N, -100.0, dtype=np.float64),
                np.zeros(N, dtype=np.float64),
                np.zeros(N, dtype=np.float64),
            )

        if probs.shape[1] < 30:
            p = np.pad(probs, ((0, 0), (0, 30 - probs.shape[1])), mode="constant")
        else:
            p = probs[:, :30]

        # 1. Module presence & completeness across all archetypes and modules: shape (M, N, 8)
        gene_module_p = np.tensordot(p, inst._module_role_indicators, axes=([1], [2]))
        gene_module_p = np.clip(gene_module_p, 0.0, 1.0)

        # Joint non-occurrence: prod_{g}(1 - p_g) across genes (axis 0) -> shape (N, 8)
        prob_none = np.prod(1.0 - gene_module_p, axis=0)
        module_s = np.clip(1.0 - prob_none, 0.0, 1.0)

        # Geometric mean over active modules
        has_absent = np.any((module_s < 1e-5) & inst._active_module_mask, axis=1)  # shape (N,)
        safe_s = np.maximum(1e-6, module_s)
        log_s = np.where(inst._active_module_mask, np.log(safe_s), 0.0)
        mean_log_s = np.sum(log_s, axis=1) / inst._num_modules.astype(np.float64)
        comp = np.where(has_absent, 0.0, np.clip(np.exp(mean_log_s), 0.0, 1.0))

        # 2. Incompatibility (forbidden machinery contamination): shape (M, N)
        gene_forbid_p = p @ inst._forbidden_role_indicators.T
        gene_forbid_p = np.clip(gene_forbid_p, 0.0, 1.0)
        if noise_gate > 0.0:
            gene_forbid_p = np.where(gene_forbid_p >= noise_gate, gene_forbid_p, 0.0)

        prob_no_forbid = np.prod(1.0 - gene_forbid_p, axis=0)  # shape (N,)
        incomp = np.clip(1.0 - prob_no_forbid, 0.0, 1.0)

        # 3. Log-likelihood ratio across genes: shape (M, N)
        p_cluster = p @ inst._relevant_role_indicators.T
        p0 = inst._null_priors[None, :]  # shape (1, N)
        llr_gene = np.where(
            p_cluster > p0,
            np.log(np.maximum(1e-12, p_cluster / p0)),
            null_gene_llr,
        )
        llr_sum = np.sum(llr_gene, axis=0)  # shape (N,)

        composite_scores = llr_sum + (w_comp * comp) - (w_incomp * incomp)
        return composite_scores, comp, incomp


# ----------------------------------------------------------------------------------------------------------------------
# Delegating Helper Functions for Existing API Consumers
# ----------------------------------------------------------------------------------------------------------------------

def compute_module_presence(probs: np.ndarray, module: frozenset[int] | set[int] | Sequence[int]) -> float:
    """Computes the probability that candidate segment contains at least one gene in module."""
    if len(probs) == 0 or not module:
        return 0.0

    valid_roles = [r for r in module if 0 <= r < probs.shape[1]]
    if not valid_roles:
        return 0.0

    gene_p = np.sum(probs[:, valid_roles], axis=1)
    gene_p = np.clip(gene_p, 0.0, 1.0)
    prob_none = np.prod(1.0 - gene_p)
    return float(np.clip(1.0 - prob_none, 0.0, 1.0))


def compute_completeness(probs: np.ndarray, archetype: LocusArchetype | int) -> float:
    """Calculates continuous geometric-mean completeness score for an archetype."""
    idx = int(archetype)
    _, comp, _ = Archetypes.score_segment_vectorized(probs)
    return float(comp[idx])


def compute_incompatibility(
    probs: np.ndarray,
    archetype: LocusArchetype | int,
    noise_gate: float = 0.30,
) -> float:
    """Calculates the probability that candidate segment contains forbidden machinery."""
    idx = int(archetype)
    _, _, incomp = Archetypes.score_segment_vectorized(probs, noise_gate=noise_gate)
    return float(incomp[idx])


def score_archetype_likelihood(
    probs: np.ndarray,
    archetype: LocusArchetype | int,
    w_comp: float = 12.0,
    w_incomp: float = 100.0,
    null_gene_llr: float = -1.5,
) -> tuple[float, float, float]:
    """Scores a candidate gene segment under an archetype for dynamic programming."""
    idx = int(archetype)
    scores, comp, incomp = Archetypes.score_segment_vectorized(
        probs, w_comp=w_comp, w_incomp=w_incomp, null_gene_llr=null_gene_llr
    )
    return float(scores[idx]), float(comp[idx]), float(incomp[idx])


def classify_best_archetype(
    probs: np.ndarray,
    w_comp: float = 12.0,
    w_incomp: float = 100.0,
) -> tuple[LocusArchetype, float, float, float]:
    """Evaluates candidate segment across all 6 archetypes and selects the highest scoring."""
    scores, comp, incomp = Archetypes.score_segment_vectorized(
        probs, w_comp=w_comp, w_incomp=w_incomp
    )
    best_idx = int(np.argmax(scores))
    return LocusArchetype(best_idx), float(scores[best_idx]), float(comp[best_idx]), float(incomp[best_idx])
