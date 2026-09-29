"""
Core abstract base classes and metaclasses for high-performance
Structure-of-Arrays (SoA) catalog singletons and registries.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import IntEnum
from types import GenericAlias
from typing import Any, Self

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Dynamic IntEnum Metaclass & Base for Catalog Registries
# ----------------------------------------------------------------------------------------------------------------------

class IndexMeta(type(IntEnum)):
    """Metaclass allowing IntEnum member lookups by byte-string or uppercase name."""

    _catalog: Any = None

    def __getitem__(cls, name: Any) -> Any:
        if isinstance(name, (bytes, bytearray)):
            if cls._catalog is not None:
                idx = cls._catalog.index(name)
                return cls(idx)
            canon = bytes(name).upper().decode("ascii", errors="replace")
            return super().__getitem__(canon)
        return super().__getitem__(name)


class BaseIndexEnum(IntEnum, metaclass=IndexMeta):
    """Base IntEnum supporting byte-string indexing and alias resolution."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
# Metaclass for Lazy Class-Level Singletons
# ----------------------------------------------------------------------------------------------------------------------

class CatalogMeta(type):
    """
    Metaclass enabling lazy singleton initialization and class-level SoA access.
    Delegates attribute access, indexing, iteration, membership testing, and length
    on the class directly to the singleton instance.
    """
    _instance: Any = None

    def _get_instance(cls) -> Any:
        if cls._instance is None:
            inst: Any = object.__new__(cls)
            inst.init_default()
            if getattr(inst, "Index", None) is None:
                inst._build_index_enum()
            cls._instance = inst
        return cls._instance

    def __getattribute__(cls, name: str) -> Any:
        if (name.startswith("__") and name.endswith("__")) or name in (
            "_instance", "_get_instance", "mro", "_CANONICAL_MANIFEST"
        ):
            return super().__getattribute__(name)
        return getattr(cls._get_instance(), name)

    def __getitem__(cls, key: Any) -> Any:
        if cls.__name__ == "Catalog":
            return GenericAlias(cls, key)
        return cls._get_instance()[key]

    def __len__(cls) -> int:
        return len(cls._get_instance())

    def __iter__(cls) -> Iterator[Any]:
        return iter(cls._get_instance())

    def __contains__(cls, item: Any) -> bool:
        return item in cls._get_instance()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if not args and not kwargs:
            return cls._get_instance()
        inst: Any = object.__new__(cls)
        if hasattr(inst, "_init_from_manifest"):
            inst._init_from_manifest(*args, **kwargs)
        elif hasattr(inst, "init_default"):
            inst.init_default()
        if getattr(inst, "Index", None) is None:
            inst._build_index_enum()
        return inst


# ----------------------------------------------------------------------------------------------------------------------
# Structure-of-Arrays (SoA) Catalog Base Class
# ----------------------------------------------------------------------------------------------------------------------

class Catalog(metaclass=CatalogMeta):
    """
    Abstract Structure-of-Arrays (SoA) catalog registry base class.
    Provides standardized storage, indexing, name/alias resolution, dynamic attribute
    access, slicing, vectorized entry registration, and continuous feature matrix caching.
    """
    _SOA_ARRAYS: tuple[str, ...] = ()
    _CANONICAL_MANIFEST: Any = None

    name_to_index: dict[bytes, int]
    alias_map: dict[bytes, bytes]
    _feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float32]] | None
    Index: type[BaseIndexEnum] | None = None

    def init_default(self) -> None:
        """Initializes the catalog from the canonical default manifest."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the catalog back to canonical standard entries."""
        self.init_default()

    def resolve_name(self, name_or_alias: bytes) -> bytes:
        """Resolves a bytes name or alias to canonical bytes name."""
        if not isinstance(name_or_alias, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {type(name_or_alias).__name__}")
        bname = bytes(name_or_alias)
        if bname in self.alias_map:
            return self.alias_map[bname]
        if bname in self.name_to_index:
            return bname
        bname_lower = bname.lower()
        if bname_lower in self.alias_map:
            return self.alias_map[bname_lower]
        bname_cap = bname.capitalize()
        if bname_cap in self.name_to_index:
            return bname_cap
        if bname_cap in self.alias_map:
            return self.alias_map[bname_cap]
        return bname

    def index(self, key: Any) -> int:
        """Returns the integer index for an entry name, alias, or custom lookup key."""
        if isinstance(key, (int, np.integer)):
            return int(key)
        if hasattr(key, "index") and isinstance(key.index, int):
            return key.index
        if isinstance(key, str):
            raise TypeError(f"Catalog lookup requires bytes, got str '{key}'. Use b'{key}' instead.")
        if isinstance(key, (bytes, bytearray)):
            bname = self.resolve_name(key)
            if bname in self.name_to_index:
                return self.name_to_index[bname]
        raise KeyError(f"Item '{key}' not found in {self.__class__.__name__} catalog.")

    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieves an index or value by index, name, or key, returning default if not found."""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __contains__(self, key: Any) -> bool:
        """Checks if a key (integer index, name, alias, or custom lookup) exists in the catalog."""
        if isinstance(key, (int, np.integer)):
            return 0 <= int(key) < len(self)
        if isinstance(key, str):
            return False
        try:
            _ = self.index(key)
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def __len__(self) -> int:
        """Returns the number of entries registered in the catalog."""
        if self._SOA_ARRAYS:
            arr = getattr(self, self._SOA_ARRAYS[0], None)
            if arr is not None and hasattr(arr, "__len__"):
                return len(arr)
        if hasattr(self, "name_to_index"):
            return len(self.name_to_index)
        return 0

    def __iter__(self) -> Iterator[int]:
        """Iterates over registered integer indices."""
        yield from range(len(self))

    def __getitem__(self, key: Any) -> int | Self:
        """Accesses an entry index by name or key, or slices the catalog into a sub-catalog."""
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for {self.__class__.__name__} of size {len(self)}.")
            return idx
        if isinstance(key, (slice, np.ndarray, list)):
            return self._slice(key)
        return self.index(key)

    def _slice(self, key: Any) -> Self:
        """Slices the SoA arrays to return a new sub-catalog instance."""
        sub = object.__new__(self.__class__)
        for arr_name in self._SOA_ARRAYS:
            arr = getattr(self, arr_name, None)
            if arr is not None:
                setattr(sub, arr_name, arr[key])
        sub.name_to_index = self.name_to_index
        sub.alias_map = self.alias_map
        sub.Index = getattr(self, "Index", None)
        return sub

    def _append_soa_entry(
        self,
        name: bytes,
        aliases: Sequence[bytes] = (),
        **arrays: Any,
    ) -> int:
        """
        Vectorized registration helper. Appends scalar values to named SoA arrays,
        registers the entry name and aliases into lookup maps.
        Returns the newly assigned integer index.
        """
        new_idx = len(self)
        for arr_name, scalar_val in arrays.items():
            arr = getattr(self, arr_name)
            setattr(self, arr_name, np.append(arr, scalar_val))

        self.name_to_index[name] = new_idx

        for al in aliases:
            if not isinstance(al, (bytes, bytearray)):
                raise TypeError(f"alias must be bytes, got {type(al).__name__}")
            bal = bytes(al)
            self.alias_map[bal] = name

        self._feature_matrix = None
        self._build_index_enum()
        return new_idx

    def _build_index_enum(self) -> type[BaseIndexEnum] | None:
        """Dynamically constructs an IntEnum representing entry indices, abbreviations, and aliases.

        Canonical entry names (UPPERCASE) take highest priority, followed by abbreviations
        and clean aliases.
        Supports byte-string indexing: Index[b"FLIPPASE"] == 0, Index[b"wzx"] == 0, and dot-access Index.FLIPPASE.
        """
        n_entries = len(self)
        if not getattr(self, "_ENABLE_INDEX_ENUM", True) or n_entries > 5000:
            return None

        members: dict[str, int] = {}
        names_arr = getattr(self, "names", getattr(self, "_names", None))
        abbrs_arr = getattr(self, "abbreviations", getattr(self, "_abbreviations", None))

        # 1. Canonical entry names (highest priority)
        if names_arr is not None:
            for i, name_val in enumerate(names_arr):
                b_name = bytes(name_val) if isinstance(name_val, (bytes, bytearray)) else str(name_val).encode("ascii")
                if b_name.isascii():
                    s_name = b_name.decode("ascii").upper()
                    if s_name.isidentifier():
                        members[s_name] = i

        # 2. Abbreviations (e.g. b"Wzx", b"GT")
        if abbrs_arr is not None:
            for i, abbr_val in enumerate(abbrs_arr):
                if abbr_val:
                    b_abbr = bytes(abbr_val) if isinstance(abbr_val, (bytes, bytearray)) else str(abbr_val).encode("ascii")
                    if b_abbr.isascii():
                        s_abbr = b_abbr.decode("ascii").upper()
                        if s_abbr.isidentifier() and s_abbr not in members:
                            members[s_abbr] = i

        # 3. Clean aliases from self.alias_map
        if hasattr(self, "alias_map") and hasattr(self, "name_to_index"):
            for al_raw, canon_raw in self.alias_map.items():
                if canon_raw in self.name_to_index:
                    idx = self.name_to_index[canon_raw]
                    b_al = bytes(al_raw) if isinstance(al_raw, (bytes, bytearray)) else str(al_raw).encode("ascii")
                    if b_al.isascii():
                        s_al = b_al.decode("ascii").upper()
                        if s_al.isidentifier() and s_al not in members:
                            members[s_al] = idx

        enum_cls_name = f"{self.__class__.__name__}Index"
        enum_cls = BaseIndexEnum(enum_cls_name, members)
        enum_cls._catalog = self
        self.Index = enum_cls
        type(self).Index = enum_cls
        return enum_cls

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)} entries)"
