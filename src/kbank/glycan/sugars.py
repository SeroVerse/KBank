r"""Module for defining and representing known monosaccharides and carbohydrate modifiers.
Provides a unified, type-safe registry (Sugars) with SoA vectorized arrays,
seamless str/bytes alias resolution.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from kbank.core.models import Catalog


# ----------------------------------------------------------------------------------------------------------------------
# Canonical Monosaccharide Database Manifest
# ----------------------------------------------------------------------------------------------------------------------

# (class, name, is_ketose, is_deoxy, is_amino, is_n_acetyl, is_abstract, mass, axial_mask, natural_chirality, aliases)
_CANONICAL_SUGARS: list[tuple[bytes, bytes, bool, bool, bool, bool, bool, float, int, int, tuple[bytes, ...]]] = [
    # Hexoses - Axial masks: C2 (1), C3 (2), C4 (4)
    (b"Hex", b"Glc", False, False, False, False, False, 180.156, 0b000, 1, (b"Glcp", b"Glcf")),
    (b"Hex", b"Man", False, False, False, False, False, 180.156, 0b001, 1, (b"Manp",)),
    (b"Hex", b"Gal", False, False, False, False, False, 180.156, 0b100, 1, (b"Galp", b"Galf")),
    (b"Hex", b"Gul", False, False, False, False, False, 180.156, 0b110, 1, ()),
    (b"Hex", b"Alt", False, False, False, False, False, 180.156, 0b011, 2, ()),
    (b"Hex", b"All", False, False, False, False, False, 180.156, 0b010, 1, ()),
    (b"Hex", b"Tal", False, False, False, False, False, 180.156, 0b101, 1, ()),
    (b"Hex", b"Ido", False, False, False, False, False, 180.156, 0b111, 2, ()),

    # HexNAc
    (b"HexNAc", b"GlcNAc", False, False, True, True, False, 221.208, 0b000, 1, (b"GlcpNAc",)),
    (b"HexNAc", b"GalNAc", False, False, True, True, False, 221.208, 0b100, 1, (b"GalpNAc",)),
    (b"HexNAc", b"ManNAc", False, False, True, True, False, 221.208, 0b001, 1, (b"ManpNAc",)),
    (b"HexNAc", b"GulNAc", False, False, True, True, False, 221.208, 0b110, 1, ()),
    (b"HexNAc", b"AltNAc", False, False, True, True, False, 221.208, 0b011, 2, ()),
    (b"HexNAc", b"AllNAc", False, False, True, True, False, 221.208, 0b010, 1, ()),
    (b"HexNAc", b"TalNAc", False, False, True, True, False, 221.208, 0b101, 1, ()),
    (b"HexNAc", b"IdoNAc", False, False, True, True, False, 221.208, 0b111, 2, ()),

    # Hexosamines
    (b"HexN", b"GlcN", False, False, True, False, False, 179.17, 0b000, 1, (b"GlcpN",)),
    (b"HexN", b"ManN", False, False, True, False, False, 179.17, 0b001, 1, (b"ManpN",)),
    (b"HexN", b"GalN", False, False, True, False, False, 179.17, 0b100, 1, (b"GalpN",)),
    (b"HexN", b"GulN", False, False, True, False, False, 179.17, 0b110, 1, ()),
    (b"HexN", b"AltN", False, False, True, False, False, 179.17, 0b011, 2, ()),
    (b"HexN", b"AllN", False, False, True, False, False, 179.17, 0b010, 1, ()),
    (b"HexN", b"TalN", False, False, True, False, False, 179.17, 0b101, 1, ()),
    (b"HexN", b"IdoN", False, False, True, False, False, 179.17, 0b111, 2, ()),

    # Hexuronates
    (b"HexA", b"GlcA", False, False, False, False, False, 194.139, 0b000, 1, (b"GlcpA",)),
    (b"HexA", b"ManA", False, False, False, False, False, 194.139, 0b001, 1, (b"ManpA",)),
    (b"HexA", b"GalA", False, False, False, False, False, 194.139, 0b100, 1, (b"GalpA",)),
    (b"HexA", b"GulA", False, False, False, False, False, 194.139, 0b110, 1, ()),
    (b"HexA", b"AltA", False, False, False, False, False, 194.139, 0b011, 2, ()),
    (b"HexA", b"AllA", False, False, False, False, False, 194.139, 0b010, 1, ()),
    (b"HexA", b"TalA", False, False, False, False, False, 194.139, 0b101, 1, ()),
    (b"HexA", b"IdoA", False, False, False, False, False, 194.139, 0b111, 2, ()),

    # Deoxyhexoses
    (b"dHex", b"Fuc", False, True, False, False, False, 164.156, 0b100, 2, (b"Fucp",)),
    (b"dHex", b"Qui", False, True, False, False, False, 164.156, 0b000, 1, (b"Quip",)),
    (b"dHex", b"Rha", False, True, False, False, False, 164.156, 0b001, 2, (b"Rhap",)),
    (b"dHex", b"6dGul", False, True, False, False, False, 164.156, 0b110, 1, (b"six_dGul", b"6-dGul")),
    (b"dHex", b"6dAlt", False, True, False, False, False, 164.156, 0b011, 2, ()),
    (b"dHex", b"6dTal", False, True, False, False, False, 164.156, 0b101, 1, ()),
    (b"dHex", b"6dxylHex", False, True, False, False, False, 164.156, 0b000, 1, ()),

    # DeoxyhexNAc
    (b"dHexNAc", b"QuiNAc", False, True, True, True, False, 205.2, 0b000, 1, (b"QuipNAc",)),
    (b"dHexNAc", b"RhaNAc", False, True, True, True, False, 205.2, 0b001, 2, ()),
    (b"dHexNAc", b"6dAltNAc", False, True, True, True, False, 205.2, 0b011, 2, ()),
    (b"dHexNAc", b"6dTalNAc", False, True, True, True, False, 205.2, 0b101, 1, ()),
    (b"dHexNAc", b"FucNAc", False, True, True, True, False, 205.2, 0b100, 2, (b"FucpNAc",)),

    # Deoxyhexosamines
    (b"dHexN", b"FucN", False, True, True, False, False, 163.17, 0b100, 2, (b"FucpN",)),
    (b"dHexN", b"QuiN", False, True, True, False, False, 163.17, 0b000, 1, (b"QuipN",)),
    (b"dHexN", b"RhaN", False, True, True, False, False, 163.17, 0b001, 2, (b"RhapN",)),
    (b"dHexN", b"QuiN4N", False, True, True, False, False, 163.17, 0b000, 1, (b"QuipN4N",)),
    (b"dHexN", b"FucN4N", False, True, True, False, False, 163.17, 0b100, 2, (b"FucpN4N",)),
    (b"dHexN", b"RhaN3N", False, True, True, False, False, 162.19, 0b001, 2, (b"RhapN3N", b"Rha2N3N", b"Rhap2N3N", b"RhaNAc3NAc", b"RhapNAc3NAc")),

    # Aminohexuronates
    (b"HexNA", b"GalNA", False, False, True, False, False, 193.15, 0b100, 1, (b"GalpNA", b"GalNAcA", b"GalpNAcA")),
    (b"HexNA", b"GlcNA", False, False, True, False, False, 193.15, 0b000, 1, (b"GlcpNA", b"GlcNAcA", b"GlcpNAcA")),
    (b"HexNA", b"ManNA", False, False, True, False, False, 193.15, 0b001, 1, (b"ManpNA", b"ManNAcA", b"ManpNAcA")),
    (b"HexNA", b"GlcN3NA", False, False, True, False, False, 193.15, 0b000, 1, (b"GlcpN3NA",)),

    # Dideoxyhexoses
    (b"ddHex", b"Oli", False, True, False, False, False, 148.16, 0b000, 1, ()),
    (b"ddHex", b"Tyv", False, True, False, False, False, 148.16, 0b001, 1, ()),
    (b"ddHex", b"Abe", False, True, False, False, False, 148.16, 0b100, 1, ()),
    (b"ddHex", b"Par", False, True, False, False, False, 148.16, 0b000, 1, ()),
    (b"ddHex", b"Dig", False, True, False, False, False, 148.16, 0b000, 1, ()),
    (b"ddHex", b"Col", False, True, False, False, False, 148.16, 0b100, 2, ()),

    # Pentoses
    (b"Pen", b"Ara", False, False, False, False, False, 150.13, 0b000, 2, (b"Araf",)),
    (b"Pen", b"Lyx", False, False, False, False, False, 150.13, 0b001, 1, ()),
    (b"Pen", b"Xyl", False, False, False, False, False, 150.13, 0b000, 1, ()),
    (b"Pen", b"Rib", False, False, False, False, False, 150.13, 0b010, 1, (b"Ribp",)),
    (b"Pen", b"Api", False, False, False, False, False, 150.13, 0b000, 2, ()),

    # Nonulosonic & Higher Acids
    (b"NeuAc", b"Neu5Ac", True, False, True, True, False, 309.27, 0, 0, (b"Sia",)),
    (b"NeuGc", b"Neu5Gc", True, False, True, True, False, 325.27, 0, 0, ()),
    (b"gNeu", b"Neu", True, False, True, False, False, 267.24, 0, 0, ()),
    (b"gKdn", b"Kdn", True, False, False, False, False, 268.22, 0, 0, (b"Kdnp",)),
    (b"gKdo", b"Kdo", True, True, False, False, False, 238.2, 0, 0, ()),
    (b"Oct", b"Ko", True, False, False, False, False, 254.2, 0, 0, ()),
    (b"gPse", b"Pse", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"gPse", b"4ePse", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"gPse", b"8ePse", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"gLeg", b"Leg", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"g4eLeg", b"4eLeg", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"gLeg", b"8eLeg", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"gAci", b"Aci", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"gAci", b"8eAci", True, True, True, False, False, 292.29, 0, 0, ()),
    (b"gBac", b"Bac", False, True, True, False, False, 228.25, 0, 0, ()),
    (b"Hep", b"LDmanHep", False, False, False, False, False, 210.18, 0, 0, ()),
    (b"Hep", b"DDmanHep", False, False, False, False, False, 210.18, 0, 0, ()),
    (b"HepA", b"Dha", True, False, False, False, False, 224.17, 0, 0, ()),
    (b"MurAc", b"MurNAc", False, False, True, True, False, 293.27, 0, 1, ()),
    (b"MurGc", b"MurNGc", False, False, True, True, False, 309.27, 0, 1, ()),
    (b"gMur", b"Mur", False, False, True, False, False, 251.23, 0, 1, ()),

    # Ketoses
    (b"Hex", b"Fru", True, False, False, False, False, 180.156, 0, 1, ()),
    (b"Hex", b"Tag", True, False, False, False, False, 180.156, 0, 1, ()),
    (b"Hex", b"Sor", True, False, False, False, False, 180.156, 0, 2, ()),
    (b"Hex", b"Psi", True, False, False, False, False, 180.156, 0, 1, ()),

    # Abstract Wildcards
    (b"Sug", b"Sug", False, False, False, False, True, 180.0, 0, 0, ()),
    (b"Hex", b"Hex", False, False, False, False, True, 180.156, 0, 0, ()),
    (b"HexNAc", b"HexNAc", False, False, True, True, True, 221.208, 0, 0, ()),
    (b"HexN", b"HexN", False, False, True, False, True, 179.17, 0, 0, ()),
    (b"HexA", b"HexA", False, False, False, False, True, 194.139, 0, 0, ()),
    (b"Pen", b"Pen", False, False, False, False, True, 150.13, 0, 0, ()),
    (b"Subst", b"Subst", False, False, False, False, True, 0.0, 0, 0, ()),
    (b"Hex", b"ExoticSugar", False, False, False, False, True, 200.0, 0, 0, ()),

    # Modifiers & Acyl Substituents
    (b"Mod", b"OAc", False, False, False, False, False, 43.04, 0, 0, (b"Ac",)),
    (b"Mod", b"Pyr", False, False, False, False, False, 87.05, 0, 0, ()),
    (b"Mod", b"P", False, False, False, False, False, 79.98, 0, 0, ()),
    (b"Mod", b"PEtn", False, False, True, False, False, 124.08, 0, 0, ()),
    (b"Mod", b"OMe", False, False, False, False, False, 15.03, 0, 0, (b"Me",)),
    (b"Mod", b"N", False, False, True, False, False, 16.02, 0, 0, ()),
    (b"Mod", b"LAla", False, False, True, False, False, 71.08, 0, 2, ()),
    (b"Mod", b"DAla", False, False, True, False, False, 71.08, 0, 1, (b"dAla",)),
    (b"Mod", b"Ala", False, False, True, False, False, 71.08, 0, 0, ()),
    (b"Mod", b"Fo", False, False, False, False, False, 29.02, 0, 0, ()),
    (b"Mod", b"Lac", False, False, False, False, False, 73.07, 0, 0, ()),
    (b"Mod", b"Gro", False, False, False, False, False, 73.09, 0, 0, ()),
    (b"Mod", b"ol", False, False, False, False, False, 0.0, 0, 0, ()),
    (b"Mod", b"Cho", False, False, True, False, False, 104.17, 0, 0, ()),
    (b"Mod", b"Lipid", False, False, False, False, False, 200.0, 0, 0, ()),
    (b"Mod", b"AminoAcid", False, False, True, False, False, 100.0, 0, 0, ()),
    (b"Mod", b"Et", False, False, False, False, False, 29.06, 0, 0, ()),
    (b"Mod", b"EtN", False, False, True, False, False, 44.07, 0, 0, ()),
    (b"Mod", b"LI", False, False, False, False, False, 0.0, 0, 0, ()),
    (b"Mod", b"Am", False, False, True, False, False, 16.02, 0, 0, ()),
    (b"Mod", b"Cm", False, False, False, False, False, 58.04, 0, 0, ()),
    (b"Mod", b"Ser", False, False, True, False, False, 87.08, 0, 0, ()),
    (b"Mod", b"Thr", False, False, True, False, False, 101.1, 0, 0, ()),
    (b"Mod", b"nucU", False, False, False, False, False, 244.2, 0, 0, ()),
    (b"Mod", b"CER", False, False, False, False, False, 500.0, 0, 0, ()),
    (b"Mod", b"3HOMyr", False, False, False, False, False, 243.37, 0, 0, ()),
    (b"Mod", b"3HOBut", False, False, False, False, False, 103.12, 0, 0, ()),
    (b"Mod", b"3HODco", False, False, False, False, False, 187.28, 0, 0, ()),
    (b"Mod", b"3HOLau", False, False, False, False, False, 215.33, 0, 0, ()),
    (b"Mod", b"3HOSte", False, False, False, False, False, 299.47, 0, 0, ()),
    (b"Mod", b"3HOCap", False, False, False, False, False, 159.23, 0, 0, ()),
    (b"Mod", b"Lau", False, False, False, False, False, 199.33, 0, 0, ()),
    (b"Mod", b"Myr", False, False, False, False, False, 227.39, 0, 0, ()),
    (b"Mod", b"Ste", False, False, False, False, False, 283.49, 0, 0, ()),
    (b"Mod", b"S", False, False, False, False, False, 0.0, 0, 0, ()),
]


# ----------------------------------------------------------------------------------------------------------------------
# Canonical Activated Nucleotide-Sugar & Metabolite Donor Precursors
# ----------------------------------------------------------------------------------------------------------------------

_SPECIFIC_DONOR_PRECURSORS: dict[bytes, tuple[bytes, bytes]] = {
    # Core monosaccharides with distinct nucleotide carriers
    b"Man": (b"GDP-Man", b"GDP"),
    b"ManA": (b"GDP-ManA", b"GDP"),
    b"Fuc": (b"GDP-Fuc", b"GDP"),
    b"FucNAc": (b"GDP-FucNAc", b"GDP"),
    b"FucN": (b"GDP-FucN", b"GDP"),
    b"FucN4N": (b"GDP-FucN4N", b"GDP"),
    # Dideoxyhexoses (Salmonella/Yersinia CDP-pathways)
    b"Tyv": (b"CDP-Tyv", b"CDP"),
    b"Abe": (b"CDP-Abe", b"CDP"),
    b"Par": (b"CDP-Par", b"CDP"),
    b"Col": (b"CDP-Col", b"CDP"),
    # Nonulosonic & higher acidic sugars (CMP activation)
    b"Neu": (b"CMP-Neu5Ac", b"CMP"),
    b"Neu5Ac": (b"CMP-Neu5Ac", b"CMP"),
    b"Neu5Gc": (b"CMP-Neu5Gc", b"CMP"),
    b"Kdn": (b"CMP-Kdn", b"CMP"),
    b"Kdo": (b"CMP-Kdo", b"CMP"),
    b"Ko": (b"CMP-Ko", b"CMP"),
    b"Pse": (b"CMP-Pse5Ac7Ac", b"CMP"),
    b"4ePse": (b"CMP-Pse5Ac7Ac", b"CMP"),
    b"8ePse": (b"CMP-Pse5Ac7Ac", b"CMP"),
    b"Leg": (b"CMP-Leg5Ac7Ac", b"CMP"),
    b"4eLeg": (b"CMP-Leg5Ac7Ac", b"CMP"),
    b"8eLeg": (b"CMP-Leg5Ac7Ac", b"CMP"),
    b"Aci": (b"CMP-Aci", b"CMP"),
    b"8eAci": (b"CMP-Aci", b"CMP"),
    # Heptoses (ADP activation)
    b"LDmanHep": (b"ADP-Hep", b"ADP"),
    b"DDmanHep": (b"ADP-Hep", b"ADP"),
    b"Dha": (b"ADP-Dha", b"ADP"),
    # Bacillosamine & Muramic acids
    b"Bac": (b"UDP-Bac", b"UDP"),
    b"MurNAc": (b"UDP-MurNAc", b"UDP"),
    b"MurNGc": (b"UDP-MurNGc", b"UDP"),
    b"Mur": (b"UDP-MurNAc", b"UDP"),
    # Modifiers & Acyl Donors
    b"OAc": (b"Acetyl-CoA", b"CoA"),
    b"Ac": (b"Acetyl-CoA", b"CoA"),
    b"Pyr": (b"PEP", b"PEP"),
    b"P": (b"ATP", b"ATP"),
    b"PEtn": (b"CDP-Etn", b"CDP"),
    b"Cho": (b"CDP-Cho", b"CDP"),
    b"Gro": (b"CDP-Gro", b"CDP"),
    b"DAla": (b"Ala", b"Ala"),
    b"LAla": (b"Ala", b"Ala"),
    b"Ala": (b"Ala", b"Ala"),
    b"3HOBut": (b"3HOBut-CoA", b"CoA"),
    b"3HOMyr": (b"3HOMyr-ACP", b"ACP"),
    b"3HODco": (b"3HODco-ACP", b"ACP"),
    b"3HOLau": (b"3HOLau-ACP", b"ACP"),
    b"3HOSte": (b"3HOSte-ACP", b"ACP"),
    b"3HOCap": (b"3HOCap-ACP", b"ACP"),
    b"Lau": (b"Lau-ACP", b"ACP"),
    b"Myr": (b"Myr-ACP", b"ACP"),
    b"Ste": (b"Ste-ACP", b"ACP"),
    b"OMe": (b"SAM", b"SAM"),
    b"Me": (b"SAM", b"SAM"),
    b"S": (b"PAPS", b"PAPS"),
    b"Fo": (b"Formyl-THF", b"THF"),
    b"Lac": (b"Lactoyl-CoA", b"CoA"),
}


def infer_donor_and_carrier(name: bytes, sugar_class: bytes) -> tuple[bytes, bytes]:
    """Infers standard activated nucleotide-sugar donor precursor and carrier for a sugar."""
    if name in _SPECIFIC_DONOR_PRECURSORS:
        return _SPECIFIC_DONOR_PRECURSORS[name]
    if sugar_class in (b"dHex", b"dHexNAc", b"dHexN", b"ddHex"):
        return b"dTDP-%b" % name, b"dTDP"
    if sugar_class == b"Mod":
        return b"%b-" % name, b"Donor"
    return b"UDP-%b" % name, b"UDP"


# ----------------------------------------------------------------------------------------------------------------------
# Sugars Registry & SoA Database Class
# ----------------------------------------------------------------------------------------------------------------------

class Sugars(Catalog):
    """
    High-performance Structure-of-Arrays (SoA) db for monosaccharides and carbohydrate modifiers.
    Acts as a lazy class-level singleton (Sugars['Glc'], Sugars.names) with direct contiguous 1D arrays.
    """
    _CANONICAL_MANIFEST = _CANONICAL_SUGARS
    _SOA_ARRAYS = (
        'names',
        'classes',
        'is_ketose',
        'is_deoxy',
        'is_amino',
        'is_n_acetyl',
        'is_abstract',
        'mass',
        'axial_mask',
        'charge',
        'natural_chirality',
        'donor_precursor',
        'carrier',
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Vectorized SoA Array Fields with Strict Dimensional & Dtype Type Hints
    # ------------------------------------------------------------------------------------------------------------------
    names: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    classes: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    is_ketose: np.ndarray[tuple[int], np.dtype[np.bool_]]
    is_deoxy: np.ndarray[tuple[int], np.dtype[np.bool_]]
    is_amino: np.ndarray[tuple[int], np.dtype[np.bool_]]
    is_n_acetyl: np.ndarray[tuple[int], np.dtype[np.bool_]]
    is_abstract: np.ndarray[tuple[int], np.dtype[np.bool_]]
    mass: np.ndarray[tuple[int], np.dtype[np.float32]]
    axial_mask: np.ndarray[tuple[int], np.dtype[np.uint8]]
    charge: np.ndarray[tuple[int], np.dtype[np.int8]]
    natural_chirality: np.ndarray[tuple[int], np.dtype[np.uint8]]
    donor_precursor: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    carrier: np.ndarray[tuple[int], np.dtype[np.bytes_]]

    name_to_index: dict[bytes, int]
    alias_map: dict[bytes, bytes]
    _aliases: tuple[tuple[bytes, ...], ...]

    @property
    def c2_ax(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        return (self.axial_mask & 1) != 0

    @property
    def c3_ax(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        return (self.axial_mask & 2) != 0

    @property
    def c4_ax(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        return (self.axial_mask & 4) != 0

    def init_default(self) -> None:
        self._init_from_manifest(self._CANONICAL_MANIFEST)

    def _init_from_manifest(
        self,
        manifest: Sequence[tuple],
        alias_map: dict[bytes, bytes] | None = None,
    ) -> None:
        N = len(manifest)
        self.names = np.empty(N, dtype="S16")
        self.classes = np.empty(N, dtype="S16")
        self.is_ketose = np.empty(N, dtype=np.bool_)
        self.is_deoxy = np.empty(N, dtype=np.bool_)
        self.is_amino = np.empty(N, dtype=np.bool_)
        self.is_n_acetyl = np.empty(N, dtype=np.bool_)
        self.is_abstract = np.empty(N, dtype=np.bool_)
        self.mass = np.empty(N, dtype=np.float32)
        self.axial_mask = np.empty(N, dtype=np.uint8)
        self.charge = np.empty(N, dtype=np.int8)
        self.natural_chirality = np.empty(N, dtype=np.uint8)
        self.donor_precursor = np.empty(N, dtype="S24")
        self.carrier = np.empty(N, dtype="S8")

        self.name_to_index = {}
        self.alias_map = dict(alias_map) if alias_map is not None else {}
        aliases_list: list[tuple[bytes, ...]] = []

        for i, item in enumerate(manifest):
            (s_class, name, ket, d, am, nac, abs_f, m, ax, chi, aliases) = item
            self.names[i] = name
            self.classes[i] = s_class
            self.is_ketose[i] = ket
            self.is_deoxy[i] = d
            self.is_amino[i] = am
            self.is_n_acetyl[i] = nac
            self.is_abstract[i] = abs_f
            self.mass[i] = m
            self.axial_mask[i] = ax
            self.natural_chirality[i] = chi

            # Infer activated donor precursor and carrier nucleotide/cofactor
            prec, carr = infer_donor_and_carrier(name, s_class)
            self.donor_precursor[i] = prec
            self.carrier[i] = carr

            # Infer Charge (-1 for Uronates/Phosphates/Sulfates, +1 for Free Amines/PEtn)
            ch = 0
            if s_class in (b"HexA", b"HepA", b"HexNA"):
                ch = -1
            elif name in (b"Neu5Ac", b"Neu5Gc", b"Neu", b"Kdn", b"Kdo", b"Pse", b"Leg", b"Aci", b"P", b"Pyr"):
                ch = -1
            elif s_class in (b"HexN", b"dHexN") or name in (b"PEtn", b"EtN", b"N", b"Cho"):
                ch = 1
            self.charge[i] = ch

            self.name_to_index[name] = i

            for al in aliases:
                self.alias_map[al] = name

            aliases_list.append(tuple(aliases))

        self._aliases = tuple(aliases_list)
        self._feature_matrix = None

    # ------------------------------------------------------------------------------------------------------------------
    # Feature Matrix for ML & Embeddings
    # ------------------------------------------------------------------------------------------------------------------
    def to_feature_matrix(self) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """
        Extracts a normalized, continuous feature representation for each sugar.
        (Columns: is_ketose, is_deoxy, is_amino, is_n_acetyl, is_abstract, c2_ax, c3_ax, c4_ax, charge, norm_mass).
        """
        if self._feature_matrix is not None:
            return self._feature_matrix

        norm_mass = self.mass / 350.0  # normalize masses around [0, 1]
        features = np.column_stack([
            self.is_ketose.astype(np.float32),
            self.is_deoxy.astype(np.float32),
            self.is_amino.astype(np.float32),
            self.is_n_acetyl.astype(np.float32),
            self.is_abstract.astype(np.float32),
            self.c2_ax.astype(np.float32),
            self.c3_ax.astype(np.float32),
            self.c4_ax.astype(np.float32),
            self.charge.astype(np.float32),
            norm_mass.astype(np.float32),
        ])
        self._feature_matrix = features
        return features


