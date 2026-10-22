"""Monosaccharide and carbohydrate modifier registry with precomputed SoA physicochemical features."""

from __future__ import annotations

from collections.abc import Iterator
from enum import IntEnum

import numpy as np

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
    (b"dHex", b"Fuc", False, True, False, False, False, 164.156, 0b100, 2, (b"Fucp", b"Fucf")),
    (b"dHex", b"Qui", False, True, False, False, False, 164.156, 0b000, 1, (b"Quip",)),
    (b"dHex", b"Rha", False, True, False, False, False, 164.156, 0b001, 2, (b"Rhap", b"Rhaf")),
    (b"dHex", b"6dGul", False, True, False, False, False, 164.156, 0b110, 1, (b"six_dGul", b"6-dGul", b"6dGulp")),
    (
        b"dHex",
        b"6dAlt",
        False,
        True,
        False,
        False,
        False,
        164.156,
        0b011,
        2,
        (b"6dAltp", b"6dAltf", b"six_dAlt", b"6-dAlt"),
    ),
    (
        b"dHex",
        b"6dTal",
        False,
        True,
        False,
        False,
        False,
        164.156,
        0b101,
        1,
        (b"6dTalp", b"6dTalf", b"six_dTal", b"6-dTal"),
    ),
    (b"dHex", b"6dxylHex", False, True, False, False, False, 164.156, 0b000, 1, (b"6dxylHexp", b"dHex", b"dHexp")),
    # DeoxyhexNAc
    (
        b"dHexNAc",
        b"QuiNAc",
        False,
        True,
        True,
        True,
        False,
        205.2,
        0b000,
        1,
        (b"QuipNAc", b"Qui3NAc", b"Quip3NAc", b"Qui4NAc", b"Quip4NAc"),
    ),
    (b"dHexNAc", b"RhaNAc", False, True, True, True, False, 205.2, 0b001, 2, (b"RhapNAc", b"Rha3NAc", b"Rhap3NAc")),
    (b"dHexNAc", b"6dAltNAc", False, True, True, True, False, 205.2, 0b011, 2, (b"6dAltpNAc",)),
    (b"dHexNAc", b"6dTalNAc", False, True, True, True, False, 205.2, 0b101, 1, (b"6dTalpNAc",)),
    (
        b"dHexNAc",
        b"FucNAc",
        False,
        True,
        True,
        True,
        False,
        205.2,
        0b100,
        2,
        (b"FucpNAc", b"Fuc3NAc", b"Fucp3NAc", b"Fuc4NAc", b"Fucp4NAc"),
    ),
    # Deoxyhexosamines
    (
        b"dHexN",
        b"FucN",
        False,
        True,
        True,
        False,
        False,
        163.17,
        0b100,
        2,
        (b"FucpN", b"Fuc3N", b"Fucp3N", b"Fuc4N", b"Fucp4N"),
    ),
    (
        b"dHexN",
        b"QuiN",
        False,
        True,
        True,
        False,
        False,
        163.17,
        0b000,
        1,
        (b"QuipN", b"Qui3N", b"Quip3N", b"Qui4N", b"Quip4N"),
    ),
    (b"dHexN", b"RhaN", False, True, True, False, False, 163.17, 0b001, 2, (b"RhapN", b"Rha3N", b"Rhap3N")),
    (
        b"dHexN",
        b"QuiN4N",
        False,
        True,
        True,
        False,
        False,
        163.17,
        0b000,
        1,
        (b"QuipN4N", b"Qui2N4N", b"Quip2N4N", b"QuiNAc4NAc", b"QuipNAc4NAc"),
    ),
    (
        b"dHexN",
        b"FucN4N",
        False,
        True,
        True,
        False,
        False,
        163.17,
        0b100,
        2,
        (b"FucpN4N", b"Fuc2N4N", b"Fucp2N4N", b"FucNAc4NAc", b"FucpNAc4NAc"),
    ),
    (
        b"dHexN",
        b"RhaN3N",
        False,
        True,
        True,
        False,
        False,
        162.19,
        0b001,
        2,
        (b"RhapN3N", b"Rha2N3N", b"Rhap2N3N", b"RhaNAc3NAc", b"RhapNAc3NAc"),
    ),
    # Aminohexuronates
    (b"HexNA", b"GalNA", False, False, True, False, False, 193.15, 0b100, 1, (b"GalpNA", b"GalNAcA", b"GalpNAcA")),
    (b"HexNA", b"GlcNA", False, False, True, False, False, 193.15, 0b000, 1, (b"GlcpNA", b"GlcNAcA", b"GlcpNAcA")),
    (b"HexNA", b"ManNA", False, False, True, False, False, 193.15, 0b001, 1, (b"ManpNA", b"ManNAcA", b"ManpNAcA")),
    (
        b"HexNA",
        b"GlcN3NA",
        False,
        False,
        True,
        False,
        False,
        193.15,
        0b000,
        1,
        (b"GlcpN3NA", b"GlcNAc3NAcA", b"GlcpNAc3NAcA"),
    ),
    # Dideoxyhexoses
    (
        b"ddHex",
        b"Oli",
        False,
        True,
        False,
        False,
        False,
        148.16,
        0b000,
        1,
        (b"Olip", b"Yer", b"Yerp", b"ddHex", b"ddHexp"),
    ),
    (b"ddHex", b"Tyv", False, True, False, False, False, 148.16, 0b001, 1, (b"Tyvp", b"Tyvf")),
    (b"ddHex", b"Abe", False, True, False, False, False, 148.16, 0b100, 1, (b"Abep", b"Abef")),
    (b"ddHex", b"Par", False, True, False, False, False, 148.16, 0b000, 1, (b"Parp", b"Asc", b"Ascp")),
    (b"ddHex", b"Dig", False, True, False, False, False, 148.16, 0b000, 1, (b"Digp",)),
    (b"ddHex", b"Col", False, True, False, False, False, 148.16, 0b100, 2, (b"Colp", b"Colf")),
    # Pentoses
    (b"Pen", b"Ara", False, False, False, False, False, 150.13, 0b000, 2, (b"Araf", b"Arap")),
    (b"Pen", b"Lyx", False, False, False, False, False, 150.13, 0b001, 1, (b"Lyxp", b"Lyxf")),
    (b"Pen", b"Xyl", False, False, False, False, False, 150.13, 0b000, 1, (b"Xylp", b"Xylf", b"Xul", b"Xulp", b"Xulf")),
    (b"Pen", b"Rib", False, False, False, False, False, 150.13, 0b010, 1, (b"Ribp", b"Ribf", b"Rul", b"Rulp", b"Rulf")),
    (b"Pen", b"Api", False, False, False, False, False, 150.13, 0b000, 2, (b"Apif", b"Apip")),
    # Nonulosonic & Higher Acids
    (b"NeuAc", b"Neu5Ac", True, False, True, True, False, 309.27, 0, 0, (b"Sia", b"Neup5Ac")),
    (b"NeuGc", b"Neu5Gc", True, False, True, True, False, 325.27, 0, 0, (b"Neup5Gc",)),
    (b"gNeu", b"Neu", True, False, True, False, False, 267.24, 0, 0, (b"Neup",)),
    (b"gKdn", b"Kdn", True, False, False, False, False, 268.22, 0, 0, (b"Kdnp",)),
    (b"gKdo", b"Kdo", True, True, False, False, False, 238.2, 0, 0, (b"Kdop", b"Kdof", b"KDO")),
    (b"Oct", b"Ko", True, False, False, False, False, 254.2, 0, 0, (b"Kop", b"KO")),
    (b"gPse", b"Pse", True, True, True, False, False, 292.29, 0, 0, (b"Psep", b"Pse5Ac7Ac", b"Psep5Ac7Ac")),
    (b"gPse", b"4ePse", True, True, True, False, False, 292.29, 0, 0, (b"4ePsep",)),
    (b"gPse", b"8ePse", True, True, True, False, False, 292.29, 0, 0, (b"8ePsep",)),
    (b"gLeg", b"Leg", True, True, True, False, False, 292.29, 0, 0, (b"Legp", b"Leg5Ac7Ac", b"Legp5Ac7Ac")),
    (b"g4eLeg", b"4eLeg", True, True, True, False, False, 292.29, 0, 0, (b"4eLegp",)),
    (b"gLeg", b"8eLeg", True, True, True, False, False, 292.29, 0, 0, (b"8eLegp",)),
    (b"gAci", b"Aci", True, True, True, False, False, 292.29, 0, 0, (b"Acip",)),
    (b"gAci", b"8eAci", True, True, True, False, False, 292.29, 0, 0, (b"8eAcip",)),
    (b"gBac", b"Bac", False, True, True, False, False, 228.25, 0, 0, (b"Bacp", b"BacNAc", b"BacpNAc")),
    (
        b"Hep",
        b"LDmanHep",
        False,
        False,
        False,
        False,
        False,
        210.18,
        0,
        0,
        (b"LDmanHepp", b"DmanHep", b"DmanHepp", b"manHep", b"manHepp", b"Hep", b"Hepp"),
    ),
    (b"Hep", b"DDmanHep", False, False, False, False, False, 210.18, 0, 0, (b"DDmanHepp",)),
    (b"HepA", b"Dha", True, False, False, False, False, 224.17, 0, 0, (b"Dhap",)),
    (b"MurAc", b"MurNAc", False, False, True, True, False, 293.27, 0, 1, (b"MurpNAc",)),
    (b"MurGc", b"MurNGc", False, False, True, True, False, 309.27, 0, 1, (b"MurpNGc",)),
    (b"gMur", b"Mur", False, False, True, False, False, 251.23, 0, 1, (b"Murp",)),
    # Ketoses
    (b"Hex", b"Fru", True, False, False, False, False, 180.156, 0, 1, (b"Fruf", b"Frup")),
    (b"Hex", b"Tag", True, False, False, False, False, 180.156, 0, 1, (b"Tagp", b"Tagf")),
    (b"Hex", b"Sor", True, False, False, False, False, 180.156, 0, 2, (b"Sorp", b"Sorf")),
    (b"Hex", b"Psi", True, False, False, False, False, 180.156, 0, 1, (b"Psip", b"Psif")),
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


class Sugars:
    """High-performance Structure-of-Arrays (SoA) registry for monosaccharides and carbohydrate modifiers."""

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
    c2_ax: np.ndarray[tuple[int], np.dtype[np.bool_]]
    c3_ax: np.ndarray[tuple[int], np.dtype[np.bool_]]
    c4_ax: np.ndarray[tuple[int], np.dtype[np.bool_]]

    name_to_index: dict[bytes, int]
    alias_map: dict[bytes, bytes]
    _aliases: tuple[tuple[bytes, ...], ...]
    _feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    Index: type[IntEnum]

    @classmethod
    def resolve_name(cls, name_or_alias: bytes | bytearray | str) -> bytes:
        """Resolve a name or alias to its canonical bytes name."""
        bname = name_or_alias.encode("ascii") if isinstance(name_or_alias, str) else bytes(name_or_alias)
        if bname in cls.alias_map:
            return cls.alias_map[bname]
        if bname in cls.name_to_index:
            return bname
        bname_lower = bname.lower()
        if bname_lower in cls.alias_map:
            return cls.alias_map[bname_lower]
        bname_cap = bname.capitalize()
        if bname_cap in cls.name_to_index:
            return bname_cap
        if bname_cap in cls.alias_map:
            return cls.alias_map[bname_cap]
        return bname

    @classmethod
    def index(cls, key: int | np.integer | bytes | bytearray | str) -> int:
        """Return integer index for a sugar name, alias, or integer."""
        if isinstance(key, (int, np.integer)):
            return int(key)
        if isinstance(key, (bytes, bytearray, str)):
            bname = cls.resolve_name(key)
            if bname in cls.name_to_index:
                return cls.name_to_index[bname]
        raise KeyError(f"Sugar '{key!r}' not found in Sugars catalog.")

    @classmethod
    def to_feature_matrix(cls) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Return precomputed `(N, 10)` float32 feature matrix for each sugar."""
        return cls._feature_matrix

    def __class_getitem__(cls, key: int | np.integer | bytes | bytearray | str) -> int:
        """Resolve sugar index via subscript notation (`Sugars[key]`)."""
        return cls.index(key)

    def __len__(self) -> int:
        """Return total number of canonical sugars and modifiers."""
        return len(self.names)

    def __iter__(self) -> Iterator[int]:
        """Iterate over sugar indices."""
        yield from range(len(self.names))

    def __getitem__(self, key: int | np.integer | bytes | bytearray | str) -> int:
        """Resolve sugar index via instance subscript notation."""
        return self.index(key)


def _init_sugars() -> None:
    manifest = _CANONICAL_SUGARS
    n_sugars = len(manifest)
    Sugars.names = np.empty(n_sugars, dtype="S16")
    Sugars.classes = np.empty(n_sugars, dtype="S16")
    Sugars.is_ketose = np.empty(n_sugars, dtype=np.bool_)
    Sugars.is_deoxy = np.empty(n_sugars, dtype=np.bool_)
    Sugars.is_amino = np.empty(n_sugars, dtype=np.bool_)
    Sugars.is_n_acetyl = np.empty(n_sugars, dtype=np.bool_)
    Sugars.is_abstract = np.empty(n_sugars, dtype=np.bool_)
    Sugars.mass = np.empty(n_sugars, dtype=np.float32)
    Sugars.axial_mask = np.empty(n_sugars, dtype=np.uint8)
    Sugars.charge = np.empty(n_sugars, dtype=np.int8)
    Sugars.natural_chirality = np.empty(n_sugars, dtype=np.uint8)
    Sugars.donor_precursor = np.empty(n_sugars, dtype="S24")
    Sugars.carrier = np.empty(n_sugars, dtype="S8")

    Sugars.name_to_index = {}
    Sugars.alias_map = {}
    aliases_list: list[tuple[bytes, ...]] = []
    enum_members: dict[str, int] = {}

    for i, item in enumerate(manifest):
        (s_class, name, ket, d, am, nac, abs_f, m, ax, chi, aliases) = item
        Sugars.names[i] = name
        Sugars.classes[i] = s_class
        Sugars.is_ketose[i] = ket
        Sugars.is_deoxy[i] = d
        Sugars.is_amino[i] = am
        Sugars.is_n_acetyl[i] = nac
        Sugars.is_abstract[i] = abs_f
        Sugars.mass[i] = m
        Sugars.axial_mask[i] = ax
        Sugars.natural_chirality[i] = chi

        prec, carr = infer_donor_and_carrier(name, s_class)
        Sugars.donor_precursor[i] = prec
        Sugars.carrier[i] = carr

        ch = 0
        if s_class in (b"HexA", b"HepA", b"HexNA"):
            ch = -1
        elif name in (b"Neu5Ac", b"Neu5Gc", b"Neu", b"Kdn", b"Kdo", b"Pse", b"Leg", b"Aci", b"P", b"Pyr"):
            ch = -1
        elif s_class in (b"HexN", b"dHexN") or name in (b"PEtn", b"EtN", b"N", b"Cho"):
            ch = 1
        Sugars.charge[i] = ch

        Sugars.name_to_index[name] = i
        s_upper = name.decode("ascii").upper()
        if s_upper.isidentifier():
            enum_members[s_upper] = i

        for al in aliases:
            Sugars.alias_map[al] = name

        aliases_list.append(tuple(aliases))

    Sugars._aliases = tuple(aliases_list)
    Sugars.c2_ax = (Sugars.axial_mask & 1) != 0
    Sugars.c3_ax = (Sugars.axial_mask & 2) != 0
    Sugars.c4_ax = (Sugars.axial_mask & 4) != 0

    features = np.empty((n_sugars, 10), dtype=np.float32)
    features[:, 0] = Sugars.is_ketose
    features[:, 1] = Sugars.is_deoxy
    features[:, 2] = Sugars.is_amino
    features[:, 3] = Sugars.is_n_acetyl
    features[:, 4] = Sugars.is_abstract
    features[:, 5] = Sugars.c2_ax
    features[:, 6] = Sugars.c3_ax
    features[:, 7] = Sugars.c4_ax
    features[:, 8] = Sugars.charge
    features[:, 9] = Sugars.mass / np.float32(350.0)
    features.setflags(write=False)
    Sugars._feature_matrix = features
    Sugars.Index = IntEnum("SugarsIndex", enum_members)


_init_sugars()
