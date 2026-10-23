"""Unit tests for `kbank.io` parsers/normalizers and `kbank.db.curation` consolidation & locus pairing."""

from __future__ import annotations

from pathlib import Path

import pytest
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import LocusArchetype
from serogen.loci.models import GeneNode, Loci, Locus

from kbank.db.curation import (
    antigen_family,
    build_curated_glycan_records,
    normalize_raw_record_taxonomy,
)
from kbank.db.models import GlycanRecord
from kbank.io.base import GlycanParseError
from kbank.io.csdb import CSDBParser
from kbank.io.iupac import IUPACParser
from kbank.io.kegg import KEGGParser, parse_kegg_glycan_flatfile
from kbank.io.sources import (
    classify_antigen_label,
    normalize_ecodab_structure,
    normalize_ek3d_iupac_to_csdb,
    normalize_epsdb_backbone,
    parse_glycan_safe,
    parse_organism_and_strain,
)
from kbank.sugars import Sugars


def test_kegg_parser_connected_vs_composition_filtering() -> None:
    """Verify KEGGParser parses connected NODE/EDGE graphs and rejects edgeless COMPOSITION entries by default."""
    kcf_connected = """ENTRY       G00123                      Glycan
NAME        O-antigen repeat
NODE        3
            1   D-GalNAc    0.0     0.0
            2   D-Gal       -10.0   0.0
            3   L-Rha       -20.0   0.0
EDGE        2
            1   2:b1        1:3
            2   3:a1        2:4
///
"""
    gly = parse_kegg_glycan_flatfile(kcf_connected, "G00123", allow_composition=False)
    assert gly is not None
    assert len(gly.nodes) == 3
    assert len(gly.edges) == 2
    names = [Sugars.names[n.sugar] for n in gly.nodes]
    assert names == [b"GalNAc", b"Gal", b"Rha"]

    kcf_composition_only = """ENTRY       G09999                      Glycan
COMPOSITION (Glc)3 (Gal)2 (Rha)1
///
"""
    assert parse_kegg_glycan_flatfile(kcf_composition_only, "G09999", allow_composition=False) is None
    with pytest.raises(GlycanParseError):
        KEGGParser().parse(kcf_composition_only, allow_composition=False)

    gly_comp = parse_kegg_glycan_flatfile(kcf_composition_only, "G09999", allow_composition=True)
    assert gly_comp is not None
    assert len(gly_comp.nodes) == 3
    assert len(gly_comp.edges) == 0


def test_csdb_expanded_sugars_and_chirality_prefix_protection() -> None:
    """Verify heptoses, pentoses, dideoxyhexoses, and 3-amino deoxyhexoses do not fall back to ExoticSugar."""
    csdb = CSDBParser()
    exotic_idx = Sugars.name_to_index[b"ExoticSugar"]

    seq = (
        b"-3)aLDmanHepp(1-4)bDDmanHepp(1-3)bXylp(1-4)aDigp(1-3)bDhap(2-4)aKdop(2-3)bQuip3NAc(1-4)aFucp3NAc(1-3)aAscp(1-"
    )
    gly = csdb.parse(seq, strict=True)
    sugar_indices = [n.sugar for n in gly.nodes]
    assert exotic_idx not in sugar_indices

    resolved_names = [Sugars.names[s] for s in sugar_indices]
    assert b"LDmanHep" in resolved_names
    assert b"DDmanHep" in resolved_names
    assert b"Xyl" in resolved_names
    assert b"Dig" in resolved_names
    assert b"Dha" in resolved_names
    assert b"Kdo" in resolved_names
    assert b"QuiNAc" in resolved_names
    assert b"FucNAc" in resolved_names
    assert b"Par" in resolved_names


def test_database_format_normalizers() -> None:
    """Verify ECODAB, EK3D, and EPS-DB structure normalizers produce valid Glycan graphs."""
    csdb = CSDBParser()
    iupac = IUPACParser()

    ecodab_ascii = """
        aDGalp(1->2)
             |
    ->3)bDGlcpNAc(1->4)aDGalp(1->3)bDGalpNAc(1->
    """
    norm_eco = normalize_ecodab_structure(ecodab_ascii)
    gly_eco = parse_glycan_safe(norm_eco, csdb, iupac)
    assert gly_eco is not None
    assert len(gly_eco.nodes) == 4
    assert gly_eco.has_polymer_repeat

    ek3d_raw = "[->3)-\u03b2-D-Galp-(1->4)-\u03b1-D-Glcp-(1->]n"
    norm_ek = normalize_ek3d_iupac_to_csdb(ek3d_raw)
    gly_ek = parse_glycan_safe(norm_ek, csdb, iupac)
    assert gly_ek is not None
    assert len(gly_ek.nodes) == 2

    eps_raw = "{ 4)bDGlcp(1-4)aDGalp(1- }"
    norm_eps = normalize_epsdb_backbone(eps_raw)
    gly_eps = parse_glycan_safe(norm_eps, csdb, iupac)
    assert gly_eps is not None
    assert len(gly_eps.nodes) == 2
    assert gly_eps.has_polymer_repeat


def test_taxonomy_and_serovar_extraction() -> None:
    """Verify Kauffmann-White Salmonella serovars and P. aeruginosa IATS serotypes normalize cleanly."""
    org, strain, sero = parse_organism_and_strain(
        "Salmonella typhimurium LT2",
        antigen_type=b"O-antigen",
    )
    assert org == b"Salmonella enterica"
    assert b"Typhimurium" in strain
    assert sero == b"O4"

    org2, strain2, sero2 = parse_organism_and_strain(
        "Pseudomonas aeruginosa IATS O6",
        antigen_type=b"O-antigen",
    )
    assert org2 == b"Pseudomonas aeruginosa"
    assert sero2 == b"O6"

    assert classify_antigen_label("B-band lipopolysaccharide O-antigen") == b"O-antigen"
    assert antigen_family(b"CPS (Group 1)") == b"CPS"
    assert antigen_family(b"LOS") == b"O-antigen"

    raw_list = [
        {"organism": b"Klebsiella pneumoniae", "serotype": b"K2", "strain": b""},
        {"organism": b"Klebsiella sp.", "serotype": b"K2", "strain": b""},
        {"organism": b"Salmonella enterica", "serotype": b"", "strain": b"Enteritidis"},
    ]
    normalize_raw_record_taxonomy(raw_list)
    assert raw_list[1]["organism"] == b"Klebsiella pneumoniae"
    assert raw_list[2]["serotype"] == b"O9"


def test_build_curated_glycan_records_preconsolidates_and_pairs_locus() -> None:
    """Verify build_curated_glycan_records consolidates fragments, grafts modifiers, and pairs with Loci."""
    csdb = CSDBParser()
    # Record 1: Undecorated repeating unit with serotype O10, no accession_hint
    gly_undec = csdb.parse(b"-3)bDGalp(1-4)aDGlcp(1-3)bDGlcpNAc(1-")
    # Record 2: Fragment from same serotype/strain that carries OAc on C6 of Glc AND has accession_hint AB123456.1
    gly_frag = csdb.parse(b"bDGalp(1-4)[Ac(1-6)]aDGlcp")

    raw_records = [
        {
            "source_db": b"CSDB",
            "record_id": b"1001",
            "organism": b"Escherichia coli",
            "strain": b"S10",
            "serotype": b"O10",
            "antigen_type": b"O-antigen",
            "accession_hint": b"",
            "structure_raw": b"-3)bDGalp(1-4)aDGlcp(1-3)bDGlcpNAc(1-",
            "structure_linear": b"-3)bDGalp(1-4)aDGlcp(1-3)bDGlcpNAc(1-",
            "glycan": gly_undec,
            "is_repeating": True,
            "pmid": 12345,
            "publication_year": 2020,
            "doi": b"",
            "notes": b"native repeat",
        },
        {
            "source_db": b"CSDB",
            "record_id": b"1002",
            "organism": b"Escherichia coli",
            "strain": b"S10",
            "serotype": b"O10",
            "antigen_type": b"O-antigen",
            "accession_hint": b"AB123456.1",
            "structure_raw": b"bDGalp(1-4)[Ac(1-6)]aDGlcp",
            "structure_linear": b"bDGalp(1-4)[Ac(1-6)]aDGlcp",
            "glycan": gly_frag,
            "is_repeating": False,
            "pmid": 12345,
            "publication_year": 2020,
            "doi": b"",
            "notes": b"fragment",
        },
    ]

    locus = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE),
            GeneNode(GeneRole.POLYMERASE),
            GeneNode(GeneRole.ACETYLTRANSFERASE),
            GeneNode(GeneRole.PYRUVYLTRANSFERASE),
        ),
        edges=(),
        archetype=LocusArchetype.LPS_WZY,
    ).with_name("AB123456.1")

    assembly_loci_cache = {b"AB123456.1": (b"AB123456.1", Loci.from_list([locus]))}
    db = build_curated_glycan_records(
        raw_records,
        assembly_loci_cache=assembly_loci_cache,
        strain_to_assembly={},
        pmid_to_doi={12345: b"10.1000/test"},
    )

    assert len(db) == 1
    rec = db[0]
    assert rec.record_id == b"1001"
    assert rec.assembly_id == b"AB123456.1"
    assert rec.doi == b"10.1000/test"
    assert rec.locus is not None
    assert rec.glycan is not None
    # Both the fragment-grafted OAc and the locus-completed Pyr should be present!
    mod_names = {Sugars.names[n.sugar] for n in rec.glycan.nodes if n.is_modifier}
    assert mod_names == {b"OAc", b"Pyr"}
    assert b"[grafted_modifiers=+OAc]" in rec.notes
    assert b"[locus_completed=+Pyr]" in rec.notes


def test_glycan_records_and_glycans_pure_soa_and_indexing(tmp_path: object) -> None:
    """Verify GlycanRecords and Glycans are 100% pure SoA (no .records tuple) and support all indexing modes."""
    from pathlib import Path

    import numpy as np

    from kbank.db.records import GlycanRecords

    csdb = CSDBParser()
    g1 = csdb.parse(b"-3)bDGalp(1-4)aDGlcp(1-")
    g2 = csdb.parse(b"-4)bDManp(1-4)aLRhap(1-3)bDGlcpNAc(1-")

    r1 = GlycanRecord(
        source_db=b"ECODAB",
        record_id=b"1",
        organism=b"Escherichia coli",
        serotype=b"O1",
        antigen_type=b"O-antigen",
        structure_raw=b"raw_1",
        structure_linear=b"-3)bDGalp(1-4)aDGlcp(1-",
        glycan=g1,
        doi=b"10.1000/o1",
        notes=b"note_1",
    )
    r2 = GlycanRecord(
        source_db=b"CSDB",
        record_id=b"2",
        organism=b"Klebsiella pneumoniae",
        serotype=b"K2",
        antigen_type=b"CPS",
        structure_raw=b"raw_2",
        structure_linear=b"-4)bDManp(1-4)aLRhap(1-3)bDGlcpNAc(1-",
        glycan=g2,
        doi=b"10.1000/k2",
        notes=b"note_2",
    )

    batch = GlycanRecords.build([r1, r2])
    assert not hasattr(batch, "records")
    assert isinstance(batch.structure_raws, np.ndarray)
    assert isinstance(batch.structure_linears, np.ndarray)
    assert isinstance(batch.dois, np.ndarray)
    assert isinstance(batch.glytoucan_ids, np.ndarray)
    assert isinstance(batch.notes, np.ndarray)
    assert isinstance(batch.glycans.canonicals, np.ndarray)

    # Scalar indexing materializes GlycanRecord on demand
    assert batch[0].compound_key == b"ECODAB:1"
    assert batch[-1].compound_key == b"CSDB:2"
    assert batch[1].notes == b"note_2"
    assert b"ECODAB:1" in batch
    assert r2 in batch

    # Slice, boolean mask, and integer-array indexing on both GlycanRecords and Glycans
    sub_slice = batch[1:]
    assert len(sub_slice) == 1 and sub_slice[0].serotype == b"K2"

    sub_mask = batch[np.array([True, False])]
    assert len(sub_mask) == 1 and sub_mask[0].serotype == b"O1"

    sub_int = batch[np.array([1, 0])]
    assert len(sub_int) == 2
    assert sub_int[0].serotype == b"K2"
    assert sub_int[1].serotype == b"O1"
    assert len(batch.glycans[np.array([1, 0])][0].nodes) == 3

    # Save & load round-trip
    npz_file = Path(str(tmp_path)) / "test_pure_soa.npz"
    batch.save(npz_file)
    reloaded = GlycanRecords.load(npz_file)
    assert not hasattr(reloaded, "records")
    assert len(reloaded) == 2
    assert reloaded[0].structure_raw == b"raw_1"
    assert reloaded[1].doi == b"10.1000/k2"


def test_biological_reducing_end_rotation_and_wzy_polymer_edge() -> None:
    """Verify repeating glycans rotate their reducing end to the biological Und-PP initiator and anchor Wzy."""
    from kbank.db.records import GlycanRecords

    csdb = CSDBParser()
    # Literature repeat unit written with L-Rhap at right-hand reducing end: -3)bDGalp(1-4)aDGlcpA(1-3)aLRhap(1-
    gly_rha_frame = csdb.parse(b"-3)bDGalp(1-4)aDGlcpA(1-3)aLRhap(1-")
    assert Sugars.names[gly_rha_frame.nodes[gly_rha_frame.reducing_end_index].sugar] == b"Rha"

    # Paired CPS_WZY locus with WbaP initiator (cluster 10), 2 GTs, and Wzy polymerase
    locus_cps = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=20),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=21),
            GeneNode(GeneRole.POLYMERASE, cluster_id=30),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )
    rec = GlycanRecord(
        source_db=b"CSDB",
        record_id=b"KP_TEST_1",
        organism=b"Klebsiella pneumoniae",
        serotype=b"K99",
        antigen_type=b"CPS",
        archetype=LocusArchetype.CPS_WZY,
        locus=locus_cps,
        glycan=gly_rha_frame,
        is_repeating=True,
    )

    batch = GlycanRecords.build([rec])
    built = batch[0]
    assert built.glycan is not None
    # Biological reducing end is rotated from *L-Rhap to *D-Galp
    red_idx = built.glycan.reducing_end_index
    assert Sugars.names[built.glycan.nodes[red_idx].sugar] == b"Gal"

    # Wzy (POLYMERASE) is anchored to the unique inter-repeat cycle edge originating at the biological reducing end
    poly_edge_idx, _, _ = built.glycan.polymer_edge_and_assembly_depths(red_idx)
    assert poly_edge_idx is not None
    assert built.glycan.edges[poly_edge_idx].source == red_idx

    wzy_assigns = [a for a in built.assignments if a.role == GeneRole.POLYMERASE]
    assert len(wzy_assigns) == 1
    assert wzy_assigns[0].edge_index == poly_edge_idx
    assert wzy_assigns[0].score >= 0.90

    init_assigns = [a for a in built.assignments if a.role == GeneRole.INITIATOR]
    assert len(init_assigns) == 1
    assert init_assigns[0].bipartite_glycan_index == red_idx


def test_pan_graph_consensus_prior_and_calibrated_confidence() -> None:
    """Verify PanGraphConsensusPrior resolves multi-GT assignments across records and calibrates confidence."""
    from kbank.biosynthesis.align import AssignmentEvidence
    from kbank.db.records import GlycanRecords

    csdb = CSDBParser()
    # Record 1 & 2 share GT cluster 101 which transfers a-L-Rhap-(1->3)-D-Galp in both records:
    # Record 1: disaccharide repeat -4)aLRhap(1-3)bDGalp(1- (1 Wzy + 1 GT cluster 101 -> unambiguous!)
    g1 = csdb.parse(b"-4)aLRhap(1-3)bDGalp(1-")
    loc1 = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=101),
            GeneNode(GeneRole.POLYMERASE, cluster_id=301),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )
    # Record 2: trisaccharide repeat -4)bDManp(1-4)aLRhap(1-3)bDGalp(1- (1 Wzy + GT cluster 101 + GT cluster 102)
    # Because GT cluster 101 is locked to aLRhap(1-3)bDGalp via pan-graph consensus with Record 1,
    # GT cluster 102 is also resolved by stoichiometric elimination to bDManp(1-4)aLRhap!
    g2 = csdb.parse(b"-4)bDManp(1-4)aLRhap(1-3)bDGalp(1-")
    loc2 = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=102),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=101),
            GeneNode(GeneRole.POLYMERASE, cluster_id=302),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )
    # Record 3: tetrasaccharide repeat with 3 singleton GTs (no shared GT cluster_id -> ambiguous 3-way GT ordering)
    g3 = csdb.parse(b"-4)bDGlcpA(1-3)bDManp(1-4)aLRhap(1-3)bDGalp(1-")
    loc3 = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=99),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=901),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=902),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=903),
            GeneNode(GeneRole.POLYMERASE, cluster_id=904),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )

    recs = [
        GlycanRecord(
            source_db=b"CSDB",
            record_id=b"PAN1",
            organism=b"Klebsiella pneumoniae",
            serotype=b"K_PAN1",
            antigen_type=b"CPS",
            archetype=LocusArchetype.CPS_WZY,
            locus=loc1,
            glycan=g1,
            is_repeating=True,
        ),
        GlycanRecord(
            source_db=b"CSDB",
            record_id=b"PAN2",
            organism=b"Klebsiella pneumoniae",
            serotype=b"K_PAN2",
            antigen_type=b"CPS",
            archetype=LocusArchetype.CPS_WZY,
            locus=loc2,
            glycan=g2,
            is_repeating=True,
        ),
        GlycanRecord(
            source_db=b"CSDB",
            record_id=b"PAN3",
            organism=b"Klebsiella pneumoniae",
            serotype=b"K_PAN3",
            antigen_type=b"CPS",
            archetype=LocusArchetype.CPS_WZY,
            locus=loc3,
            glycan=g3,
            is_repeating=True,
        ),
    ]

    batch = GlycanRecords.build(recs)
    b1, b2, b3 = batch[0], batch[1], batch[2]
    assert len(b1.assignments) >= 3

    # In Record 2, gene_index=2 (cluster 101) must be assigned to the a1-3 linkage (aLRhap(1-3)bDGalp)
    # with PAN_GRAPH_CONSENSUS evidence, and gene_index=1 (cluster 102) to the b1-4 linkage!
    gt_b2 = {a.gene_index: a for a in b2.assignments if a.role == GeneRole.GLYCOSYLTRANSFERASE}
    assert gt_b2[2].component_label == "a1-3"
    assert gt_b2[2].evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS
    assert gt_b2[2].is_pan_consensus
    assert gt_b2[1].component_label == "b1-4"
    assert gt_b2[1].evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS
    assert b2.num_pan_consensus_assignments >= 3  # Init + 2 GTs

    # Calibrated confidence: pan-graph-supported GTs in b2 score much higher than singleton 3-GT ties in b3
    mean_gt_score_b2 = sum(a.score for a in gt_b2.values()) / len(gt_b2)
    gt_b3 = [a for a in b3.assignments if a.role == GeneRole.GLYCOSYLTRANSFERASE]
    mean_gt_score_b3 = sum(a.score for a in gt_b3) / len(gt_b3)
    assert mean_gt_score_b2 >= 0.85
    assert mean_gt_score_b3 < 0.70


def test_cross_archetype_pan_graph_cassette_rescue_and_processive_gt() -> None:
    """Verify cassette role rescue, precursor-operon syntenic coupling, and processive multi-linkage GT assignment."""
    from kbank.biosynthesis.align import AssignmentEvidence, TargetType
    from kbank.db.records import GlycanRecords

    csdb = CSDBParser()
    # 1. Cassette rescue: trisaccharide repeat (3 core linkages: 1 Wzy + 2 GTs needed),
    #    but locus has 1 Wzy, 1 GT, and 1 central-cassette UNKNOWN ORF (gene_index=2)
    g_rescue = csdb.parse(b"-4)bDManp(1-4)aLRhap(1-3)bDGalp(1-")
    _loc_rescue = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=201),
            GeneNode(GeneRole.UNKNOWN, cluster_id=202),
            GeneNode(GeneRole.POLYMERASE, cluster_id=301),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )

    # 2. Processive GT in LPS_ABC: mannan trisaccharide repeat with 3 a1-2/a1-3 Man linkages and 1 processive GT
    g_proc = csdb.parse(b"-3)aDManp(1-2)aDManp(1-2)aDManp(1-")
    loc_proc = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=11),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=501),
            GeneNode(GeneRole.ABC_PERMEASE, cluster_id=601),
            GeneNode(GeneRole.ABC_ATPASE, cluster_id=602),
        ),
        edges=(),
        archetype=LocusArchetype.LPS_ABC,
    )

    # 3. Precursor operon syntenic coupling: OXIDOREDUCTASE cluster 777 appears in Rha-containing records
    g_rha_ref = csdb.parse(b"-4)aLRhap(1-3)bDGalp(1-")
    loc_rha_ref = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.OXIDOREDUCTASE, cluster_id=777),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=801),
            GeneNode(GeneRole.POLYMERASE, cluster_id=302),
        ),
        edges=(),
        archetype=LocusArchetype.LPS_WZY,
    )
    # In g_rescue, adding OXIDOREDUCTASE cluster 777 adjacent to gene_index=1 (GT 201) anchors GT 201 to aLRhap(1-3)
    loc_rescue_with_rml = Locus(
        nodes=(
            GeneNode(GeneRole.OXIDOREDUCTASE, cluster_id=777),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=201),
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.UNKNOWN, cluster_id=202),
            GeneNode(GeneRole.POLYMERASE, cluster_id=301),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )

    batch = GlycanRecords.build(
        [
            GlycanRecord(
                source_db=b"CSDB",
                record_id=b"R_RESCUE",
                organism=b"Klebsiella pneumoniae",
                serotype=b"K24_LIKE",
                antigen_type=b"CPS",
                archetype=LocusArchetype.CPS_WZY,
                locus=loc_rescue_with_rml,
                glycan=g_rescue,
                is_repeating=True,
            ),
            GlycanRecord(
                source_db=b"CSDB",
                record_id=b"R_PROC",
                organism=b"Klebsiella pneumoniae",
                serotype=b"O3_LIKE",
                antigen_type=b"O-antigen",
                archetype=LocusArchetype.LPS_ABC,
                locus=loc_proc,
                glycan=g_proc,
                is_repeating=True,
            ),
            GlycanRecord(
                source_db=b"CSDB",
                record_id=b"R_RHA_REF",
                organism=b"Escherichia coli",
                serotype=b"O_RHA",
                antigen_type=b"O-antigen",
                archetype=LocusArchetype.LPS_WZY,
                locus=loc_rha_ref,
                glycan=g_rha_ref,
                is_repeating=True,
            ),
        ]
    )

    rec_rescue = batch[0]
    assert rec_rescue.locus is not None
    # Gene index 3 (originally UNKNOWN) was rescued to GLYCOSYLTRANSFERASE
    assert rec_rescue.locus.nodes[3].role == GeneRole.GLYCOSYLTRANSFERASE
    link_assigns_rescue = [a for a in rec_rescue.assignments if a.target_type == TargetType.LINKAGE]
    assert len(link_assigns_rescue) == 3
    assert all(a.is_aligned for a in link_assigns_rescue)
    # Precursor cluster 777 (Rha) adjacent to gene_index=1 anchors gene_index=1 to aLRhap(1-3)bDGalp ("a1-3")
    gt_by_gene = {a.gene_index: a for a in link_assigns_rescue if a.role == GeneRole.GLYCOSYLTRANSFERASE}
    assert gt_by_gene[1].component_label == "a1-3"
    assert gt_by_gene[1].evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS
    assert gt_by_gene[3].component_label == "b1-4"
    assert gt_by_gene[3].evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS

    # Processive GT in rec_proc covers all 3 core linkages with gene_index=1 (zero RULE_INFERRED fallbacks)
    rec_proc = batch[1]
    link_assigns_proc = [a for a in rec_proc.assignments if a.target_type == TargetType.LINKAGE]
    assert len(link_assigns_proc) == 3
    assert all(a.is_aligned and a.gene_index == 1 for a in link_assigns_proc)
    assert all(a.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS for a in link_assigns_proc)


def test_topological_bitflags_and_gtrabc_branch_handling(tmp_path: Path) -> None:
    """Verify bit-packed BACKBONE/POLYMER flags, extra-locus GtrABC branch handling, and ClusterClassifier."""
    from kbank.biosynthesis.align import (
        AssignmentEvidence,
        BiosyntheticAssigner,
    )
    from kbank.biosynthesis.classifier import (
        BiosyntheticClassifierSuite,
        ClusterClassifier,
    )

    csdb = CSDBParser()
    # Branched repeat: main-chain trisaccharide [-4)bDManp(1-4)aLRhap(1-3)bDGalp(1-]
    # with a terminal side-chain alpha-D-Glcp(1-6) attached to bDGalp
    g_branched = csdb.parse(b"-4)bDManp(1-4)aLRhap(1-3)[aDGlcp(1-6)]bDGalp(1-")

    assert len(g_branched.backbone_indices()) == 3
    assert len(g_branched.branch_indices()) == 1
    assert len(g_branched.branch_point_indices()) == 1
    assert len(g_branched.terminal_leaf_indices()) == 1
    assert g_branched.polymer_edge_index() is not None

    bb_idx = set(g_branched.backbone_indices())
    br_idx = set(g_branched.branch_indices())
    for i, node in enumerate(g_branched.nodes):
        assert node.is_backbone == (i in bb_idx)
        assert node.is_branch == (i in br_idx)

    poly_k = g_branched.polymer_edge_index()
    for k, edge in enumerate(g_branched.edges):
        assert edge.edge_type.is_polymer == (k == poly_k)
        if edge.source in br_idx:
            assert edge.edge_type.is_branch
            assert not edge.edge_type.is_backbone
        else:
            assert edge.edge_type.is_backbone
            assert not edge.edge_type.is_branch

    # Locus has 1 Init, 2 main-chain GTs (for Man and Rha), and 1 Wzy — saturated by the 3-sugar backbone.
    # The terminal aDGlcp(1-6) branch should be recorded as RULE_INFERRED (extra-locus GtrABC)
    loc_main = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=201),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=202),
            GeneNode(GeneRole.POLYMERASE, cluster_id=301),
        ),
        edges=(),
        archetype=LocusArchetype.LPS_WZY,
    )
    assigner = BiosyntheticAssigner()
    assigns = assigner.assign_genes(loc_main, g_branched)
    branch_assigns = [a for a in assigns if a.is_branch_linkage]
    backbone_assigns = [a for a in assigns if a.is_backbone_linkage]
    poly_assigns = [a for a in assigns if a.is_polymer_linkage]

    assert len(branch_assigns) == 1
    assert len(backbone_assigns) == 3
    assert len(poly_assigns) == 1
    assert branch_assigns[0].evidence == AssignmentEvidence.RULE_INFERRED
    assert branch_assigns[0].gene_index == -1

    # Verify ClusterClassifier training, cluster-probability pooling, and save/load roundtrip
    seqs = [
        b"MKVLIVGAGPAGLAAARELRRAGHEVVVFERDPRPVG" * 5,
        b"MKVLIVGAGPAGLAAARELRRAGHEVVVFERDPRPVG" * 5,
        b"MTRVLVTGAGFIGSHLVDRLMAAGHEVVGLDNLRD" * 5,
        b"MTRVLVTGAGFIGSHLVDRLMAAGHEVVGLDNLRD" * 5,
    ]
    y_mech = [1, 1, 2, 2]
    cids = [101, 101, 202, 202]
    clf = ClusterClassifier(target="mechanism", epochs=12, num_filters=16, embed_dim=8, random_state=7)
    clf.fit(seqs, y_mech)
    assert clf.is_fitted
    assert clf.class_names == ("RETAINING", "INVERTING")
    uniq_c, c_probs = clf.predict_cluster_proba(seqs, cids, pool="log_proba")
    assert list(uniq_c) == [101, 202]
    assert c_probs.shape == (2, 2)

    suite = BiosyntheticClassifierSuite({"mechanism": clf, "donor": clf, "linkage": clf})
    save_p = tmp_path / "test_suite.npz"
    suite.save(save_p)
    loaded_suite = BiosyntheticClassifierSuite.load(save_p)
    assert loaded_suite.is_fitted
    assert loaded_suite["mechanism"].class_names == ("RETAINING", "INVERTING")


def test_glycan_to_csdb_roundtrip_and_glycan_predictor() -> None:
    """Verify `Glycan.to_csdb()` round-trip serialization and 4-stage `GlycanPredictor` assembly."""
    from kbank.biosynthesis.align import (
        AssignmentEvidence,
        GeneGlycanAssignment,
        TargetType,
    )
    from kbank.biosynthesis.predictor import GlycanPredictor
    from kbank.db.records import GlycanRecords
    from kbank.glycan import Glycan

    csdb = CSDBParser()
    for raw_csdb in (
        b"-4)bDManp(1-4)aLRhap(1-3)bDGalp(1-",
        b"-4)bDManp(1-4)aLRhap(1-3)[aDGlcp(1-6)]bDGalp(1-",
        b"-3)[OAc(1-2)]bDGalf(1-3)aDGalp(1-",
    ):
        g_orig = csdb.parse(raw_csdb)
        serialized = g_orig.to_csdb()
        g_reparsed = csdb.parse(serialized)
        assert g_orig.is_isomorphic(g_reparsed, ignore_modifiers=False)

    g_from_topo = Glycan.from_topology(g_orig.nodes, g_orig.edges)
    assert len(g_from_topo.canonical) > 0
    assert g_orig.is_isomorphic(csdb.parse(g_from_topo.canonical), ignore_modifiers=False)

    # Build a synthetic training corpus of loci + glycans and fit GlycanPredictor
    g_train_1 = csdb.parse(b"-4)bDGlcp(1-4)aLRhap(1-3)bDGalp(1-")
    g_train_2 = csdb.parse(b"-4)bDGlcp(1-4)[aDGlcp(1-6)]aLRhap(1-3)bDGalp(1-")

    loc_1 = Locus(
        nodes=(
            GeneNode(GeneRole.EPIMERASE, cluster_id=701),
            GeneNode(GeneRole.OXIDOREDUCTASE, cluster_id=702),
            GeneNode(GeneRole.NUCLEOTIDYLTRANSFERASE, cluster_id=703),
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=201),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=202),
            GeneNode(GeneRole.POLYMERASE, cluster_id=301),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )
    loc_2 = Locus(
        nodes=(
            GeneNode(GeneRole.EPIMERASE, cluster_id=701),
            GeneNode(GeneRole.OXIDOREDUCTASE, cluster_id=702),
            GeneNode(GeneRole.NUCLEOTIDYLTRANSFERASE, cluster_id=703),
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=201),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=202),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=203),
            GeneNode(GeneRole.POLYMERASE, cluster_id=301),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )

    records = GlycanRecords.build(
        [
            GlycanRecord(
                source_db=b"CSDB",
                record_id=b"P1",
                organism=b"Klebsiella pneumoniae",
                serotype=b"KP1",
                antigen_type=b"CPS",
                archetype=LocusArchetype.CPS_WZY,
                locus=loc_1,
                glycan=g_train_1,
                is_repeating=True,
            ),
            GlycanRecord(
                source_db=b"CSDB",
                record_id=b"P2",
                organism=b"Klebsiella pneumoniae",
                serotype=b"KP1_DUP",
                antigen_type=b"CPS",
                archetype=LocusArchetype.CPS_WZY,
                locus=loc_1,
                glycan=g_train_1,
                is_repeating=True,
            ),
            GlycanRecord(
                source_db=b"CSDB",
                record_id=b"P3",
                organism=b"Klebsiella pneumoniae",
                serotype=b"KP2",
                antigen_type=b"CPS",
                archetype=LocusArchetype.CPS_WZY,
                locus=loc_2,
                glycan=g_train_2,
                is_repeating=True,
            ),
            GlycanRecord(
                source_db=b"CSDB",
                record_id=b"P4",
                organism=b"Klebsiella pneumoniae",
                serotype=b"KP2_DUP",
                antigen_type=b"CPS",
                archetype=LocusArchetype.CPS_WZY,
                locus=loc_2,
                glycan=g_train_2,
                is_repeating=True,
            ),
        ]
    )

    predictor = GlycanPredictor.from_records(records)
    preds = predictor.predict(loc_1, archetype=LocusArchetype.CPS_WZY, top_k=3, allow_fingerprint_lookup=False)
    assert len(preds) >= 1
    assert b"Rha" in preds[0].precursor_pool
    assert preds[0].glycan.is_isomorphic(g_train_1)
    assert len(preds[0].glycan.canonical) > 0

    # Test clamped prediction (`known_assignments`) on loc_2
    rot_g2 = records.glycans[2]
    clamped_init = GeneGlycanAssignment(
        gene_index=3,
        role=GeneRole.INITIATOR,
        bipartite_glycan_index=rot_g2.reducing_end_index,
        target_type=TargetType.REDUCING_END,
        residue=rot_g2.nodes[rot_g2.reducing_end_index],
        linkage=None,
        edge_index=None,
        evidence=AssignmentEvidence.PAN_GRAPH_CONSENSUS,
        score=0.99,
    )
    preds_clamped = predictor.predict(
        loc_2,
        archetype=LocusArchetype.CPS_WZY,
        top_k=3,
        known_assignments=[clamped_init],
        reference_glycan=g_train_2,
        allow_fingerprint_lookup=False,
    )
    assert len(preds_clamped) >= 1
    assert any(p.glycan.is_isomorphic(g_train_2) for p in preds_clamped)


def test_relational_mpnn_acceptor_motif_and_cache_bundle(tmp_path: Path) -> None:
    """Verify SeroGen `RelationalMPNN`, 10th head `acceptor_motif`, bifunctional GT splitting, and warm cache bundle."""
    from serogen.core.graph.gnn import RelationalMPNN

    from kbank.biosynthesis.classifier import (
        BiosyntheticClassifierSuite,
        ClusterClassifier,
        decode_acceptor_motif_label,
        encode_acceptor_motif_label,
        extract_acceptor_motif_for_edge,
        split_bifunctional_gt_domains,
    )
    from kbank.biosynthesis.predictor import GlycanPredictor, GlycanPredictorParams
    from kbank.db.curation import (
        _load_predicted_loci_cache_bundle,
        save_predicted_loci_cache,
    )
    from kbank.db.records import GlycanRecords
    from kbank.glycan import Anomer, Chirality, ResidueType
    from kbank.sugars import Sugars

    # 1. Verify 10th head acceptor_motif encoding & decoding
    assert "acceptor_motif" in BiosyntheticClassifierSuite.SUITE_TARGETS
    assert len(BiosyntheticClassifierSuite.SUITE_TARGETS) == 10

    gal_idx = Sugars.name_to_index[b"Gal"]
    gal_res = ResidueType.from_sugar(gal_idx, Chirality.D, False)
    lbl_root = encode_acceptor_motif_label(gal_res, Anomer.BETA, 1)
    dec_res, dec_anom, dec_pos = decode_acceptor_motif_label(lbl_root)
    assert dec_res.sugar == gal_idx and dec_anom == Anomer.BETA and dec_pos == 1

    csdb = CSDBParser()
    g_br = csdb.parse(b"-4)bDManp(1-4)[aDGlcp(1-6)]aLRhap(1-3)bDGalp(1-")
    motifs = [decode_acceptor_motif_label(extract_acceptor_motif_for_edge(g_br, i)) for i in range(len(g_br.edges))]
    assert any(pos == 1 for _, _, pos in motifs)
    assert any(pos == 2 for _, _, pos in motifs)

    # 2. Verify bifunctional GT sequence domain splitting
    short_seq = b"MKVLIVGAGPAGLAAARELRRAGHEVVVFERDPRPVG" * 8
    assert split_bifunctional_gt_domains(short_seq, min_len=550) is None
    long_seq = b"MKVLIVGAGPAGLAAARELRRAGHEVVVFERDPRPVG" * 18
    doms = split_bifunctional_gt_domains(long_seq, min_len=550, overlap=25)
    assert doms is not None
    assert len(doms[0]) > 300 and len(doms[1]) > 300

    # 3. Verify ClusterClassifier on acceptor_motif
    seqs_am = [
        b"MKVLIVGAGPAGLAAARELRRAGHEVVVFERDPRPVG" * 5,
        b"MKVLIVGAGPAGLAAARELRRAGHEVVVFERDPRPVG" * 5,
        b"MTRVLVTGAGFIGSHLVDRLMAAGHEVVGLDNLRD" * 5,
        b"MTRVLVTGAGFIGSHLVDRLMAAGHEVVGLDNLRD" * 5,
    ]
    lbl_bb = encode_acceptor_motif_label(gal_res, Anomer.ALPHA, 2)
    clf_am = ClusterClassifier(target="acceptor_motif", epochs=10, num_filters=16, embed_dim=8, random_state=11)
    clf_am.fit(seqs_am, [lbl_root, lbl_root, lbl_bb, lbl_bb])
    assert clf_am.is_fitted
    assert all("@root" in nm or "@bb" in nm for nm in clf_am.class_names)

    # 4. Verify GlycanPredictor RelationalMPNN training and .npz round-trip
    loc_a = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR, cluster_id=10),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=201),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=202),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE, cluster_id=203),
            GeneNode(GeneRole.POLYMERASE, cluster_id=301),
        ),
        edges=(),
        archetype=LocusArchetype.CPS_WZY,
    )
    recs = GlycanRecords.build(
        [
            GlycanRecord(
                source_db=b"CSDB",
                record_id=f"MPNN_{k}".encode(),
                organism=b"Klebsiella pneumoniae",
                serotype=b"K_MPNN",
                antigen_type=b"CPS",
                archetype=LocusArchetype.CPS_WZY,
                locus=loc_a,
                glycan=g_br,
                is_repeating=True,
            )
            for k in range(3)
        ]
    )
    pred_mpnn = GlycanPredictor.from_records(recs, train_mpnn=True, mpnn_epochs=5)
    assert isinstance(pred_mpnn.mpnn, RelationalMPNN)

    custom_params = GlycanPredictorParams(w_pan_dl=4.25, w_acceptor_motif=5.10, w_mpnn_edge=2.35)
    pred_mpnn.params = custom_params
    mpnn_npz = tmp_path / "mpnn_test.npz"
    pred_mpnn.save_mpnn(mpnn_npz)
    assert mpnn_npz.exists()

    pred_loaded = GlycanPredictor.from_records(recs, train_mpnn=False)
    assert pred_loaded.load_mpnn(mpnn_npz)
    assert isinstance(pred_loaded.mpnn, RelationalMPNN)
    assert pred_loaded.params == custom_params
    out_preds = pred_loaded.predict(loc_a, archetype=LocusArchetype.CPS_WZY, top_k=2, allow_fingerprint_lookup=False)
    assert len(out_preds) >= 1

    # 5. Verify zero-redownload loci cache bundle persistence (including negative uid:* markers & maps)
    cache_npz = tmp_path / "pred_loci_cache.npz"
    loci_dict = {
        b"GCF_000001": (b"GCF_000001.1", Loci.from_list([loc_a])),
        b"uid:999888777": (b"", Loci.empty()),
        b"pmid:12345678": (b"", Loci.empty()),
    }
    strain_map = {(b"Klebsiella pneumoniae", b"ATCC 13883"): b"GCF_000001"}
    doi_map = {12345678: b"10.1016/j.carres.2020.01.001"}
    save_predicted_loci_cache(
        cache_npz,
        loci_dict,
        strain_to_assembly=strain_map,
        pmid_to_doi=doi_map,
    )
    loaded_loci, loaded_strains, loaded_dois = _load_predicted_loci_cache_bundle(cache_npz)
    assert b"GCF_000001" in loaded_loci and len(loaded_loci[b"GCF_000001"][1]) == 1
    assert b"uid:999888777" in loaded_loci and len(loaded_loci[b"uid:999888777"][1]) == 0
    assert loaded_strains == strain_map
    assert loaded_dois == doi_map


def test_relational_glycan_entries_curation_and_top_down_archetype_resolution(tmp_path: Path) -> None:
    """Verify relational GlycanEntries SoA, KEGG BRACKET repeats, ECODAB multi-branch ASCII, and top-down archetypes."""
    from kbank.db.curation import (
        consolidate_curated_records,
        resolve_top_down_locus_archetype,
    )
    from kbank.db.models import EntryRelation
    from kbank.db.records import GlycanRecords

    csdb = CSDBParser()
    iupac = IUPACParser()

    # 1. Verify KEGG KCF BRACKET polymer repeat closure & ECODAB multi-branch column-aware ASCII alignment (E. coli O27)
    kcf_o27 = """ENTRY       G13258                      Glycan
NODE        8
            1   *           0.0     0.0
            2   GalNAc      -10.0   0.0
            3   Gal         -20.0   0.0
            4   GlcNAc      -30.0   0.0
            5   *           -40.0   0.0
            6   GlcA        -10.0   10.0
            7   Fuc         -20.0   10.0
            8   Gal         -20.0   20.0
EDGE        7
            1   2:a1        1
            2   6:b1        2:3
            3   3:a1        2:4
            4   7:a1        3:4
            5   8:a1        7:3
            6   4:b1        3:3
            7   5           4:3
BRACKET     1   -35.0   -5.0    -35.0   5.0
            1   -5.0    -5.0    -5.0    5.0
            1   -5.0    5.0     n
///
"""
    g_kegg_o27 = parse_kegg_glycan_flatfile(kcf_o27, "G13258")
    assert g_kegg_o27 is not None
    assert g_kegg_o27.has_polymer_repeat
    assert len(g_kegg_o27.nodes) == 6
    assert g_kegg_o27.canonical.startswith(b"-")

    ecodab_o27 = """aDGalp(1->3)aLFucp(1->4)   bDGlcpA(1->3)
                              |               |
        ->3)bDGlcpNAc(1->3)aDGalp(1->4)aDGalpNAc(1->"""
    norm_o27 = normalize_ecodab_structure(ecodab_o27)
    g_eco_o27 = parse_glycan_safe(norm_o27, csdb, iupac)
    assert g_eco_o27 is not None
    assert g_kegg_o27.is_isomorphic(g_eco_o27, allow_open_repeat=True)

    # 2. Verify 3-stage consolidation & relational GlycanEntries across CSDB, ECODAB, and KEGG
    g_shift = csdb.parse(b"-4)[bDGlcpA(1-3)]aDGalpNAc(1-3)bDGlcpNAc(1-3)[aDGalp(1-3)aLFucp(1-4)]aDGalp(1-")
    g_mod = csdb.parse(b"-3)bDGlcpNAc(1-3)[aDGalp(1-3)aLFucp(1-4)]aDGalp(1-4)[OAc(1-6)][bDGlcpA(1-3)]aDGalpNAc(1-")
    g_frag = csdb.parse(b"aDGalp(1-3)aLFucp(1-4)aDGalp")

    recs_in = [
        GlycanRecord(
            source_db=b"ECODAB",
            record_id=b"217",
            organism=b"Escherichia coli",
            strain=b"F492",
            serotype=b"O27",
            antigen_type=b"O-antigen",
            structure_raw=ecodab_o27.encode("utf-8"),
            structure_linear=norm_o27,
            glycan=g_eco_o27,
            is_repeating=True,
            pmid=16466703,
            publication_year=2006,
        ),
        GlycanRecord(
            source_db=b"KEGG",
            record_id=b"G13258",
            organism=b"Escherichia coli",
            strain=b"",
            serotype=b"O27",
            antigen_type=b"O-antigen",
            structure_raw=b"KEGG_KCF",
            structure_linear=g_kegg_o27.canonical,
            glycan=g_kegg_o27,
            is_repeating=True,
            pmid=16466703,
            glytoucan_id=b"G99999ZZ",
            notes=b"O-antigen [xref_csdb=CSDB:23115]",
        ),
        GlycanRecord(
            source_db=b"CSDB",
            record_id=b"23115",
            organism=b"Escherichia coli",
            strain=b"F492",
            serotype=b"O27",
            antigen_type=b"O-antigen",
            structure_raw=b"-4)[bDGlcpA(1-3)]aDGalpNAc(1-3)bDGlcpNAc(1-3)[aDGalp(1-3)aLFucp(1-4)]aDGalp(1-",
            structure_linear=b"-4)[bDGlcpA(1-3)]aDGalpNAc(1-3)bDGlcpNAc(1-3)[aDGalp(1-3)aLFucp(1-4)]aDGalp(1-",
            glycan=g_shift,
            is_repeating=True,
            pmid=16466703,
        ),
        GlycanRecord(
            source_db=b"CSDB",
            record_id=b"23116",
            organism=b"Escherichia coli",
            strain=b"F492",
            serotype=b"O27",
            antigen_type=b"O-antigen",
            structure_raw=b"-3)bDGlcpNAc(1-3)[aDGalp(1-3)aLFucp(1-4)]aDGalp(1-4)[OAc(1-6)][bDGlcpA(1-3)]aDGalpNAc(1-",
            structure_linear=b"-3)bDGlcpNAc(1-3)[aDGalp(1-3)aLFucp(1-4)]aDGalp(1-4)[OAc(1-6)][bDGlcpA(1-3)]aDGalpNAc(1-",
            glycan=g_mod,
            is_repeating=True,
            pmid=16466703,
        ),
        GlycanRecord(
            source_db=b"CSDB",
            record_id=b"23117",
            organism=b"Escherichia coli",
            strain=b"F492",
            serotype=b"O27",
            antigen_type=b"O-antigen",
            structure_raw=b"aDGalp(1-3)aLFucp(1-4)aDGalp",
            structure_linear=b"aDGalp(1-3)aLFucp(1-4)aDGalp",
            glycan=g_frag,
            is_repeating=False,
            pmid=16466703,
        ),
    ]
    consolidated = consolidate_curated_records(recs_in)
    assert len(consolidated) == 1
    canon_rec = consolidated[0]
    assert canon_rec.num_entries == 5
    assert canon_rec.glytoucan_id == b"G99999ZZ"
    assert set(canon_rec.source_databases) == {b"CSDB", b"ECODAB", b"KEGG"}

    rel_by_key = {e.compound_key: e.relation for e in canon_rec.entries}
    assert rel_by_key[canon_rec.compound_key] == EntryRelation.CANONICAL
    assert EntryRelation.SUBGRAPH_FRAGMENT in rel_by_key.values()
    assert any(
        r in (EntryRelation.CYCLIC_SHIFT, EntryRelation.MODIFIER_VARIANT, EntryRelation.ISOMORPH)
        for r in rel_by_key.values()
    )

    # 3. Verify embedded GlycanEntries SoA persistence in a single .npz file
    db = GlycanRecords.build(consolidated)
    assert len(db) == 1
    assert len(db.entries.source_dbs) == 5
    assert b"KEGG:G13258" in db
    assert b"CSDB:23117" in db
    assert len(db.where_entry_key(b"KEGG:G13258")) == 1
    assert len(db.where_source_db(b"KEGG", include_entries=True)) == 1

    npz_out = tmp_path / "consolidated_entries.npz"
    db.save(npz_out)
    reloaded = GlycanRecords.load(npz_out)
    assert len(reloaded) == 1
    assert reloaded[0].num_entries == 5
    assert {e.compound_key for e in reloaded[0].entries} == {
        b"ECODAB:217",
        b"KEGG:G13258",
        b"CSDB:23115",
        b"CSDB:23116",
        b"CSDB:23117",
    }

    # 4. Verify top-down archetype resolution (fixing E. coli O98 ECODAB:66 & rejecting cross-category mismatches)
    from serogen.loci.archetypes import ArchetypeCategory

    loc_o98 = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE),
            GeneNode(GeneRole.POLYMERASE),
            GeneNode(GeneRole.FLIPPASE),
        ),
        edges=(),
        archetype=LocusArchetype.TEICHOIC_ACID,
    ).with_name("DQ180602.1")
    resolved_o98, ok_o98 = resolve_top_down_locus_archetype(
        loc_o98,
        antigen_type=b"O-antigen",
        expected_cat=ArchetypeCategory.O_ANTIGEN,
        expected_arch=LocusArchetype.LPS_WZY,
        default_arch=LocusArchetype.LPS_WZY,
    )
    assert ok_o98
    assert resolved_o98 == LocusArchetype.LPS_WZY

    # Cross-category mismatch: CPS_ABC record vs. LPS_WZY locus must be rejected
    from dataclasses import replace

    loc_lps_wzy = replace(loc_o98, archetype=LocusArchetype.LPS_WZY)
    _, ok_rejected = resolve_top_down_locus_archetype(
        loc_lps_wzy,
        antigen_type=b"CPS (Group 2)",
        expected_cat=ArchetypeCategory.CAPSULE,
        expected_arch=LocusArchetype.CPS_ABC,
        default_arch=LocusArchetype.CPS_ABC,
    )
    assert not ok_rejected


def test_curation_domain_objects_and_tier_hierarchy(tmp_path: Path) -> None:
    """Verify `GlycanFingerprints` & `LociCacheBundle` (`RaggedArrayContainer` + `ScalarView`) and `LocusResolutionTier`."""
    from dataclasses import replace

    import numpy as np
    from serogen.core.collections import ScalarView
    from serogen.core.graph.align import GraphAligner

    from kbank.biosynthesis.align import BiosyntheticAssigner, GeneGlycanAssignment
    from kbank.db.curation import (
        ClusterMember,
        ExplicitAccessionTier,
        GlycanFingerprint,
        GlycanFingerprints,
        LociCacheBundle,
        LociCacheEntry,
        LocusQueryContext,
        LocusResolutionTier,
        NuccoreSummary,
        PubMedElinkTier,
        RecordCluster,
        SerotypeTokenIndex,
        StrainAssemblyTier,
        TaxonCatalogTier,
        _iter_candidate_pools,
    )
    from kbank.db.models import EntryRelation

    assert issubclass(BiosyntheticAssigner, GraphAligner)
    assert ScalarView in GeneGlycanAssignment.__mro__
    assert ScalarView in GlycanRecord.__mro__
    assert ScalarView in GlycanFingerprint.__mro__
    assert ScalarView in LociCacheEntry.__mro__

    csdb = CSDBParser()
    g_rep = csdb.parse(b"-3)bDGalp(1-4)aDGlcp(1-3)bDGlcpNAc(1-")
    g_shift = csdb.parse(b"-4)aDGlcp(1-3)bDGlcpNAc(1-3)bDGalp(1-")
    g_frag = csdb.parse(b"bDGalp(1-4)[OAc(1-6)]aDGlcp")
    g_oligo = csdb.parse(b"bDGalp(1-4)aDGlcp(1-3)bDGlcpNAc(1-3)bDGalp(1-4)aDGlcp(1-3)bDGlcpNAc")
    g_other = csdb.parse(b"-4)aLRhap(1-3)bDManp(1-")

    fps = GlycanFingerprints.from_list([g_rep, g_shift, g_frag, g_oligo, g_other])
    assert len(fps) == 5
    fp_rep, fp_shift, fp_frag, fp_oligo, fp_other = fps[0], fps[1], fps[2], fps[3], fps[4]

    assert fp_rep.can_be_isomorphic(fp_shift)
    assert not fp_rep.can_be_isomorphic(fp_other)
    assert fp_frag.can_be_subgraph_of(fp_rep)
    assert not fp_other.can_be_subgraph_of(fp_rep)
    assert fp_oligo.can_be_oligomer_of(fp_rep)
    assert not fp_rep.can_be_oligomer_of(fp_oligo)

    # Verify RaggedArrayContainer slicing, boolean masking, concat, and batch Numba filter kernels on GlycanFingerprints
    assert len(fps[1:4]) == 3
    assert len(fps[np.array([True, False, True, False, False])]) == 2
    assert len(GlycanFingerprints.concat([fps[:2], fps[2:]])) == 5

    cand_arr = np.array([0, 1, 4], dtype=np.int32)
    iso_pos = fps.filter_isomorphic_candidates(0, cand_arr)
    assert list(cand_arr[iso_pos]) == [0, 1]

    sub_pos, sub_modes = fps.filter_sub_or_oligo_candidates(
        2, False, np.array([0, 4], dtype=np.int32), np.array([True, True])
    )
    assert list(sub_pos) == [0] and int(sub_modes[0]) & 1

    # Verify RecordCluster promotes 1x repeating monomer over 2x oligomer and grafts fragment modifiers
    rec_oligo = GlycanRecord(
        source_db=b"CSDB",
        record_id=b"OLIGO_2X",
        organism=b"Escherichia coli",
        strain=b"S1",
        serotype=b"O99",
        antigen_type=b"O-antigen",
        glycan=g_oligo,
        is_repeating=False,
        pmid=111,
    )
    rec_monomer = GlycanRecord(
        source_db=b"ECODAB",
        record_id=b"MONO_1X",
        organism=b"Escherichia coli",
        strain=b"S1",
        serotype=b"O99",
        antigen_type=b"O-antigen",
        glycan=g_rep,
        is_repeating=True,
        pmid=111,
    )
    rec_frag = GlycanRecord(
        source_db=b"CSDB",
        record_id=b"FRAG_OAC",
        organism=b"Escherichia coli",
        strain=b"S1",
        serotype=b"O99",
        antigen_type=b"O-antigen",
        glycan=g_frag,
        is_repeating=False,
        pmid=111,
    )
    rec_frag = replace(
        rec_frag,
        entries=(rec_frag.to_entry(EntryRelation.CANONICAL, accession_hint=b"AB999999.1"),),
    )

    cluster = RecordCluster(ClusterMember(rec_oligo, fp_oligo, EntryRelation.CANONICAL))
    cluster.add_member(rec_monomer, fp_rep)
    cluster.add_member(rec_frag, fp_frag)
    canon = cluster.to_canonical_record()
    assert canon.compound_key == b"ECODAB:MONO_1X"
    assert canon.glycan is not None and canon.glycan.num_modifiers == 1
    rels = {e.compound_key: e.relation for e in canon.entries}
    assert rels[b"ECODAB:MONO_1X"] == EntryRelation.CANONICAL
    assert rels[b"CSDB:OLIGO_2X"] == EntryRelation.OLIGOMER
    assert rels[b"CSDB:FRAG_OAC"] == EntryRelation.SUBGRAPH_FRAGMENT

    # Verify SerotypeTokenIndex & NuccoreSummary matching
    idx = SerotypeTokenIndex.build(["K2", "O4", "23F", "ATCC 13883"])
    s_kl2 = NuccoreSummary("1", "AB123.1", "Klebsiella pneumoniae KL2 capsular locus", "ATCC 13883", "strain", True)
    s_sal = NuccoreSummary(
        "2", "AF456.1", "Salmonella enterica Typhimurium O-antigen gene cluster", "4", "serogroup", False
    )
    assert idx.match(s_kl2) == {"K2", "ATCC 13883"}
    assert idx.match(s_sal) == {"O4"}

    # Verify LociCacheBundle RaggedArrayContainer + LociCacheEntry ScalarView + LocusResolutionTier hierarchy
    tiers: tuple[LocusResolutionTier, ...] = (
        ExplicitAccessionTier(),
        PubMedElinkTier(untagged_fallback=False),
        StrainAssemblyTier(),
        TaxonCatalogTier(),
        PubMedElinkTier(untagged_fallback=True),
    )
    loc_dummy = Locus(nodes=(GeneNode(GeneRole.INITIATOR),), edges=(), archetype=LocusArchetype.LPS_WZY)
    loci_a = Loci.from_list([loc_dummy.with_name("AB999999.1")])
    loci_b = Loci.from_list([loc_dummy.with_name("GCF_0001.1")])
    loci_c = Loci.from_list([loc_dummy.with_name("PMID_POOL")])

    bundle = LociCacheBundle.from_dicts(
        assembly_loci={
            b"AB999999": (b"AB999999.1", loci_a),
            b"GCF_0001": (b"GCF_0001.1", loci_b),
            b"pmid:111": (b"PMID_POOL", loci_c),
        },
        strain_to_assembly={(b"Escherichia coli", b"S1"): b"GCF_0001"},
        pmid_to_doi={111: b"10.1000/111"},
    )
    assert len(bundle) == 3
    ent0 = bundle[0]
    assert isinstance(ent0, LociCacheEntry) and ent0.cache_key == b"AB999999" and len(ent0.loci) == 1
    assert len(bundle[1:]) == 2
    assert len(LociCacheBundle.concat([bundle[:1], bundle[1:]])) == 3

    bundle_npz = tmp_path / "ragged_bundle.npz"
    bundle.save(bundle_npz)
    reloaded_bundle = LociCacheBundle.load(bundle_npz)
    assert len(reloaded_bundle) == 3
    assert reloaded_bundle._assembly_loci_dict is None  # zero-copy lazy load!

    ctx = LocusQueryContext.from_record(canon, reloaded_bundle)
    assert ctx.acc_candidates == (b"AB999999.1",)
    assert ctx.pmid_candidates == (111,)
    assert ctx.strain_candidates == (b"S1",)
    assert [k for t in tiers for k, _ in t.iter_cache_keys(ctx)] == [
        b"AB999999.1",
        b"AB999999",
        b"pmid:111:O99",
        b"pmid:111:S1",
        b"GCF_0001",
        b"pmid:111",
    ]
    pools, fallback_asm = _iter_candidate_pools(canon, reloaded_bundle)
    assert fallback_asm == b"AB999999.1"
    assert [p[0] for p in pools] == [b"AB999999.1", b"GCF_0001.1", b"PMID_POOL"]
    assert [p[2] for p in pools] == [False, False, True]
