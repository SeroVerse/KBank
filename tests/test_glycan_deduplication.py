"""Tests for CSDB N-acetyl normalization, Glycan topological comparison, and KBank deduplication."""

from pathlib import Path

from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import LocusArchetype
from serogen.loci.models import GeneNode, Locus

from kbank.biosynthesis.align import AssignmentEvidence, BiosyntheticAssigner
from kbank.db.models import GlycanRecord
from kbank.db.records import GlycanRecords
from kbank.io.csdb import CSDBParser
from kbank.io.iupac import IUPACParser
from kbank.sugars import Sugars


def test_csdb_n_acetyl_amino_sugar_normalization() -> None:
    """Verify `[Ac(1-2)]` on 2-amino sugars is normalized into N-acetylated monosaccharides."""
    csdb = CSDBParser()
    iupac = IUPACParser()

    g_csdb = csdb.parse(b"-3)[Ac(1-2)]bDGlcpN(1-4)aDGalp(1-")
    g_iupac = iupac.parse(b"->3)-b-D-GlcpNAc-(1->4)-a-D-Galp-(1->")

    assert len(g_csdb.nodes) == 2
    assert g_csdb.num_modifiers == 0
    assert g_csdb.nodes[1].sugar == Sugars.name_to_index[b"GlcNAc"]
    assert g_csdb.is_isomorphic(g_iupac)


def test_csdb_n_acetyl_with_true_o_acetyl_preserved() -> None:
    """Verify `[Ac(1-3),Ac(1-2)]bDGlcpN` normalizes C2 to GlcNAc while keeping C3 as OAc."""
    csdb = CSDBParser()
    gly = csdb.parse(b"-4)[Ac(1-3),Ac(1-2)]bDGlcpN(1-3)bDGalp(1-")

    assert len(gly.nodes) == 3
    assert gly.num_modifiers == 1
    sugar_names = {Sugars.names[n.sugar] for n in gly.nodes}
    assert sugar_names == {b"GlcNAc", b"Gal", b"OAc"}


def test_cyclic_shift_isomorphism() -> None:
    """Verify cyclic permutations of the same polymer repeating unit are topologically isomorphic."""
    csdb = CSDBParser()
    g1 = csdb.parse(b"-4)aDGlcp(1-3)bDGalp(1-2)aLRhap(1-")
    g2 = csdb.parse(b"-3)bDGalp(1-2)aLRhap(1-4)aDGlcp(1-")
    g3 = csdb.parse(b"-2)aLRhap(1-4)aDGlcp(1-3)bDGalp(1-")
    g_diff = csdb.parse(b"-4)aDGlcp(1-4)bDGalp(1-2)aLRhap(1-")

    assert g1.is_isomorphic(g2)
    assert g2.is_isomorphic(g3)
    assert g1.is_isomorphic(g3)
    assert not g1.is_isomorphic(g_diff)


def test_alditol_and_hydrolysis_fragment_subgraph() -> None:
    """Verify hydrolysis fragments and borohydride-reduced `-ol` alditols match as subgraphs."""
    csdb = CSDBParser()
    # Klebsiella K8 intact repeating unit (CSDB:4845)
    k8_full = csdb.parse(b"-3)[aDGlcp(1-2)bDGalf(1-4)]aLRhap(1-3)bDGlcp(1-4)aDGalpA(1-")
    # Klebsiella K8 alditol fragment (CSDB:965)
    k8_alditol = csdb.parse(b"aDGlcp(1-2)bDGalf(1-4)aLRhap(1-3)DGlcol")
    # Short disaccharide hydrolysis fragment
    k8_disac = csdb.parse(b"aDGlcp(1-2)bDGalf")
    # Unrelated fragment
    unrelated = csdb.parse(b"aLFucp(1-3)bDGalp")

    assert k8_alditol.is_alditol
    assert k8_alditol.is_subgraph_of(k8_full)
    assert k8_disac.is_subgraph_of(k8_full)
    assert not unrelated.is_subgraph_of(k8_full)


def test_undecorated_core_is_subgraph_of_decorated_repeat() -> None:
    """Verify de-O-acetylated / de-pyruvylated core variants match as subgraphs of decorated repeats."""
    csdb = CSDBParser()
    decorated = csdb.parse(b"-3)[Ac(1-6)]bDGlcp(1-4)[Pyr(1-4)]bDGalp(1-4)aLRhap(1-")
    core_only = csdb.parse(b"-3)bDGlcp(1-4)bDGalp(1-4)aLRhap(1-")

    assert core_only.is_subgraph_of(decorated)
    assert core_only.is_isomorphic(decorated, ignore_modifiers=True)
    assert not core_only.is_isomorphic(decorated, ignore_modifiers=False)


def test_dimer_is_oligomer_of_monomer_repeat() -> None:
    """Verify 2x non-repeating dimers are recognized as oligomers of 1x repeating units."""
    csdb = CSDBParser()
    repeat_1x = csdb.parse(b"-4)aDGlcp(1-3)bDGalp(1-")
    dimer_2x = csdb.parse(b"aDGlcp(1-3)bDGalp(1-4)aDGlcp(1-3)bDGalp")
    unrelated_4mer = csdb.parse(b"aDGlcp(1-3)bDGalp(1-4)aLRhap(1-3)bDGlcp")

    assert dimer_2x.is_oligomer_of(repeat_1x)
    assert not unrelated_4mer.is_oligomer_of(repeat_1x)


def test_fragment_modifier_grafting() -> None:
    """Verify un-decorated repeating units graft known modifier positions from fragments."""
    csdb = CSDBParser()
    core_repeat = csdb.parse(b"-3)bDGlcp(1-4)bDGalp(1-4)aLRhap(1-")
    decorated_fragment = csdb.parse(b"[Ac(1-6)]bDGlcp(1-4)[Pyr(1-3)]bDGalp")
    expected_full = csdb.parse(b"-3)[Ac(1-6)]bDGlcp(1-4)[Pyr(1-3)]bDGalp(1-4)aLRhap(1-")

    grafted_gly, grafted_names = core_repeat.graft_modifiers_from(decorated_fragment)
    assert set(grafted_names) == {b"OAc", b"Pyr"}
    assert grafted_gly.num_modifiers == 2
    assert grafted_gly.is_isomorphic(expected_full)


def test_locus_guided_modifier_completion_and_pathway_accounting() -> None:
    """Verify locus modifier genes complete missing glycan modifiers while subtracting dHexNAc pathway AcTs."""
    csdb = CSDBParser()
    assigner = BiosyntheticAssigner()

    # Repeat unit with 1 QuiNAc (dHexNAc -> consumes 1 locus ACETYLTRANSFERASE), 1 Gal, and 1 GlcNAc (housekeeping)
    gly = csdb.parse(b"-3)[Ac(1-2)]bDQuipN(1-4)bDGalp(1-3)[Ac(1-2)]bDGlcpN(1-")
    assert gly.num_modifiers == 0

    # Locus has 2 ACETYLTRANSFERASEs (1 consumed by QuiNAc pathway -> 1 excess OAc) and 1 PYRUVYLTRANSFERASE (-> 1 Pyr)
    locus = Locus(
        nodes=(
            GeneNode(GeneRole.INITIATOR),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE),
            GeneNode(GeneRole.GLYCOSYLTRANSFERASE),
            GeneNode(GeneRole.POLYMERASE),
            GeneNode(GeneRole.ACETYLTRANSFERASE),
            GeneNode(GeneRole.ACETYLTRANSFERASE),
            GeneNode(GeneRole.PYRUVYLTRANSFERASE),
        ),
        edges=(),
        archetype=LocusArchetype.LPS_WZY,
    )

    rec = GlycanRecord(
        source_db=b"CSDB",
        record_id=b"999999",
        organism=b"Escherichia coli",
        serotype=b"O999",
        antigen_type=b"O-antigen",
        archetype=LocusArchetype.LPS_WZY,
        locus=locus,
        glycan=gly,
        is_repeating=True,
    )
    batch = GlycanRecords.build([rec])
    built = batch[0]

    assert built.glycan is not None
    assert built.glycan.num_modifiers == 2
    mod_names = {Sugars.names[n.sugar] for n in built.glycan.nodes if n.is_modifier}
    assert mod_names == {b"OAc", b"Pyr"}
    assert b"[locus_completed=+Pyr,+OAc]" in built.notes

    completed_assigns = [a for a in built.assignments if a.evidence == AssignmentEvidence.LOCUS_COMPLETED]
    assert len(completed_assigns) == 2
    assert {a.role for a in completed_assigns} == {GeneRole.ACETYLTRANSFERASE, GeneRole.PYRUVYLTRANSFERASE}
    assert all(a.is_locus_completed and a.is_aligned for a in completed_assigns)

    # Calling complete_glycan_modifiers a second time is idempotent
    comp2, names2, _ = assigner.complete_glycan_modifiers(built.glycan, locus)
    assert names2 == ()
    assert len(comp2.nodes) == len(built.glycan.nodes)


def test_curated_kbank_database_invariants(tmp_path: Path) -> None:
    """Verify compiled KBank database has zero UNKNOWN archetypes, zero duplicate typed serotypes, and locus completion."""
    db = GlycanRecords.load()
    assert len(db) > 500

    # 1. No standalone GLYTOUCAN records
    assert b"GLYTOUCAN" not in set(db.source_db.tolist())

    # 2. Zero UNKNOWN LocusArchetype entries
    assert int(LocusArchetype.UNKNOWN) not in set(db.archetypes.tolist())

    # 3. Every typed (organism, serotype, antigen_family) bucket has at most 1 canonical record
    seen_typed: set[tuple[bytes, bytes, bytes]] = set()
    for i in range(len(db)):
        sero = bytes(db.serotypes[i])
        if not sero:
            continue
        org = bytes(db.organisms[i])
        ag = bytes(db.antigen_types[i])
        fam = b"CPS" if ag.startswith(b"CPS") else (b"O-antigen" if ag in (b"O-antigen", b"LPS", b"LOS") else ag)
        key = (org, sero, fam)
        assert key not in seen_typed, f"Duplicate typed record found for {key!r}"
        seen_typed.add(key)

    # 4. Verify locus-completed modifier assignments exist and round-trip cleanly
    assert int(AssignmentEvidence.LOCUS_COMPLETED) in set(db.assignments.evidences.tolist())

    # 5. Round-trip save/load parity
    out_npz = tmp_path / "roundtrip.npz"
    db.save(out_npz)
    reloaded = GlycanRecords.load(out_npz)
    assert len(reloaded) == len(db)
    assert reloaded[0].glycan is not None and db[0].glycan is not None
    assert reloaded[0].glycan.is_isomorphic(db[0].glycan)
