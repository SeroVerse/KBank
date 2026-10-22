# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair>=5.5.0",
#     "kbank",
#     "marimo>=0.25.1",
#     "numpy>=2.4.4",
#     "scipy>=1.17.1",
#     "serogen",
# ]
#
# [tool.uv.sources]
# serogen = { path = "../../SeroGen", editable = true }
# kbank = { path = "..", editable = true }
# ///

"""Interactive KBank Geno-Glycomic Database Explorer powered by SeroGen LayoutEngine & Altair."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    app_title="KBank — Interactive Geno-Glycomic Database Explorer",
)


@app.cell
def _():
    import json
    from abc import ABC, abstractmethod
    from collections import Counter, defaultdict
    from collections.abc import Sequence
    from dataclasses import dataclass, field
    from typing import Any

    import altair as alt
    import marimo as mo
    import numpy as np
    from scipy.sparse import csr_array
    from serogen.core.graph.layout import LayoutEngine
    from serogen.core.graph.models import GraphBatch, GraphModel
    from serogen.genes.roles import GeneRole
    from serogen.loci.archetypes import LocusArchetype
    from serogen.loci.models import GeneNode, IntergenicOrientation, Loci, Locus

    from kbank.biosynthesis.align import (
        AssignmentEvidence,
        BiosyntheticAssigner,
        GeneGlycanAssignment,
        PanGraphConsensusPrior,
        TargetType,
    )
    from kbank.db.models import GlycanRecord
    from kbank.db.records import GlycanRecords
    from kbank.glycan import Anomer, Chirality, Glycan, Glycans, LinkageType, ResidueType
    from kbank.io.csdb import CSDBParser
    from kbank.sugars import Sugars

    def _compact_vegalite_renderer(spec: dict[str, Any]) -> dict[str, str]:
        return {"application/vnd.vegalite.v6+json": json.dumps(spec, separators=(",", ":"))}

    alt.renderers.register("compact_vegalite", _compact_vegalite_renderer)
    _ = alt.renderers.enable("compact_vegalite")
    _ = alt.data_transformers.disable_max_rows()
    return (
        ABC,
        Anomer,
        Any,
        AssignmentEvidence,
        BiosyntheticAssigner,
        CSDBParser,
        Chirality,
        Counter,
        GeneGlycanAssignment,
        GeneNode,
        GeneRole,
        Glycan,
        GlycanRecord,
        GlycanRecords,
        Glycans,
        GraphBatch,
        GraphModel,
        IntergenicOrientation,
        LayoutEngine,
        LinkageType,
        Loci,
        Locus,
        LocusArchetype,
        PanGraphConsensusPrior,
        ResidueType,
        Sequence,
        Sugars,
        TargetType,
        abstractmethod,
        alt,
        csr_array,
        dataclass,
        defaultdict,
        field,
        mo,
        np,
    )


@app.cell
def _(
    ABC,
    Anomer,
    Any,
    AssignmentEvidence,
    Chirality,
    GeneGlycanAssignment,
    GeneNode,
    GeneRole,
    Glycan,
    Glycans,
    GraphBatch,
    GraphModel,
    IntergenicOrientation,
    LayoutEngine,
    LinkageType,
    Literal,
    Loci,
    Locus,
    ResidueType,
    Sequence,
    Sugars,
    abstractmethod,
    alt,
    csr_array,
    dataclass,
    field,
    np,
):
    # ==================================================================================================================
    # 1. Official SNFG (Symbol Nomenclature for Glycans) Visual Encoding & GeneRole Palettes
    # ==================================================================================================================

    # Directional SVG arrow paths in [-1, 1] x [-1, 1] unit box for Altair mark_point(shape=...)
    ARROW_FWD_PATH = "M -0.95,-0.48 L 0.28,-0.48 L 0.28,-0.88 L 0.98,0 L 0.28,0.88 L 0.28,0.48 L -0.95,0.48 Z"
    ARROW_REV_PATH = "M 0.95,-0.48 L -0.28,-0.48 L -0.28,-0.88 L -0.98,0 L -0.28,0.88 L -0.28,0.48 L 0.95,0.48 Z"

    # Custom SNFG 5-pointed star (Pentoses) and 6-sided hexagon (Heptoses) SVG paths
    STAR5_PATH = (
        "M 0,-0.95 L 0.22,-0.30 L 0.90,-0.30 L 0.35,0.12 L 0.56,0.80 "
        "L 0,0.38 L -0.56,0.80 L -0.35,0.12 L -0.90,-0.30 L -0.22,-0.30 Z"
    )
    HEXAGON_PATH = "M 0,-0.92 L 0.80,-0.46 L 0.80,0.46 L 0,0.92 L -0.80,0.46 L -0.80,-0.46 Z"

    GENE_ROLE_COLORS: dict[str, str] = {
        "INITIATOR": "#dc2626",  # Red — Und-PP-sugar initiating transferase
        "GLYCOSYLTRANSFERASE": "#2563eb",  # Royal Blue — Core/branch glycosyltransferases
        "POLYMERASE": "#d97706",  # Amber — Wzy repeating-unit polymerase
        "FLIPPASE": "#059669",  # Emerald — Wzx lipid-linked repeat flippase
        "COPOLYMERASE": "#0d9488",  # Teal — Wzz chain-length regulator
        "TRANSLOCON": "#0891b2",  # Cyan — Wza outer-membrane translocon
        "CAPSULE_ANCHOR": "#0284c7",  # Sky — Wzi / Wzg capsule anchor
        "ABC_PERMEASE": "#0284c7",  # Sky — Wzm ABC permease
        "ABC_ATPASE": "#4f46e5",  # Indigo — Wzt ABC ATPase
        "TRANSMEMBRANE_SYNTHASE": "#7c3aed",  # Violet — HasA / Kps processive synthase
        "LIGASE": "#b45309",  # Dark Amber — WaaL ligase
        "KINASE": "#14b8a6",  # Teal — Wzc tyrosine autokinase
        "PHOSPHATASE": "#2dd4bf",  # Light Teal — Wzb phosphatase
        "EPIMERASE": "#16a34a",  # Green — NDP-sugar epimerase (Gne, RmlC)
        "OXIDOREDUCTASE": "#15803d",  # Forest Green — Dehydrogenase / reductase (Ugd, RmlB, RmlD)
        "AMINOTRANSFERASE": "#65a30d",  # Lime — Sugar aminotransferase
        "NUCLEOTIDYLTRANSFERASE": "#047857",  # Dark Emerald — RmlA / GalU / ManC
        "ALDOLASE": "#22c55e",  # Light Green — NeuB / KdsA synthase
        "NUCLEOTIDASE": "#10b981",  # Emerald
        "ACETYLTRANSFERASE": "#ec4899",  # Pink — O-acetyltransferase (WbbJ, OatA)
        "PYRUVYLTRANSFERASE": "#db2777",  # Rose — Pyruvyltransferase (WcaK)
        "METHYLTRANSFERASE": "#a855f7",  # Purple — O-methyltransferase
        "PHOSPHOTRANSFERASE": "#e11d48",  # Crimson — Phosphotransferase / LicD
        "FORMYLTRANSFERASE": "#9333ea",  # Violet — Formyltransferase
        "ALANINE_TRANSFERASE": "#6366f1",  # Indigo — D-alanyl/aminoacyl transferase
        "DEACETYLASE": "#f43f5e",  # Rose — Deacetylase
        "GLYCOSYLHYDROLASE": "#f59e0b",  # Amber — Glycosylhydrolase / lyase
        "TRANSPOSASE": "#94a3b8",  # Light Slate — IS mobile element / transposase
        "INTEGRASE": "#64748b",  # Slate — Integrase / recombinase
        "UNKNOWN": "#cbd5e1",  # Muted Gray — Unclassified ORF
    }

    # Stereochemical base color map per SNFG standard
    _SNFG_STEREO_COLORS: dict[str, str] = {
        "Glc": "#0072BC",  # SNFG Blue (gluco-)
        "Gal": "#FFD400",  # SNFG Yellow (galacto-)
        "Man": "#00A651",  # SNFG Green (manno-)
        "Fuc": "#ED1C24",  # SNFG Red (fuco-)
        "Rha": "#00A651",  # SNFG Green (rhamno / 6-deoxy-L-manno)
        "Qui": "#0072BC",  # SNFG Blue (quinovo / 6-deoxy-D-gluco)
        "Gul": "#F7941D",  # SNFG Orange (gulo-)
        "Alt": "#F49AC1",  # SNFG Pink (altro-)
        "All": "#A25EA5",  # SNFG Purple (allo-)
        "Tal": "#8FD8F8",  # SNFG Light Blue (talo-)
        "Ido": "#A0724E",  # SNFG Brown (ido-)
        "Xyl": "#F7941D",  # SNFG Orange (xylo-)
        "Ara": "#00A651",  # SNFG Green (arabino-)
        "Rib": "#F49AC1",  # SNFG Pink (ribo-)
        "Lyx": "#FFD400",  # SNFG Yellow (lyxo-)
        "Neu5Ac": "#A54399",  # SNFG Purple (Neu5Ac)
        "Neu5Gc": "#8FD8F8",  # SNFG Light Blue (Neu5Gc)
        "Neu": "#A0724E",  # SNFG Brown (Neu)
        "Kdn": "#00A651",  # SNFG Green (Kdn)
        "Kdo": "#FFD400",  # SNFG Yellow (Kdo)
        "Pse": "#00A651",  # SNFG Green (Pseudaminic acid)
        "Leg": "#FFD400",  # SNFG Yellow (Legionaminic acid)
        "Aci": "#F49AC1",  # SNFG Pink (Acinetaminic acid)
        "8eLeg": "#0072BC",  # SNFG Blue (8-epilegionaminic acid)
        "MurNAc": "#A25EA5",  # SNFG Purple (MurNAc)
        "Mur": "#A0724E",  # SNFG Brown (Mur)
        "Bac": "#0072BC",  # SNFG Blue (Bacillosamine)
        "Hep": "#00A651",  # SNFG Green (Heptose)
    }

    @dataclass(frozen=True, slots=True)
    class SnfgVisualSpec:
        """SNFG visual symbol specification for a monosaccharide or chemical modifier."""

        shape: str
        fill: str
        stroke: str
        class_label: str
        short_label: str

    def get_snfg_spec(res: ResidueType) -> SnfgVisualSpec:
        """Resolve a `ResidueType` into its SNFG shape, color, and label."""
        s_idx = int(res.sugar)
        s_name = Sugars.names[s_idx].decode("ascii", errors="replace")
        s_cls = Sugars.classes[s_idx].decode("ascii", errors="replace")

        # Determine stereochemical root color
        fill_color = "#94a3b8"
        for root_key in (
            "Neu5Ac",
            "Neu5Gc",
            "8eLeg",
            "MurNAc",
            "Glc",
            "Gal",
            "Man",
            "Fuc",
            "Rha",
            "Qui",
            "Gul",
            "Alt",
            "All",
            "Tal",
            "Ido",
            "Xyl",
            "Ara",
            "Rib",
            "Lyx",
            "Kdo",
            "Kdn",
            "Pse",
            "Leg",
            "Aci",
            "Mur",
            "Bac",
            "Hep",
        ):
            if root_key in s_name:
                fill_color = _SNFG_STEREO_COLORS[root_key]
                break

        if s_cls == "Mod":
            if s_name in ("OAc", "Ac", "Fo", "Lac"):
                fill_color = "#f59e0b"
            elif s_name == "Pyr":
                fill_color = "#ec4899"
            elif s_name in ("P", "PEtn", "GroP", "RibitolP"):
                fill_color = "#ef4444"
            elif s_name == "OMe":
                fill_color = "#14b8a6"
            else:
                fill_color = "#6366f1"
            return SnfgVisualSpec(
                shape="diamond",
                fill=fill_color,
                stroke="#0f172a",
                class_label="Modifier (Mod)",
                short_label=s_name,
            )

        # SNFG shape by carbohydrate class
        if s_cls == "Hex":
            shape, cls_lbl = "circle", "Hexose (Circle)"
        elif s_cls == "HexNAc":
            shape, cls_lbl = "square", "HexNAc (Square)"
        elif s_cls == "HexN":
            shape, cls_lbl = "cross", "Hexosamine (Cross)"
        elif s_cls == "HexA":
            shape, cls_lbl = "diamond", "Hexuronate (Diamond)"
        elif s_cls == "dHex":
            shape, cls_lbl = "triangle-up", "Deoxyhexose (Triangle)"
        elif s_cls in ("dHexNAc", "dHexN", "ddHexN"):
            shape, cls_lbl = "triangle-down", "DeoxyhexNAc (Inv. Triangle)"
        elif s_cls == "ddHex":
            shape, cls_lbl = "square", "Dideoxyhexose (Rectangle)"
            if fill_color == "#94a3b8":
                fill_color = "#f7941d"
        elif s_cls == "Non":
            shape, cls_lbl = "diamond", "Nonulosonate / Sialic (Diamond)"
        elif s_cls == "Pen":
            shape, cls_lbl = STAR5_PATH, "Pentose (Star)"
        elif s_cls == "Hep":
            shape, cls_lbl = HEXAGON_PATH, "Heptose (Hexagon)"
        else:
            shape, cls_lbl = HEXAGON_PATH, f"Other ({s_cls})"

        ring_suffix = "f" if res.is_furanose else ""
        short_lbl = f"{s_name}{ring_suffix}"
        return SnfgVisualSpec(
            shape=shape,
            fill=fill_color,
            stroke="#0f172a",
            class_label=cls_lbl,
            short_label=short_lbl,
        )

    def format_linkage_short(lt: LinkageType) -> str:
        """Format a `LinkageType` as a compact glycosidic label (e.g. `β1→4`, `α2→8`)."""
        anom = "α" if lt.anomer == Anomer.ALPHA else ("β" if lt.anomer == Anomer.BETA else "?")
        d_str = str(lt.donor) if lt.donor > 0 else "1"
        a_str = str(lt.acceptor) if lt.acceptor > 0 else "?"
        poly = " [Wzy]" if lt.is_polymer else ""
        return f"{anom}{d_str}→{a_str}{poly}"

    # ==================================================================================================================
    # 2. GraphPlotter(ABC) powered by SeroGen LayoutEngine & Altair
    # ==================================================================================================================

    @dataclass
    class GraphPlotter(ABC):
        """Abstract base class for plotting `GraphModel` and `GraphBatch` objects via `LayoutEngine` and `altair`."""

        layout_engine: LayoutEngine = field(
            default_factory=lambda: LayoutEngine(
                mode="fr",
                dim=2,
                iterations=120,
                gravity=0.15,
                seed=42,
            )
        )
        width: int = 760
        height: int = 280

        def compute_layout(
            self,
            graph: GraphModel | GraphBatch[Any, Any] | csr_array | np.ndarray,
            *,
            pos: np.ndarray | None = None,
            fixed: Sequence[int] | np.ndarray | None = None,
        ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
            """Compute 2D coordinates `(N, 2)` for `graph` using `self.layout_engine` (`LayoutEngine`)."""
            return self.layout_engine.layout(graph, pos=pos, fixed=fixed)

        @abstractmethod
        def extract_elements(
            self,
            graph: Any,
            *,
            coords: np.ndarray | None = None,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            """Extract `(node_rows, edge_rows)` dictionaries for Altair rendering."""

        @abstractmethod
        def plot(self, graph: Any, **kwargs: Any) -> alt.Chart | alt.LayerChart:
            """Render a single `GraphModel` instance as an interactive Altair chart."""

        @abstractmethod
        def plot_pan_graph(self, batch: Any, **kwargs: Any) -> alt.Chart | alt.LayerChart:
            """Render a `GraphBatch` consensus pan-graph in 2D via `LayoutEngine`."""

    # ==================================================================================================================
    # 3. LocusPlotter(GraphPlotter) — Strand-Aware Gene Arrows Coloured by Role or Cluster
    # ==================================================================================================================

    @dataclass
    class LocusPlotter(GraphPlotter):
        """Plot `Locus` gene clusters and `Loci` pan-graphs with strand-aware arrows coloured by role or cluster."""

        color_by: str = "role"
        use_force_layout: bool = False
        height: int = 195

        def _compute_locus_coords(self, locus: Locus) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
            """Compute 2D gene coordinates via `LayoutEngine` with 5'->3' syntenic initialization and pinned ends."""
            n = len(locus.nodes)
            if n == 0:
                return np.empty((0, 2), dtype=np.float64)
            if n == 1:
                return np.zeros((1, 2), dtype=np.float64)

            init_pos = np.zeros((n, 2), dtype=np.float64)
            if len(locus.genes) == n and len(locus.genes.intervals.starts) == n:
                mids = 0.5 * (
                    locus.genes.intervals.starts.astype(np.float64) + locus.genes.intervals.ends.astype(np.float64)
                )
                span = float(np.max(mids) - np.min(mids))
                if span > 1.0:
                    init_pos[:, 0] = 2.0 * (mids - float(np.min(mids))) / span - 1.0
                else:
                    init_pos[:, 0] = np.linspace(-1.0, 1.0, n)
            else:
                init_pos[:, 0] = np.linspace(-1.0, 1.0, n)

            if not self.use_force_layout:
                # Pin all genes on the 5'->3' genomic axis (0 iterations of FR displacement)
                fixed_all = np.ones(n, dtype=np.bool_)
                return self.compute_layout(locus, pos=init_pos, fixed=fixed_all)

            # Warm-start LayoutEngine from 5'->3' axis, pinning 5' and 3' flanking anchor genes
            init_pos[:, 1] = np.where(np.arange(n) % 2 == 0, 0.06, -0.06)
            init_pos[0, 1] = 0.0
            init_pos[-1, 1] = 0.0
            return self.compute_layout(locus, pos=init_pos, fixed=[0, n - 1])

        def extract_elements(
            self,
            locus: Locus,
            *,
            coords: np.ndarray | None = None,
            assignments: Sequence[GeneGlycanAssignment] = (),
            record_label: str = "",
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            """Extract strand-aware gene arrow nodes and intergenic synteny edges for `locus`."""
            n = len(locus.nodes)
            if n == 0:
                return [], []

            pos = coords if coords is not None else self._compute_locus_coords(locus)
            assign_by_gene: dict[int, list[GeneGlycanAssignment]] = {}
            for a in assignments:
                if a.gene_index >= 0:
                    assign_by_gene.setdefault(int(a.gene_index), []).append(a)

            has_genes = len(locus.genes) == n
            if has_genes and len(locus.genes.intervals.strands) == n:
                strands_arr = locus.genes.intervals.strands
            else:
                strands_arr = np.ones(n, dtype=np.int8)
                if int(locus.strand) < 0:
                    strands_arr[0] = -1
                for edge in locus.edges:
                    u_e, v_e = int(edge.source), int(edge.target)
                    if 0 <= u_e < n and 0 <= v_e < n and v_e == u_e + 1:
                        orient = edge.edge_type.orientation
                        if orient == IntergenicOrientation.CODIRECTIONAL:
                            strands_arr[v_e] = strands_arr[u_e]
                        elif orient == IntergenicOrientation.CONVERGENT:
                            strands_arr[v_e] = -1
                        elif orient == IntergenicOrientation.DIVERGENT:
                            strands_arr[v_e] = 1

            node_rows: list[dict[str, Any]] = []
            for idx, g_node in enumerate(locus.nodes):
                role_name = g_node.role.name or "UNKNOWN"
                role_abbrev = g_node.role.abbreviation
                cid = int(g_node.cluster_id)
                strand_val = int(strands_arr[idx])
                if has_genes:
                    start_bp = int(locus.genes.intervals.starts[idx])
                    end_bp = int(locus.genes.intervals.ends[idx])
                    length_bp = max(1, end_bp - start_bp)
                else:
                    start_bp, end_bp, length_bp = idx * 1000, (idx + 1) * 1000, 1000

                g_assigns = assign_by_gene.get(idx, [])
                if g_assigns:
                    rxn_str = ", ".join(a.component_label for a in g_assigns)
                    ev_str = g_assigns[0].evidence.name or "ALIGNED"
                    score_val = float(max(a.score for a in g_assigns))
                    gene_slot = f"G{idx}"
                else:
                    rxn_str = "—"
                    ev_str = "UNASSIGNED"
                    score_val = 0.0
                    gene_slot = f"U{idx}"

                node_rows.append(
                    {
                        "record_label": record_label,
                        "gene_index": idx,
                        "gene_slot": gene_slot,
                        "x": round(float(pos[idx, 0]), 4),
                        "y": round(float(pos[idx, 1]), 4),
                        "role": role_name,
                        "role_abbrev": role_abbrev,
                        "cluster_id": f"CID-{cid:04d}" if cid > 0 else "Unclustered",
                        "strand": "+1 (→)" if strand_val >= 0 else "-1 (←)",
                        "strand_dir": "forward" if strand_val >= 0 else "reverse",
                        "arrow_label": f"{role_abbrev} →" if strand_val >= 0 else f"← {role_abbrev}",
                        "coords_bp": f"{start_bp:,}..{end_bp:,} bp ({length_bp:,} bp)",
                        "length_bp": length_bp,
                        "assigned_reaction": rxn_str,
                        "evidence": ev_str,
                        "alignment_score": round(score_val, 3),
                        "is_assigned": bool(g_assigns),
                    }
                )

            edge_rows: list[dict[str, Any]] = []
            for edge in locus.edges:
                u, v = int(edge.source), int(edge.target)
                if not (0 <= u < n and 0 <= v < n):
                    continue
                orient = edge.edge_type.orientation
                orient_str = {
                    IntergenicOrientation.CODIRECTIONAL: "Codirectional (→ →)",
                    IntergenicOrientation.CONVERGENT: "Convergent (→ ←)",
                    IntergenicOrientation.DIVERGENT: "Divergent (← →)",
                }.get(orient, "Syntenic")
                edge_rows.append(
                    {
                        "record_label": record_label,
                        "x": round(float(pos[u, 0]), 4),
                        "y": round(float(pos[u, 1]), 4),
                        "x2": round(float(pos[v, 0]), 4),
                        "y2": round(float(pos[v, 1]), 4),
                        "orientation": orient_str,
                    }
                )
            return node_rows, edge_rows

        def plot(
            self,
            locus: Locus,
            *,
            assignments: Sequence[GeneGlycanAssignment] = (),
            title: str | None = None,
            gene_selection: Any = None,
        ) -> alt.Chart | alt.LayerChart:
            """Plot a single `Locus` with strand-aware directional arrows coloured by role or cluster."""
            node_rows, edge_rows = self.extract_elements(locus, assignments=assignments)
            if not node_rows:
                return alt.Chart(alt.Data(values=[])).mark_point()

            x_scale = alt.Scale(domain=[-1.15, 1.15])
            y_dom = [-0.45, 0.45] if not self.use_force_layout else [-1.20, 1.20]
            y_scale = alt.Scale(domain=y_dom)

            backbone_rule = (
                alt.Chart(alt.Data(values=edge_rows))
                .mark_rule(color="#64748b", strokeWidth=2.2, opacity=0.45)
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    x2="x2:Q",
                    y2="y2:Q",
                    tooltip=[alt.Tooltip("orientation:N", title="Intergenic Synteny")],
                )
            )

            if self.color_by == "role":
                present_roles = list(dict.fromkeys(str(r["role"]) for r in node_rows))
                present_colors = [GENE_ROLE_COLORS.get(r, "#94a3b8") for r in present_roles]
                color_enc = alt.Color(
                    "role:N",
                    scale=alt.Scale(domain=present_roles, range=present_colors),
                    title="Gene Role",
                )
            else:
                color_enc = alt.Color(
                    "cluster_id:N",
                    scale=alt.Scale(scheme="category20"),
                    title="Homology Cluster",
                )

            opacity_enc = (
                alt.condition(gene_selection, alt.value(1.0), alt.value(0.28))
                if gene_selection is not None
                else alt.value(0.95)
            )

            arrows_layer = (
                alt.Chart(alt.Data(values=node_rows))
                .mark_point(filled=True, stroke="#0f172a", strokeWidth=1.4)
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    shape=alt.Shape(
                        "strand_dir:N",
                        scale=alt.Scale(
                            domain=["forward", "reverse"],
                            range=[ARROW_FWD_PATH, ARROW_REV_PATH],
                        ),
                        legend=None,
                    ),
                    size=alt.Size(
                        "length_bp:Q",
                        scale=alt.Scale(range=[420, 920]),
                        legend=None,
                    ),
                    color=color_enc,
                    opacity=opacity_enc,
                    tooltip=[
                        alt.Tooltip("gene_index:Q", title="ORF Index"),
                        alt.Tooltip("role_abbrev:N", title="Gene Role"),
                        alt.Tooltip("role:N", title="Role Class"),
                        alt.Tooltip("cluster_id:N", title="Homology Cluster"),
                        alt.Tooltip("strand:N", title="Strand"),
                        alt.Tooltip("coords_bp:N", title="Interval"),
                        alt.Tooltip("assigned_reaction:N", title="Assigned Glycan Target"),
                        alt.Tooltip("evidence:N", title="Assignment Evidence"),
                        alt.Tooltip("alignment_score:Q", title="Assignment Score", format=".2f"),
                    ],
                )
            )

            if gene_selection is not None:
                arrows_layer = arrows_layer.add_params(gene_selection)

            labels_layer = (
                alt.Chart(alt.Data(values=node_rows))
                .mark_text(dy=-22, fontSize=10, fontWeight="bold", color="#334155")
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    text="arrow_label:N",
                )
            )

            rxn_labels_layer = (
                alt.Chart(alt.Data(values=[r for r in node_rows if r["is_assigned"]]))
                .mark_text(dy=24, fontSize=9, color="#059669", fontWeight="bold")
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    text="assigned_reaction:N",
                )
            )

            chart_title = title or (
                f"Biosynthetic Gene Cluster ({locus.archetype.name}, {len(node_rows)} ORFs, "
                f"Completeness={locus.completeness:.2f})"
            )
            return (backbone_rule + arrows_layer + labels_layer + rxn_labels_layer).properties(
                width=self.width, height=self.height, title=chart_title
            )

        def plot_pan_graph(
            self,
            loci: Loci,
            *,
            min_node_count: int = 3,
            min_edge_support: int = 2,
            title: str = "Pan-Locus Synteny Network (SeroGen LayoutEngine Fruchterman-Reingold)",
        ) -> alt.Chart | alt.LayerChart:
            """Plot a 2D Pan-Locus homology-cluster synteny network across a `Loci` batch via `LayoutEngine`."""
            pan_adj, pan_nodes, pan_counts = loci.as_pan_graph(node_mask=GeneNode.mask("ROLE_ID", "CLUSTER_ID"))
            pan_csr = pan_adj.tocsr().copy()
            pan_csr.setdiag(0)
            pan_csr.data[pan_csr.data < min_edge_support] = 0
            pan_csr.eliminate_zeros()

            has_edge = (np.diff(pan_csr.indptr) > 0) | (np.diff(pan_csr.tocsc().indptr) > 0)
            keep_mask = (pan_counts >= min_node_count) & (GeneNode.CLUSTER_ID.extract_array(pan_nodes) > 0) & has_edge
            keep_indices = np.nonzero(keep_mask)[0]
            if len(keep_indices) == 0:
                return alt.Chart(alt.Data(values=[])).mark_point()

            sub_csr = pan_csr[keep_indices, :][:, keep_indices].tocsr()
            sub_nodes = pan_nodes[keep_indices]
            sub_counts = pan_counts[keep_indices]

            layout_csr = sub_csr.copy().astype(np.float64)
            if layout_csr.nnz > 0:
                layout_csr.data = 1.0 + 0.30 * np.log1p(layout_csr.data)

            pan_engine = LayoutEngine(
                mode="fr",
                dim=2,
                iterations=200,
                gravity=1.20,
                scale=1.0,
                seed=42,
            )
            coords = pan_engine.layout(layout_csr)

            n_sub = len(sub_nodes)
            max_edge_per_node = np.zeros(n_sub, dtype=np.int32)
            edge_rows: list[dict[str, Any]] = []
            for u in range(n_sub):
                u_val = int(sub_nodes[u])
                u_role = GeneRole(int(GeneNode.ROLE_ID.extract(u_val)))
                u_cid = int(GeneNode.CLUSTER_ID.extract(u_val))
                for k_idx in range(int(sub_csr.indptr[u]), int(sub_csr.indptr[u + 1])):
                    v = int(sub_csr.indices[k_idx])
                    w = int(sub_csr.data[k_idx])
                    if u == v or w < min_edge_support:
                        continue
                    max_edge_per_node[u] = max(max_edge_per_node[u], w)
                    max_edge_per_node[v] = max(max_edge_per_node[v], w)
                    v_val = int(sub_nodes[v])
                    v_role = GeneRole(int(GeneNode.ROLE_ID.extract(v_val)))
                    v_cid = int(GeneNode.CLUSTER_ID.extract(v_val))
                    edge_rows.append(
                        {
                            "x": round(float(coords[u, 0]), 4),
                            "y": round(float(coords[u, 1]), 4),
                            "x2": round(float(coords[v, 0]), 4),
                            "y2": round(float(coords[v, 1]), 4),
                            "weight": w,
                            "transition": f"{u_role.abbreviation} (C{u_cid}) → {v_role.abbreviation} (C{v_cid})",
                        }
                    )

            node_rows: list[dict[str, Any]] = []
            top_indices = set(np.argsort(sub_counts)[-28:].tolist())
            for i in range(n_sub):
                n_val = int(sub_nodes[i])
                role_obj = GeneRole(int(GeneNode.ROLE_ID.extract(n_val)))
                cid = int(GeneNode.CLUSTER_ID.extract(n_val))
                cnt = int(sub_counts[i])
                role_name = role_obj.name or "UNKNOWN"
                node_rows.append(
                    {
                        "x": round(float(coords[i, 0]), 4),
                        "y": round(float(coords[i, 1]), 4),
                        "role": role_name,
                        "role_abbrev": role_obj.abbreviation,
                        "category": role_obj.category.value,
                        "cluster_id": f"CID-{cid:04d}",
                        "locus_count": cnt,
                        "max_edge": max(int(max_edge_per_node[i]), min_edge_support),
                        "label": f"{role_obj.abbreviation} (C{cid})" if i in top_indices else "",
                    }
                )

            max_w = max((int(r["weight"]) for r in edge_rows), default=min_edge_support + 2)
            edge_slider = alt.binding_range(
                min=min_edge_support,
                max=max(min_edge_support + 1, min(25, max_w)),
                step=1,
                name="Min Shared Loci per Synteny Edge: ",
            )
            edge_param = alt.param(name="locus_pan_min_edge", value=min_edge_support, bind=edge_slider)
            cat_sel = alt.selection_point(fields=["category"], bind="legend")

            edges_chart = (
                alt.Chart(alt.Data(values=edge_rows))
                .mark_rule(strokeCap="round", color="#64748b")
                .transform_filter(alt.datum.weight >= edge_param)
                .encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    x2="x2:Q",
                    y2="y2:Q",
                    strokeWidth=alt.StrokeWidth(
                        "weight:Q",
                        scale=alt.Scale(type="sqrt", range=[0.8, 5.0]),
                        title="Shared Loci",
                    ),
                    opacity=alt.Opacity("weight:Q", scale=alt.Scale(range=[0.20, 0.78]), legend=None),
                    tooltip=[
                        alt.Tooltip("transition:N", title="Syntenic Transition"),
                        alt.Tooltip("weight:Q", title="Shared Loci"),
                    ],
                )
            )

            nodes_base = alt.Chart(alt.Data(values=node_rows)).transform_filter(alt.datum.max_edge >= edge_param)

            nodes_chart = (
                nodes_base.mark_point(filled=True, shape=ARROW_FWD_PATH, stroke="#0f172a", strokeWidth=1.0)
                .encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    size=alt.Size("locus_count:Q", scale=alt.Scale(range=[80, 620]), title="Locus Count"),
                    color=alt.Color("category:N", scale=alt.Scale(scheme="tableau10"), title="Functional Category"),
                    opacity=alt.condition(cat_sel, alt.value(0.92), alt.value(0.15)),
                    tooltip=[
                        alt.Tooltip("cluster_id:N", title="Homology Cluster"),
                        alt.Tooltip("role_abbrev:N", title="Gene Role"),
                        alt.Tooltip("role:N", title="Role Class"),
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("locus_count:Q", title="Locus Count"),
                    ],
                )
                .add_params(edge_param, cat_sel)
            )

            text_chart = (
                nodes_base.transform_filter(alt.datum.label != "")
                .mark_text(dy=-12, fontSize=9.5, fontWeight="bold", color="#1e293b")
                .encode(
                    x="x:Q",
                    y="y:Q",
                    text="label:N",
                )
            )

            return (edges_chart + nodes_chart + text_chart).properties(
                width=self.width,
                height=440,
                title=title,
            )

    # ==================================================================================================================
    # 4. GlycanPlotter(GraphPlotter) — SNFG (Symbol Nomenclature for Glycans) Graph & Pan-Graph Plotter
    # ==================================================================================================================

    @dataclass
    class GlycanPlotter(GraphPlotter):
        """Plot `Glycan` repeating units and `Glycans` pan-graphs according to SNFG conventions."""

        relax_branches_with_fr: bool = True
        height: int = 250

        def _compute_snfg_coords(self, glycan: Glycan) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
            """Initialize SNFG right-to-left backbone + branch coordinates and refine free branches via `LayoutEngine`."""
            n = len(glycan.nodes)
            if n == 0:
                return np.empty((0, 2), dtype=np.float64)
            if n == 1:
                return np.zeros((1, 2), dtype=np.float64)

            red_idx = glycan.reducing_end_index
            _, depths, _ = glycan.polymer_edge_and_assembly_depths(red_idx)

            # Separate backbone nodes from side-chain branches and modifiers
            bb_nodes = [i for i, r in enumerate(glycan.nodes) if r.is_backbone and not r.is_modifier]
            if not bb_nodes:
                bb_nodes = [i for i, r in enumerate(glycan.nodes) if not r.is_modifier]
            bb_nodes.sort(key=lambda idx: (depths.get(idx, 99), idx))

            init_pos = np.zeros((n, 2), dtype=np.float64)
            fixed_mask = np.zeros(n, dtype=np.bool_)

            # Place main-chain backbone horizontally from right (+0.85, reducing end) to left (-0.85, non-reducing tip)
            n_bb = len(bb_nodes)
            for rank, u_bb in enumerate(bb_nodes):
                if n_bb == 1:
                    init_pos[u_bb, 0] = 0.0
                else:
                    init_pos[u_bb, 0] = 0.85 - (1.70 * rank / (n_bb - 1))
                init_pos[u_bb, 1] = 0.0
                fixed_mask[u_bb] = True

            # Build parent map for non-polymer edges (donor u -> acceptor w)
            parent_of: dict[int, int] = {}
            children_of: dict[int, list[int]] = {}
            for e in glycan.edges:
                if e.edge_type.is_polymer:
                    continue
                u, w = int(e.source), int(e.target)
                parent_of[u] = w
                children_of.setdefault(w, []).append(u)

            # BFS from backbone nodes to place side-chain monosaccharides (+y) and modifiers (-y)
            placed = set(bb_nodes)
            queue = list(bb_nodes)
            branch_slot_counter: dict[int, int] = {}
            mod_slot_counter: dict[int, int] = {}
            dx_step = 1.70 / max(n_bb, 3)

            while queue:
                p_idx = queue.pop(0)
                px, py = float(init_pos[p_idx, 0]), float(init_pos[p_idx, 1])
                for c_idx in children_of.get(p_idx, []):
                    if c_idx in placed:
                        continue
                    c_res = glycan.nodes[c_idx]
                    if c_res.is_modifier:
                        m_k = mod_slot_counter.get(p_idx, 0)
                        mod_slot_counter[p_idx] = m_k + 1
                        init_pos[c_idx, 0] = px + (0.22 * (m_k - 0.5) if m_k > 0 else 0.0)
                        init_pos[c_idx, 1] = py - 0.52
                    else:
                        b_k = branch_slot_counter.get(p_idx, 0)
                        branch_slot_counter[p_idx] = b_k + 1
                        if p_idx in bb_nodes:
                            # First branch off backbone goes upward (+y) and slightly toward non-reducing left
                            init_pos[c_idx, 0] = px - 0.35 * dx_step + (0.28 * b_k)
                            init_pos[c_idx, 1] = 0.56 + 0.18 * b_k
                        else:
                            # Extension along a multi-sugar side-chain branch continues leftward
                            init_pos[c_idx, 0] = px - 0.78 * dx_step
                            init_pos[c_idx, 1] = py + (0.22 * b_k if b_k > 0 else 0.06)
                    placed.add(c_idx)
                    queue.append(c_idx)

            for u_unplaced in range(n):
                if u_unplaced not in placed:
                    init_pos[u_unplaced, 0] = 0.0
                    init_pos[u_unplaced, 1] = 0.65

            if self.relax_branches_with_fr and not np.all(fixed_mask):
                # Refine free branch/modifier positions via LayoutEngine while keeping backbone pinned
                snfg_engine = LayoutEngine(
                    mode="fr",
                    dim=2,
                    iterations=25,
                    k=0.45,
                    gravity=0.05,
                    scale=None,
                    seed=42,
                )
                coords = snfg_engine.layout(glycan.without_polymer_edge(), pos=init_pos, fixed=fixed_mask)
                # Blend 80% deterministic SNFG tree geometry + 20% FR repulsion to prevent overlaps
                coords = 0.82 * init_pos + 0.18 * coords
                return coords

            return self.compute_layout(glycan, pos=init_pos, fixed=np.ones(n, dtype=np.bool_))

        def extract_elements(
            self,
            glycan: Glycan,
            *,
            coords: np.ndarray | None = None,
            assignments: Sequence[GeneGlycanAssignment] = (),
            record_label: str = "",
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            """Extract SNFG residue nodes and glycosidic linkage edges for `glycan`."""
            n = len(glycan.nodes)
            if n == 0:
                return [], []

            pos = coords if coords is not None else self._compute_snfg_coords(glycan)
            red_idx = glycan.reducing_end_index

            # Map residue/edge indices to assigned gene_index via GeneGlycanAssignments
            node_to_assign: dict[int, GeneGlycanAssignment] = {}
            edge_to_assign: dict[int, GeneGlycanAssignment] = {}
            for a in assignments:
                if a.target_type.name == "REDUCING_END":
                    node_to_assign[red_idx] = a
                elif a.edge_index is not None and 0 <= int(a.edge_index) < len(glycan.edges):
                    e_idx = int(a.edge_index)
                    edge_to_assign[e_idx] = a
                    e_obj = glycan.edges[e_idx]
                    if not e_obj.edge_type.is_polymer:
                        node_to_assign.setdefault(int(e_obj.source), a)

            node_rows: list[dict[str, Any]] = []
            for idx, res in enumerate(glycan.nodes):
                spec = get_snfg_spec(res)
                chi_str = "D-" if res.chirality == Chirality.D else ("L-" if res.chirality == Chirality.L else "")
                ring_str = "furanose (f)" if res.is_furanose else "pyranose (p)"
                topo_role = (
                    "Reducing-End Root"
                    if idx == red_idx
                    else (
                        "Modifier"
                        if res.is_modifier
                        else ("Main-Chain Backbone" if res.is_backbone else "Side-Chain Branch")
                    )
                )
                a_node = node_to_assign.get(idx)
                gene_slot = f"G{a_node.gene_index}" if (a_node is not None and a_node.gene_index >= 0) else f"N{idx}"
                gene_desc = (
                    f"{a_node.role.abbreviation} [ORF #{a_node.gene_index}] ({a_node.evidence.name})"
                    if a_node is not None
                    else "—"
                )

                node_rows.append(
                    {
                        "record_label": record_label,
                        "gene_slot": gene_slot,
                        "x": round(float(pos[idx, 0]), 4),
                        "y": round(float(pos[idx, 1]), 4),
                        "sugar_name": f"{chi_str}{spec.short_label}",
                        "short_label": spec.short_label,
                        "snfg_class": spec.class_label,
                        "snfg_shape": spec.shape,
                        "snfg_fill": spec.fill,
                        "ring_form": ring_str,
                        "topology_role": topo_role,
                        "node_size": 360 if res.is_modifier else (780 if idx == red_idx else 620),
                        "assigned_enzyme": gene_desc,
                    }
                )

            edge_rows: list[dict[str, Any]] = []
            for e_idx, edge in enumerate(glycan.edges):
                u, v = int(edge.source), int(edge.target)
                if not (0 <= u < n and 0 <= v < n):
                    continue
                lt = edge.edge_type
                is_poly = bool(lt.is_polymer)
                ux, uy = float(pos[u, 0]), float(pos[u, 1])
                vx, vy = float(pos[v, 0]), float(pos[v, 1])
                if is_poly:
                    # Polymer repeat closure edge: offset slightly below backbone for clear visual distinction
                    uy_plot, vy_plot = uy - 0.16, vy - 0.16
                else:
                    uy_plot, vy_plot = uy, vy

                mx = 0.5 * (ux + vx)
                my = 0.5 * (uy_plot + vy_plot) + (-0.11 if is_poly else 0.11)
                a_edge = edge_to_assign.get(e_idx)
                gene_slot = f"G{a_edge.gene_index}" if (a_edge is not None and a_edge.gene_index >= 0) else f"E{e_idx}"
                enz_str = (
                    f"{a_edge.role.abbreviation} [ORF #{a_edge.gene_index}] ({a_edge.evidence.name}, score={a_edge.score:.2f})"
                    if a_edge is not None
                    else "—"
                )

                edge_rows.append(
                    {
                        "record_label": record_label,
                        "gene_slot": gene_slot,
                        "x": round(ux, 4),
                        "y": round(uy_plot, 4),
                        "x2": round(vx, 4),
                        "y2": round(vy_plot, 4),
                        "mx": round(mx, 4),
                        "my": round(my, 4),
                        "linkage_label": format_linkage_short(lt),
                        "bond_type": (
                            "Wzy Polymer Repeat Edge"
                            if is_poly
                            else ("Main-Chain Backbone Bond" if lt.is_backbone else "Side-Chain / Modifier Bond")
                        ),
                        "is_polymer": is_poly,
                        "assigned_enzyme": enz_str,
                    }
                )

            return node_rows, edge_rows

        def plot(
            self,
            glycan: Glycan,
            *,
            assignments: Sequence[GeneGlycanAssignment] = (),
            title: str | None = None,
            gene_selection: Any = None,
        ) -> alt.Chart | alt.LayerChart:
            """Plot a single `Glycan` according to SNFG shape and colour conventions."""
            node_rows, edge_rows = self.extract_elements(glycan, assignments=assignments)
            if not node_rows:
                return alt.Chart(alt.Data(values=[])).mark_point()

            x_scale = alt.Scale(domain=[-1.18, 1.18])
            y_scale = alt.Scale(domain=[-0.82, 0.92])

            intra_edges = [e for e in edge_rows if not e["is_polymer"]]
            poly_edges = [e for e in edge_rows if e["is_polymer"]]

            intra_layer = (
                alt.Chart(alt.Data(values=intra_edges))
                .mark_rule(color="#1e293b", strokeWidth=2.6)
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    x2="x2:Q",
                    y2="y2:Q",
                    tooltip=[
                        alt.Tooltip("linkage_label:N", title="Glycosidic Linkage"),
                        alt.Tooltip("bond_type:N", title="Bond Topology"),
                        alt.Tooltip("assigned_enzyme:N", title="Catalyzing Enzyme"),
                    ],
                )
            )

            poly_layer = (
                alt.Chart(alt.Data(values=poly_edges))
                .mark_rule(color="#d97706", strokeWidth=2.4, strokeDash=[6, 4])
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    x2="x2:Q",
                    y2="y2:Q",
                    tooltip=[
                        alt.Tooltip("linkage_label:N", title="Wzy Polymer Repeat Bond"),
                        alt.Tooltip("assigned_enzyme:N", title="Polymerase"),
                    ],
                )
            )

            edge_text_layer = (
                alt.Chart(alt.Data(values=edge_rows))
                .mark_text(fontSize=9.5, fontWeight="bold", color="#334155")
                .encode(
                    x=alt.X("mx:Q", scale=x_scale, axis=None),
                    y=alt.Y("my:Q", scale=y_scale, axis=None),
                    text="linkage_label:N",
                )
            )

            opacity_enc = (
                alt.condition(gene_selection, alt.value(1.0), alt.value(0.28))
                if gene_selection is not None
                else alt.value(0.98)
            )

            nodes_layer = (
                alt.Chart(alt.Data(values=node_rows))
                .mark_point(filled=True, stroke="#0f172a", strokeWidth=1.8)
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    shape=alt.Shape("snfg_shape:N", scale=None, legend=None),
                    fill=alt.Fill("snfg_fill:N", scale=None, legend=None),
                    size=alt.Size("node_size:Q", scale=None, legend=None),
                    opacity=opacity_enc,
                    tooltip=[
                        alt.Tooltip("sugar_name:N", title="Residue / Modifier"),
                        alt.Tooltip("snfg_class:N", title="SNFG Class"),
                        alt.Tooltip("ring_form:N", title="Ring Form"),
                        alt.Tooltip("topology_role:N", title="Topological Role"),
                        alt.Tooltip("assigned_enzyme:N", title="Assigned Enzyme"),
                    ],
                )
            )

            if gene_selection is not None:
                nodes_layer = nodes_layer.add_params(gene_selection)

            node_text_layer = (
                alt.Chart(alt.Data(values=node_rows))
                .mark_text(dy=24, fontSize=10, fontWeight="bold", color="#0f172a")
                .encode(
                    x=alt.X("x:Q", scale=x_scale, axis=None),
                    y=alt.Y("y:Q", scale=y_scale, axis=None),
                    text="short_label:N",
                )
            )

            csdb_str = glycan.to_csdb().decode("ascii", errors="replace")
            chart_title = title or f"SNFG Repeating-Unit Graph: {csdb_str}"
            return (intra_layer + poly_layer + edge_text_layer + nodes_layer + node_text_layer).properties(
                width=self.width, height=self.height, title=chart_title
            )

        def plot_pan_graph(
            self,
            glycans: Glycans,
            *,
            min_edge_support: int = 4,
            title: str = "Pan-Glycan Monosaccharide Transition Network (SNFG + LayoutEngine)",
        ) -> alt.Chart | alt.LayerChart:
            """Plot a 2D Pan-Glycan monosaccharide transition graph across a `Glycans` batch via `LayoutEngine`."""
            chem_mask = int(ResidueType.SUGAR.mask | ResidueType.CHIRALITY.mask | ResidueType.FURANOSE.mask)
            pan_adj, pan_nodes, pan_counts = glycans.as_pan_graph(node_mask=chem_mask)
            pan_csr = pan_adj.tocsr().copy()
            pan_csr.setdiag(0)
            pan_csr.data[pan_csr.data < min_edge_support] = 0
            pan_csr.eliminate_zeros()

            has_edge = (np.diff(pan_csr.indptr) > 0) | (np.diff(pan_csr.tocsc().indptr) > 0)
            keep_mask = (pan_counts >= 3) & has_edge
            keep_indices = np.nonzero(keep_mask)[0]
            if len(keep_indices) == 0:
                return alt.Chart(alt.Data(values=[])).mark_point()

            sub_csr = pan_csr[keep_indices, :][:, keep_indices].tocsr()
            sub_nodes = pan_nodes[keep_indices]
            sub_counts = pan_counts[keep_indices]

            layout_csr = sub_csr.copy().astype(np.float64)
            if layout_csr.nnz > 0:
                layout_csr.data = 1.0 + 0.30 * np.log1p(layout_csr.data)

            pan_engine = LayoutEngine(
                mode="fr",
                dim=2,
                iterations=220,
                gravity=1.10,
                scale=1.0,
                seed=42,
            )
            coords = pan_engine.layout(layout_csr)

            n_sub = len(sub_nodes)
            max_edge_per_node = np.zeros(n_sub, dtype=np.int32)
            edge_rows: list[dict[str, Any]] = []
            for u in range(n_sub):
                u_res = ResidueType(int(sub_nodes[u]))
                u_spec = get_snfg_spec(u_res)
                for k_idx in range(int(sub_csr.indptr[u]), int(sub_csr.indptr[u + 1])):
                    v = int(sub_csr.indices[k_idx])
                    w = int(sub_csr.data[k_idx])
                    if u == v or w < min_edge_support:
                        continue
                    max_edge_per_node[u] = max(max_edge_per_node[u], w)
                    max_edge_per_node[v] = max(max_edge_per_node[v], w)
                    v_res = ResidueType(int(sub_nodes[v]))
                    v_spec = get_snfg_spec(v_res)
                    edge_rows.append(
                        {
                            "x": round(float(coords[u, 0]), 4),
                            "y": round(float(coords[u, 1]), 4),
                            "x2": round(float(coords[v, 0]), 4),
                            "y2": round(float(coords[v, 1]), 4),
                            "weight": w,
                            "disaccharide": f"{u_spec.short_label} → {v_spec.short_label}",
                        }
                    )

            node_rows: list[dict[str, Any]] = []
            for i in range(n_sub):
                res = ResidueType(int(sub_nodes[i]))
                spec = get_snfg_spec(res)
                chi = "D-" if res.chirality == Chirality.D else ("L-" if res.chirality == Chirality.L else "")
                cnt = int(sub_counts[i])
                node_rows.append(
                    {
                        "x": round(float(coords[i, 0]), 4),
                        "y": round(float(coords[i, 1]), 4),
                        "sugar_label": f"{chi}{spec.short_label}",
                        "short_label": spec.short_label,
                        "snfg_class": spec.class_label,
                        "snfg_shape": spec.shape,
                        "snfg_fill": spec.fill,
                        "count": cnt,
                        "max_edge": max(int(max_edge_per_node[i]), min_edge_support),
                    }
                )

            max_w = max((int(r["weight"]) for r in edge_rows), default=min_edge_support + 5)
            edge_slider = alt.binding_range(
                min=min_edge_support,
                max=max(min_edge_support + 1, min(60, max_w)),
                step=1,
                name="Min Disaccharide Bond Frequency: ",
            )
            edge_param = alt.param(name="glycan_pan_min_edge", value=min_edge_support, bind=edge_slider)

            edges_chart = (
                alt.Chart(alt.Data(values=edge_rows))
                .mark_rule(strokeCap="round", color="#64748b")
                .transform_filter(alt.datum.weight >= edge_param)
                .encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    x2="x2:Q",
                    y2="y2:Q",
                    strokeWidth=alt.StrokeWidth(
                        "weight:Q",
                        scale=alt.Scale(type="sqrt", range=[0.8, 5.5]),
                        title="Bond Count",
                    ),
                    opacity=alt.Opacity("weight:Q", scale=alt.Scale(range=[0.22, 0.80]), legend=None),
                    tooltip=[
                        alt.Tooltip("disaccharide:N", title="Disaccharide Transition"),
                        alt.Tooltip("weight:Q", title="Curated Occurrences"),
                    ],
                )
            )

            nodes_base = alt.Chart(alt.Data(values=node_rows)).transform_filter(alt.datum.max_edge >= edge_param)

            nodes_chart = (
                nodes_base.mark_point(filled=True, stroke="#0f172a", strokeWidth=1.5)
                .encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
                    shape=alt.Shape("snfg_shape:N", scale=None, legend=None),
                    fill=alt.Fill("snfg_fill:N", scale=None, legend=None),
                    size=alt.Size("count:Q", scale=alt.Scale(range=[140, 880]), title="Residue Count"),
                    tooltip=[
                        alt.Tooltip("sugar_label:N", title="Residue / Modifier"),
                        alt.Tooltip("snfg_class:N", title="SNFG Class"),
                        alt.Tooltip("count:Q", title="Total Occurrences"),
                    ],
                )
                .add_params(edge_param)
            )

            labels_chart = nodes_base.mark_text(dy=-15, fontSize=10, fontWeight="bold", color="#0f172a").encode(
                x="x:Q",
                y="y:Q",
                text="short_label:N",
            )

            return (edges_chart + nodes_chart + labels_chart).properties(
                width=self.width,
                height=440,
                title=title,
            )

    return (
        ARROW_FWD_PATH,
        ARROW_REV_PATH,
        GENE_ROLE_COLORS,
        GlycanPlotter,
        GraphPlotter,
        HEXAGON_PATH,
        LocusPlotter,
        STAR5_PATH,
        SnfgVisualSpec,
        format_linkage_short,
        get_snfg_spec,
    )


@app.cell
def _(
    ARROW_FWD_PATH,
    ARROW_REV_PATH,
    Any,
    Counter,
    GENE_ROLE_COLORS,
    GlycanPlotter,
    GlycanRecord,
    GlycanRecords,
    LocusPlotter,
    alt,
    np,
):
    # ==================================================================================================================
    # 5. Load Curated KBank Database & Build Interactive Client-Side Vega-Lite Explorer Charts
    # ==================================================================================================================
    db = GlycanRecords.load()
    paired_db = db.where_has_locus()

    locus_plotter = LocusPlotter(width=820, height=195)
    glycan_plotter = GlycanPlotter(width=820, height=255)

    # Include all paired glycan-locus records across KBank
    showcase_records: list[tuple[str, GlycanRecord]] = []
    seen_labels: set[str] = set()

    for idx_p in range(len(paired_db)):
        rec = paired_db[int(idx_p)]
        if rec.glycan is None or rec.locus is None or len(rec.glycan.nodes) == 0 or len(rec.locus.nodes) == 0:
            continue
        org_s = rec.organism.decode("utf-8", errors="replace")
        sero_s = rec.serotype.decode("utf-8", errors="replace") or "Unassigned"
        ag_s = rec.antigen_type.decode("utf-8", errors="replace") or "CPS"
        asm_s = rec.assembly_id.decode("utf-8", errors="replace")
        comp_s = rec.compound_key.decode("utf-8", errors="replace")
        short_org = f"{org_s.split()[0][0]}. {' '.join(org_s.split()[1:])}" if len(org_s.split()) >= 2 else org_s
        lbl = f"{short_org} {sero_s} ({ag_s} · {asm_s})"
        if lbl in seen_labels:
            lbl = f"{short_org} {sero_s} ({ag_s} · {asm_s} · {comp_s})"
        seen_labels.add(lbl)
        showcase_records.append((lbl, rec))

    showcase_records.sort(key=lambda item: item[0])
    showcase_labels = [lbl for lbl, _ in showcase_records]
    record_ids = [str(i) for i in range(len(showcase_records))]

    # Extract pooled node/edge tables across all paired records for pure client-side Vega-Lite switching
    all_locus_nodes: list[dict[str, Any]] = []
    all_locus_edges: list[dict[str, Any]] = []
    all_glycan_nodes: list[dict[str, Any]] = []
    all_glycan_edges: list[dict[str, Any]] = []
    all_meta_rows: list[dict[str, Any]] = []

    for r_idx, (_lbl, rec) in enumerate(showcase_records):
        assert rec.locus is not None and rec.glycan is not None
        rid = str(r_idx)
        lnodes, ledges = locus_plotter.extract_elements(
            rec.locus,
            assignments=rec.assignments,
            record_label=rid,
        )
        gnodes, gedges = glycan_plotter.extract_elements(
            rec.glycan,
            assignments=rec.assignments,
            record_label=rid,
        )
        all_locus_nodes.extend(lnodes)
        all_locus_edges.extend(ledges)
        all_glycan_nodes.extend(gnodes)
        all_glycan_edges.extend(gedges)
        all_meta_rows.append(
            {
                "record_label": rid,
                "header_text": (
                    f"{rec.organism.decode('utf-8', 'replace')} — {rec.serotype.decode('utf-8', 'replace')} "
                    f"[{rec.antigen_type.decode('utf-8', 'replace')} | {rec.archetype.name} | "
                    f"Accession: {rec.assembly_id.decode('utf-8', 'replace')} | "
                    f"Alignment Score: {rec.alignment_score:.2f}]"
                ),
                "csdb_text": f"CSDB Linear: {rec.structure_linear.decode('utf-8', 'replace')}",
            }
        )

    # Interactive client-side record selector + cross-panel gene<->glycan hover/click highlight
    default_id = record_ids[0] if record_ids else "0"
    record_dropdown = alt.binding_select(
        options=record_ids,
        labels=showcase_labels,
        name=f"Select Paired Glycan–Locus Record ({len(showcase_records)} Paired Loci): ",
    )
    record_param = alt.param(
        name="selected_record",
        value=default_id,
        bind=record_dropdown,
    )
    gene_hover = alt.selection_point(
        name="gene_link_hover",
        fields=["gene_slot"],
        on="pointerover",
        empty=True,
    )

    x_loc_scale = alt.Scale(domain=[-1.15, 1.15])
    y_loc_scale = alt.Scale(domain=[-0.48, 0.48])

    present_roles = list(dict.fromkeys(str(r["role"]) for r in all_locus_nodes))
    present_role_colors = [GENE_ROLE_COLORS.get(r, "#94a3b8") for r in present_roles]

    meta_banner = (
        alt.Chart()
        .mark_text(align="left", baseline="middle", fontSize=12.5, fontWeight="bold", color="#0f172a")
        .encode(
            x=alt.value(8),
            y=alt.value(14),
            text="header_text:N",
        )
    )
    csdb_banner = (
        alt.Chart()
        .mark_text(align="left", baseline="middle", fontSize=11, font="JetBrains Mono, monospace", color="#059669")
        .encode(
            x=alt.value(8),
            y=alt.value(34),
            text="csdb_text:N",
        )
    )
    banner_chart = (
        alt.layer(meta_banner, csdb_banner, data=alt.Data(values=all_meta_rows))
        .transform_filter(alt.datum.record_label == record_param)
        .properties(width=820, height=44)
    )

    loc_edges_chart = (
        alt.Chart(alt.Data(values=all_locus_edges))
        .transform_filter(alt.datum.record_label == record_param)
        .mark_rule(color="#64748b", strokeWidth=2.2, opacity=0.45)
        .encode(
            x=alt.X("x:Q", scale=x_loc_scale, axis=None),
            y=alt.Y("y:Q", scale=y_loc_scale, axis=None),
            x2="x2:Q",
            y2="y2:Q",
        )
    )

    loc_nodes_chart = (
        alt.Chart()
        .mark_point(filled=True, stroke="#0f172a", strokeWidth=1.4)
        .encode(
            x=alt.X("x:Q", scale=x_loc_scale, axis=None),
            y=alt.Y("y:Q", scale=y_loc_scale, axis=None),
            shape=alt.Shape(
                "strand_dir:N",
                scale=alt.Scale(domain=["forward", "reverse"], range=[ARROW_FWD_PATH, ARROW_REV_PATH]),
                legend=None,
            ),
            size=alt.Size("length_bp:Q", scale=alt.Scale(range=[420, 920]), legend=None),
            color=alt.Color(
                "role:N",
                scale=alt.Scale(domain=present_roles, range=present_role_colors),
                title="Gene Role (LocusPlotter)",
            ),
            opacity=alt.condition(gene_hover, alt.value(1.0), alt.value(0.25)),
            tooltip=[
                alt.Tooltip("gene_index:Q", title="ORF Index"),
                alt.Tooltip("role_abbrev:N", title="Gene Role"),
                alt.Tooltip("role:N", title="Functional Role"),
                alt.Tooltip("cluster_id:N", title="Homology Cluster"),
                alt.Tooltip("strand:N", title="Strand"),
                alt.Tooltip("coords_bp:N", title="Genomic Interval"),
                alt.Tooltip("assigned_reaction:N", title="Assigned Glycan Target"),
                alt.Tooltip("evidence:N", title="Assignment Evidence"),
                alt.Tooltip("alignment_score:Q", title="Assignment Score", format=".2f"),
            ],
        )
        .add_params(record_param, gene_hover)
    )

    loc_labels_chart = (
        alt.Chart()
        .mark_text(dy=-22, fontSize=10, fontWeight="bold", color="#334155")
        .encode(
            x=alt.X("x:Q", scale=x_loc_scale, axis=None),
            y=alt.Y("y:Q", scale=y_loc_scale, axis=None),
            text="arrow_label:N",
        )
    )

    loc_rxn_chart = (
        alt.Chart()
        .transform_filter(alt.datum.is_assigned)
        .mark_text(dy=24, fontSize=9, fontWeight="bold", color="#059669")
        .encode(
            x=alt.X("x:Q", scale=x_loc_scale, axis=None),
            y=alt.Y("y:Q", scale=y_loc_scale, axis=None),
            text="assigned_reaction:N",
        )
    )

    loc_nodes_layer = alt.layer(
        loc_nodes_chart,
        loc_labels_chart,
        loc_rxn_chart,
        data=alt.Data(values=all_locus_nodes),
    ).transform_filter(alt.datum.record_label == record_param)

    top_locus_panel = (loc_edges_chart + loc_nodes_layer).properties(
        width=820,
        height=185,
        title="1. Biosynthetic Gene Cluster — LocusPlotter (Strand-Aware Arrows Coloured by GeneRole; Hover Gene to Highlight Assigned Sugar/Linkage)",
    )

    # Bottom SNFG Glycan panel linked to the same record_param & gene_hover
    x_gly_scale = alt.Scale(domain=[-1.18, 1.18])
    y_gly_scale = alt.Scale(domain=[-0.82, 0.92])

    gly_intra_chart = (
        alt.Chart()
        .transform_filter(~alt.datum.is_polymer)
        .mark_rule(color="#1e293b", strokeWidth=2.6)
        .encode(
            x=alt.X("x:Q", scale=x_gly_scale, axis=None),
            y=alt.Y("y:Q", scale=y_gly_scale, axis=None),
            x2="x2:Q",
            y2="y2:Q",
            opacity=alt.condition(gene_hover, alt.value(1.0), alt.value(0.30)),
            tooltip=[
                alt.Tooltip("linkage_label:N", title="Glycosidic Linkage"),
                alt.Tooltip("bond_type:N", title="Bond Topology"),
                alt.Tooltip("assigned_enzyme:N", title="Catalyzing Enzyme"),
            ],
        )
    )

    gly_poly_chart = (
        alt.Chart()
        .transform_filter(alt.datum.is_polymer)
        .mark_rule(color="#d97706", strokeWidth=2.4, strokeDash=[6, 4])
        .encode(
            x=alt.X("x:Q", scale=x_gly_scale, axis=None),
            y=alt.Y("y:Q", scale=y_gly_scale, axis=None),
            x2="x2:Q",
            y2="y2:Q",
            opacity=alt.condition(gene_hover, alt.value(1.0), alt.value(0.35)),
            tooltip=[
                alt.Tooltip("linkage_label:N", title="Wzy Polymer Repeat Bond"),
                alt.Tooltip("assigned_enzyme:N", title="Polymerase"),
            ],
        )
    )

    gly_edge_text = (
        alt.Chart()
        .mark_text(fontSize=9.5, fontWeight="bold", color="#334155")
        .encode(
            x=alt.X("mx:Q", scale=x_gly_scale, axis=None),
            y=alt.Y("my:Q", scale=y_gly_scale, axis=None),
            text="linkage_label:N",
        )
    )

    gly_edges_layer = alt.layer(
        gly_intra_chart,
        gly_poly_chart,
        gly_edge_text,
        data=alt.Data(values=all_glycan_edges),
    ).transform_filter(alt.datum.record_label == record_param)

    gly_nodes_chart = (
        alt.Chart()
        .mark_point(filled=True, stroke="#0f172a", strokeWidth=1.8)
        .encode(
            x=alt.X("x:Q", scale=x_gly_scale, axis=None),
            y=alt.Y("y:Q", scale=y_gly_scale, axis=None),
            shape=alt.Shape("snfg_shape:N", scale=None, legend=None),
            fill=alt.Fill("snfg_fill:N", scale=None, legend=None),
            size=alt.Size("node_size:Q", scale=None, legend=None),
            opacity=alt.condition(gene_hover, alt.value(1.0), alt.value(0.25)),
            tooltip=[
                alt.Tooltip("sugar_name:N", title="Residue / Modifier"),
                alt.Tooltip("snfg_class:N", title="SNFG Class"),
                alt.Tooltip("ring_form:N", title="Ring Form"),
                alt.Tooltip("topology_role:N", title="Topological Role"),
                alt.Tooltip("assigned_enzyme:N", title="Assigned Enzyme"),
            ],
        )
    )

    gly_node_text = (
        alt.Chart()
        .mark_text(dy=24, fontSize=10, fontWeight="bold", color="#0f172a")
        .encode(
            x=alt.X("x:Q", scale=x_gly_scale, axis=None),
            y=alt.Y("y:Q", scale=y_gly_scale, axis=None),
            text="short_label:N",
        )
    )

    gly_nodes_layer = alt.layer(
        gly_nodes_chart,
        gly_node_text,
        data=alt.Data(values=all_glycan_nodes),
    ).transform_filter(alt.datum.record_label == record_param)

    bottom_glycan_panel = (gly_edges_layer + gly_nodes_layer).properties(
        width=820,
        height=245,
        title="2. SNFG Polysaccharide Repeating-Unit Graph — GlycanPlotter (Right-to-Left Reducing End; Hover Sugar/Bond to Highlight Gene)",
    )

    paired_interactive_chart = (
        alt.vconcat(banner_chart, top_locus_panel, bottom_glycan_panel, spacing=12)
        .resolve_scale(color="independent", shape="independent", size="independent")
        .configure_view(stroke="#cbd5e1", strokeWidth=1.0, cornerRadius=8)
    )

    # Pan-Graphs via LayoutEngine
    pan_glycan_chart = glycan_plotter.plot_pan_graph(db.glycans, min_edge_support=5)
    pan_locus_chart = locus_plotter.plot_pan_graph(paired_db.loci, min_node_count=4, min_edge_support=3)

    # Build compact flat table rows for curated records
    table_rows: list[dict[str, Any]] = []
    for idx_r in range(len(db)):
        r_obj = db[idx_r]
        n_nodes_g = len(r_obj.glycan.nodes) if r_obj.glycan is not None else 0
        n_edges_g = len(r_obj.glycan.edges) if r_obj.glycan is not None else 0
        n_orfs_l = len(r_obj.locus.nodes) if r_obj.locus is not None else 0
        rxn_summary = (
            ", ".join(a.format_short() for a in r_obj.assignments if a.is_aligned) if r_obj.assignments else "—"
        )
        table_rows.append(
            {
                "Record": r_obj.compound_key.decode("utf-8", errors="replace"),
                "Organism": r_obj.organism.decode("utf-8", errors="replace"),
                "Serotype": r_obj.serotype.decode("utf-8", errors="replace") or "—",
                "Antigen": r_obj.antigen_type.decode("utf-8", errors="replace") or "—",
                "Archetype": r_obj.archetype.name or "UNKNOWN",
                "Assembly": r_obj.assembly_id.decode("utf-8", errors="replace") or "—",
                "ORFs": n_orfs_l,
                "Sugars": n_nodes_g,
                "Bonds": n_edges_g,
                "Score": round(float(r_obj.alignment_score), 3),
                "CSDB Linear": r_obj.structure_linear.decode("utf-8", errors="replace"),
                "Aligned Reactions": rxn_summary,
            }
        )

    return (
        db,
        glycan_plotter,
        locus_plotter,
        paired_db,
        paired_interactive_chart,
        pan_glycan_chart,
        pan_locus_chart,
        showcase_records,
        table_rows,
    )


@app.cell
def _(db, mo, paired_db):
    _total_recs = len(db)
    _paired_recs = len(paired_db)
    _total_assigns = int(db.assignments.total_assignments)
    _unique_orgs = len(set(db.organisms.tolist()))

    mo.md(
        f"## 1. Interactive Paired `LocusPlotter` + `GlycanPlotter` (SNFG) Explorer\n"
        f"**`{_total_recs:,}` Curated Surface Polysaccharides** · **`{_unique_orgs}` Bacterial Species** · "
        f"**`{_paired_recs:,}` Paired `SeroGen` Loci** · **`{_total_assigns:,}` Bipartite `GeneGlycanAssignments`**\n\n"
        f"Use the **Select Paired Glycan–Locus Record ({_paired_recs} Paired Loci)** dropdown below the chart to switch "
        "across all curated capsular and O-antigen loci, and **hover over any gene arrow** to highlight its cross-domain "
        "`GeneGlycanAssignment` in the SNFG repeating-unit graph."
    )
    return


@app.cell
def _(paired_interactive_chart):
    paired_interactive_chart
    return


@app.cell
def _(mo):
    mo.md(
        "## 2. 2D Consensus Pan-Graphs (`Glycans.as_pan_graph()` & `Loci.as_pan_graph()` via `LayoutEngine`)\n"
        "2D Fruchterman–Reingold force-directed layouts of recurring **SNFG monosaccharide/modifier transitions** "
        "and **`GeneNode(ROLE_ID, CLUSTER_ID)` synteny transitions** across KBank. Use the interactive sliders "
        "below each network to filter by minimum bond or synteny edge support."
    )
    return


@app.cell
def _(pan_glycan_chart):
    pan_glycan_chart
    return


@app.cell
def _(pan_locus_chart):
    pan_locus_chart
    return


@app.cell
def _(db, mo, paired_db):
    _total_recs = len(db)
    _paired_recs = len(paired_db)
    _total_assigns = int(db.assignments.total_assignments)

    mo.md(
        f"## 3. All `{_total_recs:,}` Curated KBank Records (`{_paired_recs:,}` Locus-Paired · `{_total_assigns:,}` Assignments)\n"
        "Scroll through all curated records below (sticky headers enabled for all 1,530 entries)."
    )
    return


@app.cell
def _(db, mo, table_rows):
    _total_recs = len(db)
    mo.ui.table(
        table_rows,
        pagination=False,
        max_height=600,
        selection=None,
        label=f"KBank Geno-Glycomic Registry ({_total_recs:,} Records)",
    )
    return


if __name__ == "__main__":
    app.run()
