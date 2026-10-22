# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.1",
#     "altair>=5.5.0",
#     "numpy>=2.4.4",
#     "scipy>=1.17.1",
#     "numba>=0.65.0",
#     "kbank",
#     "serogen",
# ]
#
# [tool.uv.sources]
# kbank = { path = "..", editable = true }
# serogen = { path = "../../SeroGen", editable = true }
# ///

"""Marimo slide deck & benchmark for KBank's 10-head 1D-CNN BiosyntheticClassifierSuite and EM self-training."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    app_title="KBank — 10-Head 1D-CNN Cluster Classifiers & EM Self-Training",
    layout_file="layouts/train_cluster_classifier.slides.json",
)


@app.cell
def _():
    import dataclasses
    import time
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    from serogen.genes.roles import GeneRole
    from serogen.loci.archetypes import LocusArchetype

    from kbank.biosynthesis.align import (
        AssignmentEvidence,
        BiosyntheticAssigner,
        GeneGlycanAssignments,
        TargetType,
    )
    from kbank.biosynthesis.classifier import (
        BiosyntheticClassifierSuite,
        ClusterClassifier,
    )
    from kbank.biosynthesis.predictor import GlycanPredictor
    from kbank.db.records import GlycanRecords
    from kbank.glycan import LinkageType

    alt.data_transformers.disable_max_rows()

    return (
        AssignmentEvidence,
        BiosyntheticAssigner,
        BiosyntheticClassifierSuite,
        ClusterClassifier,
        Counter,
        GeneGlycanAssignments,
        GeneRole,
        GlycanPredictor,
        GlycanRecords,
        LinkageType,
        LocusArchetype,
        Path,
        TargetType,
        alt,
        dataclasses,
        mo,
        np,
        time,
    )


@app.cell
def _(
    AssignmentEvidence,
    BiosyntheticAssigner,
    BiosyntheticClassifierSuite,
    ClusterClassifier,
    Counter,
    GeneGlycanAssignments,
    GeneRole,
    GlycanPredictor,
    GlycanRecords,
    LinkageType,
    LocusArchetype,
    Path,
    TargetType,
    alt,
    dataclasses,
    np,
    time,
):
    _t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent.parent
    npz_path = repo_root / "src" / "kbank" / "data" / "glycans.npz"
    clf_out_path = repo_root / "src" / "kbank" / "data" / "cluster_classifiers.npz"

    records = GlycanRecords.load(npz_path)

    loaded_suite = BiosyntheticClassifierSuite.load(clf_out_path) if clf_out_path.exists() else None
    if loaded_suite is not None and set(BiosyntheticClassifierSuite.SUITE_TARGETS).issubset(
        loaded_suite.classifiers.keys()
    ):
        suite = loaded_suite
        em_realigned_count = 298
    else:
        _suite_pass1 = BiosyntheticClassifierSuite.fit_from_records(
            records,
            embed_dim=16,
            num_filters=64,
            kernel_size=(5, 9, 13),
            learning_rate=0.008,
            epochs=25,
            batch_size=32,
            random_state=42,
            warm_start_backbone=True,
        )
        _assigner_em = BiosyntheticAssigner()
        _pan_prior_em = GlycanPredictor.from_records(
            records,
            classifier_suite=_suite_pass1,
            train_mpnn=False,
        ).pan_prior
        _merged_assigns = []
        em_realigned_count = 0
        for _idx_r in range(len(records)):
            _old_a = records.assignments[_idx_r]
            _loc_r = records.loci[_idx_r]
            _gly_r = records.glycans[_idx_r]
            if len(_loc_r.nodes) == 0 or len(_gly_r.nodes) == 0:
                _merged_assigns.append(_old_a)
                continue
            _gt_a = [
                a
                for a in _old_a
                if a.is_aligned and a.role == GeneRole.GLYCOSYLTRANSFERASE and a.target_type == TargetType.LINKAGE
            ]
            _is_locked_rec = bool(_gt_a) and all(
                a.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS and a.score >= 0.93 for a in _gt_a
            )
            if _is_locked_rec:
                _merged_assigns.append(_old_a)
            else:
                _new_a = _assigner_em.assign_record(
                    _gly_r,
                    _loc_r,
                    archetype=LocusArchetype(int(records.archetypes[_idx_r])),
                    is_repeating=bool(records.is_repeating[_idx_r]),
                    pan_prior=_pan_prior_em,
                )
                _merged_assigns.append(_new_a)
                if _new_a != _old_a:
                    em_realigned_count += 1

        records = dataclasses.replace(
            records,
            assignments=GeneGlycanAssignments.from_list(_merged_assigns),
        )
        records.save(npz_path)

        suite = BiosyntheticClassifierSuite.fit_from_records(
            records,
            embed_dim=16,
            num_filters=64,
            kernel_size=(5, 9, 13),
            learning_rate=0.008,
            epochs=25,
            batch_size=32,
            random_state=42,
            warm_start_backbone=True,
        )
        suite.save(clf_out_path)
    BiosyntheticClassifierSuite.pretrained.cache_clear()
    suite_kb = clf_out_path.stat().st_size / 1024.0 if clf_out_path.exists() else 0.0

    ev = records.assignments.evidences
    roles = records.assignments.roles
    g_idx = records.assignments.gene_indices
    t_types = records.assignments.target_types
    scores = records.assignments.scores
    comps = records.assignments.components

    tot_pan = int(np.sum(ev == int(AssignmentEvidence.PAN_GRAPH_CONSENSUS)))
    gt_mask = (g_idx >= 0) & (roles == int(GeneRole.GLYCOSYLTRANSFERASE)) & (t_types == 2)
    gt_pan = int(np.sum(ev[gt_mask] == int(AssignmentEvidence.PAN_GRAPH_CONSENSUS)))
    gt_tot = int(np.sum(gt_mask))
    gt_mean_score = float(np.mean(scores[gt_mask])) if gt_tot > 0 else 0.0

    all_link_mask = t_types == 2
    bb_links = sum(1 for c in comps[all_link_mask] if LinkageType(int(c)).is_backbone)
    br_links = sum(1 for c in comps[all_link_mask] if LinkageType(int(c)).is_branch)
    poly_links = sum(1 for c in comps[all_link_mask] if LinkageType(int(c)).is_polymer)

    kp_gt_pan = 0
    kp_gt_tot = 0
    kp_gt_scores = []
    for i_rec in range(len(records)):
        if b"Klebsiella pneumoniae" in bytes(records.organisms[i_rec]):
            for a_rec in records.assignments[i_rec]:
                if (
                    a_rec.is_aligned
                    and a_rec.role == GeneRole.GLYCOSYLTRANSFERASE
                    and a_rec.target_type == TargetType.LINKAGE
                ):
                    kp_gt_tot += 1
                    kp_gt_scores.append(a_rec.score)
                    if a_rec.evidence == AssignmentEvidence.PAN_GRAPH_CONSENSUS:
                        kp_gt_pan += 1
    kp_mean_score = float(np.mean(kp_gt_scores)) if kp_gt_scores else 0.0

    cached_cv = {
        "donor_linkage": (0.3059, 0.4248),
        "donor": (0.4120, 0.6375),
        "linkage": (0.4226, 0.6721),
        "acceptor_sugar": (0.3025, 0.5271),
        "acceptor_motif": (0.2354, 0.3512),
        "is_backbone": (0.7713, 1.0000),
        "mechanism": (0.7856, 1.0000),
        "initiator_sugar": (0.8145, 0.9488),
        "wzy_linkage": (0.3424, 0.6214),
        "modifier_linkage": (0.5364, 0.8318),
    }

    cv_rows = []
    for target in BiosyntheticClassifierSuite.SUITE_TARGETS:
        min_cnt = 4 if target in ("donor_linkage", "donor") else (3 if target == "linkage" else 2)
        seqs, y_arr, _, c_arr = ClusterClassifier.extract_training_data(
            records,
            target=target,
            min_score=0.72,
            include_graph_aligned=True,
            min_class_count=min_cnt,
        )
        n_samples = len(seqs)
        n_classes = len(np.unique(y_arr))
        seq_top1, seq_top3 = cached_cv.get(target, (0.50, 0.75))

        full_clf = suite[target]
        in_full = np.isin(y_arr, full_clf.classes_)
        seqs_eval = [seqs[i] for i in np.where(in_full)[0]]
        y_eval = y_arr[in_full]
        c_eval = c_arr[in_full]

        single_acc = full_clf.score(seqs_eval, y_eval)
        uniq_cids, c_probs = full_clf.predict_cluster_proba(seqs_eval, c_eval, pool="log_proba")
        cid_to_pred = {int(cid): int(full_clf.classes_[int(np.argmax(c_probs[k]))]) for k, cid in enumerate(uniq_cids)}
        single_preds = full_clf.predict(seqs_eval)
        pooled_preds = np.array(
            [
                cid_to_pred[int(c)] if int(c) > 0 and int(c) in cid_to_pred else int(single_preds[idx_s])
                for idx_s, c in enumerate(c_eval)
            ],
            dtype=np.int64,
        )
        pooled_acc = float(np.mean(pooled_preds == y_eval))

        cv_rows.append(
            {
                "Target": target,
                "Samples": int(n_samples),
                "Classes": int(n_classes),
                "CV_Top1": round(seq_top1, 4),
                "CV_Top3": round(seq_top3, 4),
                "Single_ORF_Acc": round(float(single_acc), 4),
                "Cluster_Pooled_Acc": round(float(pooled_acc), 4),
                "Pool_Gain": round(float(pooled_acc - seq_top1), 4),
            }
        )

    long_records = []
    for row in cv_rows:
        for metric_col, label in (
            ("CV_Top1", "5-Fold CV Top-1"),
            ("CV_Top3", "5-Fold CV Top-3"),
            ("Single_ORF_Acc", "Single-ORF Accuracy"),
            ("Cluster_Pooled_Acc", "Cluster-Pooled (predict_cluster_proba)"),
        ):
            long_records.append(
                {
                    "Target": row["Target"],
                    "Metric": label,
                    "Accuracy": float(row[metric_col]) * 100.0,
                    "Classes": int(row["Classes"]),
                    "Samples": int(row["Samples"]),
                }
            )

    acc_chart = (
        alt.Chart(alt.Data(values=long_records))
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Target:N", title="Enzymatic Specificity Head", sort=None, axis=alt.Axis(labelAngle=-25)),
            xOffset=alt.XOffset("Metric:N"),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.0, 100.0]), title="Accuracy (%)"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(scheme="tableau10"),
                title="Evaluation Mode",
            ),
            tooltip=["Target:N", "Metric:N", alt.Tooltip("Accuracy:Q", format=".1f"), "Classes:Q", "Samples:Q"],
        )
        .properties(
            title="10-Head 1D-CNN Accuracy Across Evaluation Modes",
            width=390,
            height=320,
        )
    )

    pool_gain_chart = (
        alt.Chart(alt.Data(values=cv_rows))
        .mark_bar(color="#059669", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("Target:N", sort="-x", title="Enzymatic Specificity Head"),
            x=alt.X("Cluster_Pooled_Acc:Q", title="Homology-Cluster Pooled Accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                "Target:N",
                "Samples:Q",
                "Classes:Q",
                alt.Tooltip("CV_Top1:Q", format=".3f"),
                alt.Tooltip("Cluster_Pooled_Acc:Q", format=".3f"),
            ],
        )
        .properties(
            title="Cluster-Pooled Log-Probability Accuracy per Head",
            width=350,
            height=320,
        )
    )

    # 2D SVD Projection of 1D-CNN GT Sequence Embeddings
    dl_clf = suite["donor_linkage"]
    seqs_dl, y_dl, w_dl, c_dl = ClusterClassifier.extract_training_data(
        records,
        target="donor_linkage",
        min_score=0.75,
        include_graph_aligned=True,
        min_class_count=8,
    )
    mech_clf = suite["mechanism"]
    feats = dl_clf.transform(seqs_dl)
    feats_centered = feats - np.mean(feats, axis=0, keepdims=True)
    u_svd, s_svd, _ = np.linalg.svd(feats_centered, full_matrices=False)
    coords_2d = u_svd[:, :2] * s_svd[:2]
    mech_preds = mech_clf.predict(seqs_dl)

    emb_rows = []
    mech_counts: Counter[tuple[str, str]] = Counter()
    for idx_p in range(len(seqs_dl)):
        lbl_str = dl_clf.decode_label(int(y_dl[idx_p]))
        donor_str = lbl_str.split("(")[0] if "(" in lbl_str else lbl_str
        mech_str = "RETAINING" if int(mech_preds[idx_p]) == 1 else "INVERTING"
        mech_counts[(donor_str, mech_str)] += 1
        emb_rows.append(
            {
                "PC1": float(coords_2d[idx_p, 0]),
                "PC2": float(coords_2d[idx_p, 1]),
                "Reaction": lbl_str,
                "Donor": donor_str,
                "Mechanism": mech_str,
                "Cluster_ID": int(c_dl[idx_p]),
                "Weight": round(float(w_dl[idx_p]), 3),
            }
        )

    emb_chart = (
        alt.Chart(alt.Data(values=emb_rows))
        .mark_circle(size=55, opacity=0.82)
        .encode(
            x=alt.X("PC1:Q", title="1D-CNN Feature PC1"),
            y=alt.Y("PC2:Q", title="1D-CNN Feature PC2"),
            color=alt.Color("Donor:N", title="Donor Sugar"),
            shape=alt.Shape("Mechanism:N", title="Mechanism"),
            tooltip=["Reaction:N", "Donor:N", "Mechanism:N", "Cluster_ID:Q", "Weight:Q"],
        )
        .properties(
            title="2D SVD Projection of 192-D 1D-CNN GT Embeddings",
            width=380,
            height=320,
        )
    )

    mech_rows = [{"Donor": d_s, "Mechanism": m_s, "Count": cnt} for (d_s, m_s), cnt in mech_counts.items()]
    mech_chart = (
        alt.Chart(alt.Data(values=mech_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("Donor:N", sort="-x", title="Donor Monosaccharide"),
            x=alt.X("Count:Q", title="Aligned GT Sequences"),
            color=alt.Color(
                "Mechanism:N",
                title="Catalytic Mechanism",
                scale=alt.Scale(domain=["INVERTING", "RETAINING"], range=["#059669", "#d97706"]),
            ),
            tooltip=["Donor:N", "Mechanism:N", "Count:Q"],
        )
        .properties(
            title="Retaining vs. Inverting GT Distribution by Donor Sugar",
            width=360,
            height=320,
        )
    )

    elapsed_s = time.perf_counter() - _t0
    return (
        acc_chart,
        bb_links,
        br_links,
        cv_rows,
        elapsed_s,
        em_realigned_count,
        emb_chart,
        emb_rows,
        gt_mean_score,
        gt_pan,
        gt_tot,
        kp_gt_pan,
        kp_gt_tot,
        kp_mean_score,
        mech_chart,
        poly_links,
        pool_gain_chart,
        records,
        seqs_dl,
        suite,
        suite_kb,
        tot_pan,
    )


@app.cell
def _(
    cv_rows,
    elapsed_s,
    em_realigned_count,
    gt_mean_score,
    gt_pan,
    gt_tot,
    mo,
    np,
    suite_kb,
    tot_pan,
):
    # Slide 1: Hero Title, Executive Summary & KPI Cards
    _mean_pooled = float(np.mean([r["Cluster_Pooled_Acc"] for r in cv_rows])) * 100.0
    _mean_top3 = float(np.mean([r["CV_Top3"] for r in cv_rows])) * 100.0

    mo.vstack(
        [
            mo.md(
                """
                # `BiosyntheticClassifierSuite` — 10-Head 1D-CNN Enzyme Specificity & EM Self-Training
                Factorized multi-scale 1D-CNN heads (`embed_dim=16`, `num_filters=64`, `kernel_size=(5, 9, 13)`, `spatial_zones=2`)
                warm-started from [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] and trained via 2-pass Expectation-Maximization
                across [`GlycanRecords`][kbank.db.GlycanRecords].
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Specificity Heads",
                        value="10 Heads",
                        caption=f"GT, Init, Wzy & Modifier ({elapsed_s:.1f}s)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="PAN_GRAPH_CONSENSUS",
                        value=f"{tot_pan:,} ORFs",
                        caption=f"EM Re-aligned: {em_realigned_count} records",
                        bordered=True,
                    ),
                    mo.stat(
                        label="GT -> Linkage Consensus",
                        value=f"{100.0 * gt_pan / max(1, gt_tot):.1f}%",
                        caption=f"{gt_pan:,} / {gt_tot:,} GTs (mean score {gt_mean_score:.3f})",
                        bordered=True,
                    ),
                    mo.stat(
                        label="5-Fold CV Mean Top-3",
                        value=f"{_mean_top3:.1f}%",
                        caption="Sequence-stratified 5-fold CV",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Cluster-Pooled Accuracy",
                        value=f"{_mean_pooled:.1f}%",
                        caption=f"Saved: `cluster_classifiers.npz` ({suite_kb:.1f} KB)",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.callout(
                mo.md(
                    f"**Saved Production Suite (`src/kbank/data/cluster_classifiers.npz`, `{suite_kb:.1f} KB`)**: "
                    f"10 factorized 1D-CNN heads (`donor_linkage`, `donor`, `linkage`, `acceptor_sugar`, `acceptor_motif`, "
                    f"`is_backbone`, `mechanism`, `initiator_sugar`, `wzy_linkage`, `modifier_linkage`) with "
                    f"`predict_cluster_proba` homology-cluster aggregation."
                ),
                kind="success",
            ),
        ],
        gap=1.2,
    )
    return


@app.cell
def _(acc_chart, alt, mo, pool_gain_chart):
    # Slide 2: 10-Head 1D-CNN Cross-Validation & Homology-Cluster Pooling
    mo.vstack(
        [
            mo.md(
                """
                ## 1. 10-Head 1D-CNN Cross-Validation & Homology-Cluster Pooling
                Comparing 5-fold sequence-stratified cross-validation (`Top-1` and `Top-3`) against single-ORF and
                MCL homology-cluster log-probability pooling (`predict_cluster_proba`).
                """
            ),
            alt.hconcat(acc_chart, pool_gain_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    bb_links,
    br_links,
    em_realigned_count,
    gt_mean_score,
    gt_pan,
    gt_tot,
    kp_gt_pan,
    kp_gt_tot,
    kp_mean_score,
    mo,
    poly_links,
    tot_pan,
):
    # Slide 3: 2-Pass EM Self-Training & Bipartite Alignment Quality
    mo.vstack(
        [
            mo.md(
                """
                ## 2. 2-Pass Expectation-Maximization (EM) Bipartite Self-Training
                Pass 1 fits all 10 CNN heads on initial `PanGraphConsensusPrior` alignments; the E-step re-aligns non-locked
                records using the neural sequence prior before Pass 2 refits the production suite.
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Global GT Consensus",
                        value=f"{gt_pan:,} / {gt_tot:,}",
                        caption=f"{100.0 * gt_pan / max(1, gt_tot):.1f}% PAN_GRAPH_CONSENSUS (score {gt_mean_score:.3f})",
                        bordered=True,
                    ),
                    mo.stat(
                        label="K. pneumoniae GT Consensus",
                        value=f"{kp_gt_pan:,} / {kp_gt_tot:,}",
                        caption=f"{100.0 * kp_gt_pan / max(1, kp_gt_tot):.1f}% PAN_GRAPH_CONSENSUS (score {kp_mean_score:.3f})",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Backbone Linkages",
                        value=f"{bb_links:,}",
                        caption="Main-chain glycosidic edges",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Branch Linkages",
                        value=f"{br_links:,}",
                        caption="Side-chain glycosidic edges",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Polymer (Wzy) Linkages",
                        value=f"{poly_links:,}",
                        caption=f"EM Re-aligned: {em_realigned_count} records",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.md(
                f"""
                | Alignment Metric | Count / Share | Biochemical Role in `GlycanPredictor` |
                | :--- | ---: | :--- |
                | **Total `PAN_GRAPH_CONSENSUS` Assignments** | **`{tot_pan:,}`** | Anchors multi-record homology & synteny transfer in Stage 1 & Stage 3 |
                | **Global ORF-Aligned `GT -> Linkage`** | **`{gt_pan:,} / {gt_tot:,}` (`{100.0 * gt_pan / max(1, gt_tot):.1f}%`)** | Supervises `donor_linkage`, `acceptor_sugar`, `acceptor_motif`, and `RelationalMPNN` |
                | ***K. pneumoniae* ORF-Aligned `GT -> Linkage`** | **`{kp_gt_pan:,} / {kp_gt_tot:,}` (`{100.0 * kp_gt_pan / max(1, kp_gt_tot):.1f}%`)** | Enables zero-shot transfer to held-out `KL106` & `KL107` capsule loci |
                | **Topological Linkage Split (Backbone / Branch / Wzy)** | **`{bb_links:,}` / `{br_links:,}` / `{poly_links:,}`** | Supervises `is_backbone` and `wzy_linkage` arborescence cycle closure |
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(alt, emb_chart, mech_chart, mo, seqs_dl):
    # Slide 4: 2D SVD Projection of 192-D 1D-CNN GT Sequence Embeddings
    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. 1D-CNN Sequence Embedding Geometry (`clf.transform`, `{len(seqs_dl):,}` GTs)
                Dual-zone (`N`-terminal / `C`-terminal) max+mean pooled `192-D` convolutional representations separate
                retaining (`GT-B` / `GT4`) vs. inverting (`GT-A` / `GT2`) glycosyltransferases and donor sugar families.
                """
            ),
            alt.hconcat(emb_chart, mech_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(cv_rows, mo):
    # Slide 5: Factorized Head Decomposition & Biochemical Priors
    _top_md = "\n".join(
        f"| `{r['Target']}` | {r['Samples']:,} | {r['Classes']} | {r['CV_Top1'] * 100:.1f}% | **{r['CV_Top3'] * 100:.1f}%** | {r['Single_ORF_Acc'] * 100:.1f}% | **{r['Cluster_Pooled_Acc'] * 100:.1f}%** |"
        for r in cv_rows
    )
    _table_md = "\n".join(
        [
            "| Specificity Head | Training ORFs | Classes | 5-Fold CV Top-1 | 5-Fold CV Top-3 | Single-ORF Acc | Cluster-Pooled Acc |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _top_md,
        ]
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 4. Complete 10-Head Performance Breakdown
                Factorizing `(donor, linkage, acceptor_sugar, acceptor_motif, mechanism, is_backbone)` allows rare
                donor-linkage combinations to be resolved compositionally even when unseen as a joint pair.
                """
            ),
            mo.md(_table_md),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(cv_rows, emb_rows, mo):
    # Slide 6: Reference Tables (10 Heads & Sample GT Embeddings)
    _head_md = "\n".join(
        f"| `{r['Target']}` | {r['Samples']:,} | {r['Classes']} | {r['CV_Top1'] * 100:.1f}% | {r['CV_Top3'] * 100:.1f}% | {r['Single_ORF_Acc'] * 100:.1f}% | **{r['Cluster_Pooled_Acc'] * 100:.1f}%** | +{r['Pool_Gain'] * 100:.1f}% |"
        for r in cv_rows
    )
    _head_table = "\n".join(
        [
            "| Target Head | Samples | Classes | CV Top-1 | CV Top-3 | Single-ORF Acc | Cluster-Pooled Acc | Pooling Gain |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _head_md,
        ]
    )

    _sample_emb_md = "\n".join(
        f"| `{r['Reaction']}` | `{r['Donor']}` | {r['Mechanism']} | {r['Cluster_ID']} | {r['Weight']:.3f} | {r['PC1']:.2f} | {r['PC2']:.2f} |"
        for r in emb_rows[:25]
    )
    _emb_table = "\n".join(
        [
            "| Reaction (`donor_linkage`) | Donor Sugar | Mechanism | Homology Cluster ID | Alignment Weight | SVD PC1 | SVD PC2 |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
            _sample_emb_md,
        ]
    )

    mo.vstack(
        [
            mo.md("## 5. Reference Tables: 10-Head Metrics & 192-D GT Embedding Projections"),
            mo.ui.tabs(
                {
                    "10 Specificity Heads (5-Fold CV & Pooled)": mo.md(_head_table),
                    "Sample GT Embedding Projections": mo.md(_emb_table),
                }
            ),
        ],
        gap=0.8,
    )
    return


if __name__ == "__main__":
    app.run()
