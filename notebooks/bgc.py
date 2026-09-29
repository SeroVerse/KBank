# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kaptive @ file:///Users/tsta0015/Programming/Kaptive",
#     "marimo>=0.23.16",
#     "kbank @ file:///Users/tsta0015/Programming/KaptiveBank",
#     "torch>=2"
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from kaptive.core.interval import Strand, Interval, Intervals
    from gb_io import iter as GenbankIterator
    from pathlib import Path
    from kbank.genes.roles import GeneRoles
    from kbank.loci.loci import GeneCluster, GeneNode


@app.cell
def _():
    roles_db = GeneRoles()
    clusters = []

    with Path('notebooks/strep_pneumo.gbk').open('rb') as fh:
        for rec in GenbankIterator(fh):
            intervals = []
            gene_nodes = []
            strands = []
            for feat in rec.features:
                if feat.kind == 'CDS':
                    strand = Strand(feat.location.strand)
                    intervals.append(Interval(start=feat.location.start, end=feat.location.end, strand=strand))
                    strands.append(strand)
                    gene_names = [q.value for q in feat.qualifiers if q.key in ('gene', 'product')]
                    query_name = gene_names[0].encode() if gene_names else b'unknown'
                    entry = roles_db.label(query_name)
                    gene_nodes.append(GeneNode.from_role(entry))

            cluster = GeneCluster(
                nodes=tuple(gene_nodes),
                edges=GeneCluster.create_intergenic_edges(strands),
            )
            clusters.append((rec.name, cluster, Intervals.from_intervals(intervals)))
            break
    return (clusters,)


if __name__ == "__main__":
    app.run()
