from __future__ import annotations

import scanpy as sc


def qcfilter(adata, MinCounts=500, MinGenes=200, MaxMito=15):
    ad = adata.copy()
    ad.layers["counts"] = ad.X.copy()

    sc.pp.filter_cells(ad, min_counts=MinCounts)
    sc.pp.filter_cells(ad, min_genes=MinGenes)
    ad.var["mt"] = ad.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], inplace=True)
    return ad[ad.obs["pct_counts_mt"] <= MaxMito].copy()
