from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image


def loadspatial(SpatialData, TissuePositions, ScaleFactors, SpatialFolder, SampleID="sample"):
    SpatialData = Path(SpatialData)
    TissuePositions = Path(TissuePositions)
    ScaleFactors = Path(ScaleFactors)
    SpatialFolder = Path(SpatialFolder)

    adata = sc.read_10x_h5(SpatialData)
    adata.var_names_make_unique()
    adata.obs_names = pd.Index([str(x) for x in adata.obs_names], dtype=object)
    adata.var_names = pd.Index([str(x) for x in adata.var_names], dtype=object)

    pos = pd.read_csv(TissuePositions)
    pos["barcode"] = pos["barcode"].astype(str)
    pos = pos.set_index("barcode", drop=True)
    common = adata.obs_names.intersection(pos.index)
    adata = adata[common].copy()
    pos = pos.loc[common].copy()
    keep = pos["in_tissue"].astype(int) == 1
    adata = adata[keep.to_numpy()].copy()
    pos = pos.loc[adata.obs_names].copy()

    for col in ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
        adata.obs[col] = pos[col].to_numpy()

    with ScaleFactors.open("r") as f:
        adata.uns["scalefactors"] = json.load(f)
    adata.uns["sample_id"] = SampleID
    adata.uns["spatial_dir"] = str(SpatialFolder)
    adata.obsm["spatial"] = adata.obs[["array_row", "array_col"]].to_numpy()
    return adata


def loadhires(ImageFile, ScaleFactors):
    ImageFile = Path(ImageFile)
    ScaleFactors = Path(ScaleFactors)

    with ScaleFactors.open("r") as f:
        scale = float(json.load(f)["tissue_hires_scalef"])
    return np.asarray(Image.open(ImageFile).convert("RGB")), scale
