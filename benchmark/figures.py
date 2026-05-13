from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def spotcoords(adata, HiresScale):
    X = adata.obs["pxl_col_in_fullres"].to_numpy() * HiresScale
    Y = adata.obs["pxl_row_in_fullres"].to_numpy() * HiresScale
    return X, Y


def zscore(Values):
    Values = np.asarray(Values, dtype=np.float64)
    Std = Values.std()
    if Std == 0:
        Std = 1
    Z = (Values - Values.mean()) / Std
    return Z


def plotspots(Image, X, Y, Values, Title, Cmap="viridis", SpotSize=4):
    Figure, Axis = plt.subplots(figsize=(6, 6))
    Axis.imshow(Image)
    Scatter = Axis.scatter(X, Y, c=Values, cmap=Cmap, s=SpotSize, alpha=0.75)
    Axis.set_title(Title)
    Axis.axis("off")
    Figure.colorbar(Scatter, ax=Axis, fraction=0.035, pad=0.02)
    return Figure, Axis


def plotclonelegend(Image, X, Y, Labels, Title, SpotSize=4):
    Figure, Axis = plt.subplots(figsize=(6, 6))
    Axis.imshow(Image)
    LabelValues = np.asarray(Labels, dtype=object)
    UniqueLabels = list(pd.unique(LabelValues))
    UniqueLabels.sort()
    Colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(UniqueLabels), 1)))

    for Index, Label in enumerate(UniqueLabels):
        Mask = LabelValues == Label
        Axis.scatter(X[Mask], Y[Mask], color=Colors[Index], s=SpotSize, alpha=0.75, label=str(Label))

    Axis.set_title(Title)
    Axis.axis("off")
    Axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2)
    return Figure, Axis


def clonefigure(adata, Image, HiresScale):
    X, Y = spotcoords(adata, HiresScale)
    CloneCounts = adata.obs["cnv_subclone"].value_counts().rename_axis("clone").reset_index(name="spots")
    CloneScores = pd.Series(adata.uns.get("cnv_subclone_scores", {}), name="silhouette").rename_axis("k").reset_index()

    plotspots(
        Image,
        X,
        Y,
        adata.obs["cnv_phi"].to_numpy(),
        "CNV magnitude",
        Cmap="viridis",
    )
    plotclonelegend(Image, X, Y, adata.obs["cnv_subclone"].astype(str).to_numpy(), "CNV subclone")

    return CloneCounts, CloneScores


def necrosisscore(adata):
    Hist = np.asarray(adata.obsm["hist_features"], dtype=np.float64)
    Names = list(adata.uns["hist_feature_names"])
    HCoverage = Hist[:, Names.index("H_coverage")]
    EMean = Hist[:, Names.index("E_mean")]
    HMean = Hist[:, Names.index("H_mean")]
    EosinDominance = EMean - HMean
    Score = 0.5 * (-zscore(HCoverage) + zscore(EMean))
    return Score, HCoverage, EosinDominance


def necrosisfigure(adata, Image, HiresScale, GroundTruth=None, Quantile=0.95):
    Score, HCoverage, EosinDominance = necrosisscore(adata)
    Threshold = np.quantile(Score, Quantile)
    Flagged = Score >= Threshold
    X, Y = spotcoords(adata, HiresScale)

    Summary = {
        "threshold": float(Threshold),
        "flagged_spots": int(Flagged.sum()),
        "total_spots": int(len(Flagged)),
    }

    if GroundTruth is not None:
        Mask = np.asarray(GroundTruth["gt_mask"], dtype=bool)
        Summary["flagged_in_gt_mask"] = int((Flagged & Mask).sum())
        Summary["flagged_out_gt_mask"] = int((Flagged & ~Mask).sum())

    if "beta_smooth" in adata.obsm:
        LeafNames = list(adata.uns["leaf_names"])
        TumorColumns = []
        for Index, Name in enumerate(LeafNames):
            if Name in ("tumor_in_situ", "tumor_invasive"):
                TumorColumns.append(Index)
        if len(TumorColumns) > 0:
            TumorMass = adata.obsm["beta_smooth"][:, TumorColumns].sum(axis=1)
            Summary["mean_tumor_flagged"] = float(TumorMass[Flagged].mean())
            Summary["mean_tumor_other"] = float(TumorMass[~Flagged].mean())

    SummaryTable = pd.DataFrame([Summary])

    Figure, Axes = plt.subplots(1, 2, figsize=(12, 5))
    Scatter = Axes[0].scatter(HCoverage, EosinDominance, c=Score, cmap="viridis", s=5, alpha=0.65)
    Axes[0].scatter(HCoverage[Flagged], EosinDominance[Flagged], facecolors="none", edgecolors="red", s=40, linewidths=0.8)
    Axes[0].set_xlabel("H_coverage")
    Axes[0].set_ylabel("E_mean - H_mean")
    Axes[0].set_title("Necrosis score features")
    Figure.colorbar(Scatter, ax=Axes[0], fraction=0.035, pad=0.02)

    Axes[1].imshow(Image)
    Axes[1].scatter(X, Y, c=Score, cmap="viridis", s=4, alpha=0.6)
    Axes[1].scatter(X[Flagged], Y[Flagged], facecolors="none", edgecolors="red", s=40, linewidths=0.8)
    Axes[1].set_title("Top necrosis-score spots")
    Axes[1].axis("off")
    plt.tight_layout()

    return SummaryTable
