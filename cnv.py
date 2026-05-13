from __future__ import annotations

from pathlib import Path

import infercnvpy as cnv
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy import sparse
from sklearn.metrics import silhouette_score


def normalspots(adata, MinNormal=50):
    Scores = adata.obsm["marker_scores_super_raw"]
    Tumor = Scores[:, 0]
    Immune = Scores[:, 1]
    Stromal = Scores[:, 2]
    TumorLimit = np.quantile(Tumor, 0.40)
    ImmuneLimit = np.quantile(Immune, 0.60)
    StromalLimit = np.quantile(Stromal, 0.60)
    Normal = Tumor <= TumorLimit
    Normal = Normal & ((Immune >= ImmuneLimit) | (Stromal >= StromalLimit))

    Step = 0
    while Normal.sum() < MinNormal and Step < 5:
        TumorLimit = np.quantile(Tumor, 0.40 + 0.05 * Step)
        ImmuneLimit = np.quantile(Immune, 0.60 - 0.05 * Step)
        StromalLimit = np.quantile(Stromal, 0.60 - 0.05 * Step)
        Normal = Tumor <= TumorLimit
        Normal = Normal & ((Immune >= ImmuneLimit) | (Stromal >= StromalLimit))
        Step = Step + 1

    return Normal


def chromosomecnv(CnvMatrix, ChromosomeStarts):
    NumberBins = CnvMatrix.shape[1]
    ChromosomeItems = list(ChromosomeStarts.items())
    ChromosomeItems.sort(key=lambda Item: Item[1])
    Starts = []
    for Chromosome, Start in ChromosomeItems:
        Starts.append(Start)
    Starts.append(NumberBins)

    Magnitude = np.zeros(CnvMatrix.shape[0], dtype=np.float32)
    for i in range(len(Starts) - 1):
        Start = Starts[i]
        End = Starts[i + 1]
        ChromosomeBlock = CnvMatrix[:, Start:End]
        ChromosomeMean = ChromosomeBlock.mean(axis=1)
        Magnitude = Magnitude + np.abs(ChromosomeMean)

    return Magnitude


def choosesubclones(CnvTumor, MaxSubclones=5, BootstrapSamples=20, SampleFraction=0.8, MinClusterSpots=15, RandomSeed=0):
    NumberTumor = CnvTumor.shape[0]
    if NumberTumor < 10:
        Labels = np.zeros(NumberTumor, dtype=np.int32)
        Scores = {1: 0}
        BestK = 1
        return Labels, BestK, Scores

    PossibleMax = min(MaxSubclones, NumberTumor - 1)
    Random = np.random.default_rng(RandomSeed)
    Scores = {}

    for K in range(2, PossibleMax + 1):
        KScores = []

        for Bootstrap in range(BootstrapSamples):
            SampleSize = NumberTumor * SampleFraction
            SampleSize = int(np.ceil(SampleSize))
            SampleSize = max(SampleSize, K + 2)
            SampleSize = min(SampleSize, NumberTumor)
            SampleIndex = Random.choice(NumberTumor, size=SampleSize, replace=False)
            SampleCnv = CnvTumor[SampleIndex, :]
            LinkageTree = linkage(SampleCnv, method="ward")
            SampleLabels = fcluster(LinkageTree, t=K, criterion="maxclust")
            SampleLabels = SampleLabels - 1
            NumberLabels = len(set(SampleLabels.tolist()))
            if NumberLabels < 2:
                continue
            ClusterCounts = np.bincount(SampleLabels)
            SmallestCluster = ClusterCounts.min()
            if SmallestCluster < MinClusterSpots:
                continue
            Score = silhouette_score(SampleCnv, SampleLabels)
            KScores.append(Score)

        if len(KScores) > 0:
            MeanScore = np.mean(KScores)
            Scores[K] = MeanScore

    if len(Scores) == 0:
        Labels = np.zeros(NumberTumor, dtype=np.int32)
        Scores = {1: 0}
        BestK = 1
        return Labels, BestK, Scores

    BestK = max(Scores, key=Scores.get)
    LinkageTree = linkage(CnvTumor, method="ward")
    Labels = fcluster(LinkageTree, t=BestK, criterion="maxclust")
    Labels = Labels - 1
    Labels = Labels.astype(np.int32)
    ClusterCounts = np.bincount(Labels)
    if ClusterCounts.min() < MinClusterSpots:
        Labels = np.zeros(NumberTumor, dtype=np.int32)
        BestK = 1

    return Labels, BestK, Scores


def cnvscore(adata, GtfFile, TumorQuantile=0.65, WindowSize=100, MaxSubclones=10, MinClusterSpots=15):
    ad = adata.copy()
    GtfFile = Path(GtfFile)
    cnv.io.genomic_position_from_gtf(str(GtfFile), adata=ad, gtf_gene_id="gene_name")
    VarIndex = np.array([str(Name) for Name in ad.var.index], dtype=object)
    ad.var.index = pd.Index(VarIndex, dtype=object)

    for Column in list(ad.var.columns):
        Values = ad.var[Column]
        if pd.api.types.is_string_dtype(Values):
            NewValues = []
            for Value in Values:
                if pd.isna(Value):
                    NewValues.append(None)
                else:
                    NewValues.append(str(Value))
            ad.var[Column] = np.array(NewValues, dtype=object)

    Normal = normalspots(ad)
    CnvReference = np.full(ad.n_obs, "observation", dtype=object)
    CnvReference[Normal] = "normal"
    ad.obs["cnv_reference"] = pd.Categorical(
        CnvReference,
        categories=["normal", "observation"],
    )
    cnv.tl.infercnv(
        ad,
        reference_key="cnv_reference",
        reference_cat=["normal"],
        window_size=WindowSize,
        layer="counts",
    )
    del ad.obs["cnv_reference"]

    CnvMatrix = ad.obsm["X_cnv"]
    if sparse.issparse(CnvMatrix):
        CnvMatrix = CnvMatrix.toarray()
    CnvMatrix = np.asarray(CnvMatrix, dtype=np.float32)
    ChromosomeStarts = dict(ad.uns["cnv"]["chr_pos"])

    Magnitude = chromosomecnv(CnvMatrix, ChromosomeStarts)
    PhiLimit = np.percentile(Magnitude, 95)
    if PhiLimit == 0:
        PhiLimit = 1
    Phi = Magnitude / PhiLimit
    Phi = np.clip(Phi, 0, 1)

    TumorLimit = np.quantile(Magnitude, TumorQuantile)
    IsTumor = Magnitude >= TumorLimit
    TumorCnv = CnvMatrix[IsTumor, :]
    TumorLabels, NumberSubclones, SubcloneScores = choosesubclones(
        TumorCnv,
        MaxSubclones=MaxSubclones,
        BootstrapSamples=20,
        SampleFraction=0.8,
        MinClusterSpots=MinClusterSpots,
        RandomSeed=0,
    )

    Subclone = np.full(ad.n_obs, "normal", dtype=object)
    SubcloneIndex = np.full(ad.n_obs, -1, dtype=np.int32)
    TumorSpotIndex = np.where(IsTumor)[0]
    SubcloneNames = []
    for K in range(NumberSubclones):
        SubcloneName = "tumor_subclone_" + str(K)
        SubcloneNames.append(SubcloneName)

    for TumorNumber in range(len(TumorSpotIndex)):
        Spot = TumorSpotIndex[TumorNumber]
        Label = TumorLabels[TumorNumber]
        Subclone[Spot] = SubcloneNames[Label]
        SubcloneIndex[Spot] = Label

    ad.obs["cnv_magnitude"] = Magnitude.astype(np.float32)
    ad.obs["cnv_phi"] = Phi.astype(np.float32)
    ad.obs["is_tumor_candidate"] = IsTumor
    ad.obs["cnv_subclone"] = Subclone
    ad.obs["cnv_subclone_index"] = SubcloneIndex
    ad.obsm["X_cnv"] = CnvMatrix
    ad.uns["cnv_subclones"] = SubcloneNames
    ad.uns["cnv_subclone_scores"] = SubcloneScores

    return ad
