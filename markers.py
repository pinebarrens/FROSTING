from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata


SUPERTYPES = ("tumor", "immune", "stromal", "normal_epi")

SUPERMARKERS = {
    "tumor": ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2", "ESR1", "PGR", "MKI67", "FOXA1", "GATA3", "MUC1", "AGR2"],
    "immune": ["PTPRC", "CD3D", "CD3E", "CD2", "CD68", "CD14", "LYZ", "ITGAX", "MS4A1", "CD79A", "CD79B", "IGKC", "JCHAIN", "TPSAB1", "CPA3"],
    "stromal": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "ACTA2", "FAP", "PDGFRA", "PDGFRB", "PECAM1", "VWF", "CDH5", "CD34", "RGS5", "MCAM"],
    "normal_epi": ["KRT5", "KRT14", "KRT17", "TP63", "KIT", "ALDH1A3", "KRT8", "KRT18", "ESR1", "GATA3", "PGR", "AGR2"],
}

SUBTYPES = (
    "tumor_in_situ", "tumor_invasive",
    "T_cell", "NK", "B_cell", "plasma", "myeloid", "mast",
    "CAF", "endothelial", "perivascular",
    "luminal_normal", "basal_normal",
)

SUBTYPETOSUPER = {
    "tumor_in_situ": "tumor",
    "tumor_invasive": "tumor",
    "T_cell": "immune",
    "NK": "immune",
    "B_cell": "immune",
    "plasma": "immune",
    "myeloid": "immune",
    "mast": "immune",
    "CAF": "stromal",
    "endothelial": "stromal",
    "perivascular": "stromal",
    "luminal_normal": "normal_epi",
    "basal_normal": "normal_epi",
}

SUBMARKERS = {
    "tumor_in_situ": ["EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "ERBB2", "ESR1", "MUC1", "FOXA1", "GATA3", "AGR2", "CDH1", "CLDN3", "CLDN4", "KRT5", "KRT14", "TP63", "ACTA2", "MKI67"],
    "tumor_invasive": ["EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "ERBB2", "ESR1", "MUC1", "VIM", "SNAI2", "ZEB1", "ZEB2", "CDH2", "MMP2", "MMP9", "FN1", "CD44", "MKI67", "TOP2A"],
    "T_cell": ["CD3D", "CD3E", "CD3G", "CD2", "CD7", "CD4", "CD8A", "CCL5", "CCR7", "CD28", "CD40LG", "IL7R", "IL2RA", "GZMA", "GZMB", "GZMK", "PRF1", "NKG7", "KLRD1"],
    "NK": ["NKG7", "KLRD1", "KLRC1", "KLRB1", "NCR1", "NCAM1", "GNLY", "XCL1", "CD8A"],
    "B_cell": ["MS4A1", "CD19", "CD20", "CD22", "CD79A", "CD79B", "BANK1", "TNFRSF13B", "CD27", "CD38", "IGHM", "IGHD"],
    "plasma": ["IGKC", "IGHG1", "IGHG2", "IGHA1", "JCHAIN", "MZB1", "XBP1", "DERL3", "SDC1"],
    "myeloid": ["CD68", "CD14", "CD163", "LYZ", "ITGAX", "ITGAM", "FCGR3A", "C1QA", "C1QB", "C1QC", "AIF1", "CSF1R", "MARCO", "MSR1", "CD86", "HLA-DRA"],
    "mast": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HDC", "CTSG"],
    "CAF": ["COL1A1", "COL1A2", "COL3A1", "FAP", "PDGFRA", "PDGFRB", "DCN", "LUM", "POSTN", "MMP11", "SPARC", "S100A4", "ACTA2", "THY1"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "CD34", "ENG", "CLDN5", "ESAM", "KDR", "EMCN", "CLEC14A", "SOX18"],
    "perivascular": ["RGS5", "MCAM", "PDGFRB", "MYH11", "MYL9", "ACTA2", "TAGLN", "CAV1"],
    "luminal_normal": ["KRT8", "KRT18", "KRT19", "ESR1", "PGR", "GATA3", "FOXA1", "AGR2", "ANKRD30A", "MUC1", "KIT", "KRT15"],
    "basal_normal": ["KRT5", "KRT14", "KRT17", "TP63", "KIT", "ALDH1A3", "MYLK", "ACTG2"],
}


def geneweight(MarkerPanels):
    # Down weight genes that are in multiple panels
    Count = {}
    for Genes in MarkerPanels.values():
        for Gene in Genes:
            Count[Gene] = Count.get(Gene, 0) + 1

    Weights = {}
    for Gene in Count:
        Weights[Gene] = 1 / Count[Gene]

    return Weights


def rankspots(adata, MaxRank=1500):
    # Get the ranks of each gene at each spot
    X = adata.layers["counts"].tocsr()
    Ranks = np.empty_like(X.data, dtype=np.int32)

    for Spot in range(X.shape[0]):
        Start = X.indptr[Spot]
        End = X.indptr[Spot + 1]
        Values = X.data[Start:End]
        if Values.size == 0:
            continue

        NegativeValues = -Values
        SpotRanks = rankdata(NegativeValues, method="average")
        SpotRanks = np.minimum(SpotRanks, MaxRank + 1)
        Ranks[Start:End] = SpotRanks

    RankMatrix = sparse.csr_matrix((Ranks, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    return RankMatrix


def ucellscore(Ranks, GeneNames, MarkerPanels, MaxRank=1500):
    # Compute scores for each spot for each gene marker list
    GeneIndex = {}
    for i, Gene in enumerate(GeneNames):
        GeneIndex[Gene] = i

    Weights = geneweight(MarkerPanels)
    Scores = np.zeros((Ranks.shape[0], len(MarkerPanels)), dtype=np.float32)
    # Compute scores for each spot for each gene marker list
    for PanelNumber, (PanelName, Genes) in enumerate(MarkerPanels.items()):
        Columns = []
        PanelWeights = []
        for Gene in Genes:
            if Gene in GeneIndex:
                Columns.append(GeneIndex[Gene])
                PanelWeights.append(Weights[Gene])

        PanelWeights = np.array(PanelWeights)
        TotalWeight = PanelWeights.sum()
        if len(Columns) == 0 or TotalWeight == 0:
            Scores[:, PanelNumber] = 0
            continue

        RankSum = np.zeros(Ranks.shape[0])
        SeenWeight = np.zeros(Ranks.shape[0])
        SeenCount = np.zeros(Ranks.shape[0])

        for Column, Weight in zip(Columns, PanelWeights):
            GeneRanks = Ranks[:, Column].tocsc()
            Spots = GeneRanks.indices
            Values = GeneRanks.data
            RankSum[Spots] += Weight * Values
            SeenWeight[Spots] += Weight
            SeenCount[Spots] += 1

        MissingWeight = TotalWeight - SeenWeight
        MissingPenalty = MissingWeight * (MaxRank + 1)
        RankSum = RankSum + MissingPenalty
        # Compare mean rank with "ideal" rank then normalize to 0-1
        MeanRank = RankSum / TotalWeight
        PerfectRank = (len(Columns) + 1) / 2
        Score = 1 - ((MeanRank - PerfectRank) / MaxRank)
        Score = np.clip(Score, 0, 1)
        Coherence = SeenCount / len(Columns)
        Score = Score * Coherence
        Scores[:, PanelNumber] = Score.astype(np.float32)

    return Scores


def markerscore(adata, MaxRank=1500):
    # Compute scores for each spot for each super and sub type
    ad = adata.copy()
    GeneNames = np.asarray(ad.var_names)
    Ranks = rankspots(ad, MaxRank=MaxRank)
    SuperScoresRaw = ucellscore(Ranks, GeneNames, SUPERMARKERS, MaxRank=MaxRank)
    SubScores = ucellscore(Ranks, GeneNames, SUBMARKERS, MaxRank=MaxRank)
    RowSums = SuperScoresRaw.sum(axis=1, keepdims=True)
    RowSums = np.where(RowSums > 0, RowSums, 1)

    ad.obsm["marker_scores_super_raw"] = SuperScoresRaw
    ad.obsm["marker_scores_super"] = SuperScoresRaw / RowSums
    ad.obsm["marker_scores_sub"] = SubScores
    ad.uns["supertype_names"] = list(SUPERTYPES)
    ad.uns["subtype_names"] = list(SUBTYPES)

    SubtypeToSuperIndex = []
    for Subtype in SUBTYPES:
        Supertype = SUBTYPETOSUPER[Subtype]
        SuperIndex = SUPERTYPES.index(Supertype)
        SubtypeToSuperIndex.append(SuperIndex)

    ad.uns["subtype_to_supertype_index"] = np.array(SubtypeToSuperIndex)
    return ad
