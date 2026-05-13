from __future__ import annotations

import numpy as np
from scipy import sparse

from .markers import SUBMARKERS, SUBTYPES, SUBTYPETOSUPER, SUPERTYPES, rankspots, ucellscore


def siblinggroups():
    # Group subtypes by super type
    Groups = {}
    for Subtype in SUBTYPES:
        Supertype = SUBTYPETOSUPER[Subtype]
        if Supertype not in Groups:
            Groups[Supertype] = []
        SubtypeIndex = SUBTYPES.index(Subtype)
        Groups[Supertype].append(SubtypeIndex)

    return Groups


def tumorconfirm(adata):
    # Checks tumor spots using CNV labels
    Subclones = adata.obs["cnv_subclone"].astype(str).to_numpy()
    Confirm = Subclones != "normal"
    Confirm = Confirm & (Subclones != "")
    return Confirm


def subtypeanchors(adata, Dominance=1.5, ParentPi=0.5, ParentHist=0.3):
    Pi = np.asarray(adata.obsm["pi"])
    Hist = np.asarray(adata.obsm["histology_fractions"])
    SubScores = np.asarray(adata.obsm["marker_scores_sub"])
    Groups = siblinggroups()
    TumorConfirm = tumorconfirm(adata)
    Anchors = {}
    # Checks each subtype for dominance, parent confirmation, and presence test
    for SubtypeIndex, Subtype in enumerate(SUBTYPES):
        Supertype = SUBTYPETOSUPER[Subtype]
        SuperIndex = SUPERTYPES.index(Supertype)
        Siblings = Groups[Supertype]
        MyScore = SubScores[:, SubtypeIndex]
        SiblingScores = SubScores[:, Siblings]
        BestSibling = SiblingScores.max(axis=1)
        OtherScores = []

        for SiblingIndex in Siblings:
            if SiblingIndex != SubtypeIndex:
                OtherScores.append(SubScores[:, SiblingIndex])
        if len(OtherScores) == 0:
            SecondBest = np.zeros(adata.n_obs)
        else:
            OtherScores = np.array(OtherScores)
            SecondBest = OtherScores.max(axis=0)
        Dominant = MyScore >= BestSibling
        Ratio = MyScore / np.maximum(SecondBest, 1e-8)
        StrongEnough = Ratio >= Dominance
        ParentEnough = Pi[:, SuperIndex] > ParentPi

        if Supertype == "tumor":
            ParentConfirmed = TumorConfirm
        else:
            ParentConfirmed = Hist[:, SuperIndex] > ParentHist

        Anchors[Subtype] = Dominant & StrongEnough & ParentEnough & ParentConfirmed

    return Anchors


def residualanchors(adata, Supertype, Dominance=1.5, ParentPi=0.5, ParentHist=0.3):
    # Creates anchors for residual subtypes at the super type level
    Pi = np.asarray(adata.obsm["pi"])
    Hist = np.asarray(adata.obsm["histology_fractions"])
    SuperScores = np.asarray(adata.obsm["marker_scores_super"])
    TumorConfirm = tumorconfirm(adata)
    SuperIndex = SUPERTYPES.index(Supertype)
    MyScore = SuperScores[:, SuperIndex]
    OtherScores = []
    for OtherIndex in range(len(SUPERTYPES)):
        if OtherIndex != SuperIndex:
            OtherScores.append(SuperScores[:, OtherIndex])
    OtherScores = np.array(OtherScores)
    SecondBest = OtherScores.max(axis=0)

    BestSuper = SuperScores.max(axis=1)
    Dominant = MyScore >= BestSuper
    Ratio = MyScore / np.maximum(SecondBest, 1e-8)
    StrongEnough = Ratio >= Dominance
    ParentEnough = Pi[:, SuperIndex] > ParentPi
    if Supertype == "tumor":
        ParentConfirmed = TumorConfirm
    else:
        ParentConfirmed = Hist[:, SuperIndex] > ParentHist

    AnchorMask = Dominant & StrongEnough
    AnchorMask = AnchorMask & ParentEnough
    AnchorMask = AnchorMask & ParentConfirmed
    return AnchorMask


def presencetest(adata, Subtype, AnchorMask, Ranks, NullPanels=50, NullQuantile=0.99, MaxRank=1500):
    # Checks if the subtype's anchor pool scores higher than random gene set distribution through permutation test
    AnchorIndex = np.where(AnchorMask)[0]
    if len(AnchorIndex) < 3:
        Passed = True
        return Passed

    SubtypeIndex = SUBTYPES.index(Subtype)
    RealScores = adata.obsm["marker_scores_sub"][:, SubtypeIndex]
    RealMean = RealScores[AnchorIndex].mean()
    Genes = np.asarray(adata.var_names)
    PanelSize = len(SUBMARKERS[Subtype])
    if PanelSize == 0 or PanelSize > len(Genes):
        Passed = False
        return Passed

    Random = np.random.default_rng(0)
    NullMeans = []

    for NullRun in range(NullPanels):
        RandomGenes = Random.choice(Genes, size=PanelSize, replace=False)
        RandomPanel = {}
        RandomPanel["random"] = list(RandomGenes)
        RandomScore = ucellscore(Ranks, Genes, RandomPanel, MaxRank=MaxRank)
        RandomMean = RandomScore[AnchorIndex, 0].mean()
        NullMeans.append(RandomMean)

    NullMeans = np.array(NullMeans)
    Threshold = np.quantile(NullMeans, NullQuantile)
    Passed = RealMean > Threshold
    return Passed


def buildsignatures(adata, AnchorMatrix, TargetLib=1e4, PriorPseudoSpots=10.0):
    # Builds signatures for each anchor pool
    Counts = adata.layers["counts"]
    Counts = Counts.tocsr() if sparse.issparse(Counts) else sparse.csr_matrix(Counts)
    Libraries = np.asarray(Counts.sum(axis=1)).ravel()
    NumberGenes = adata.n_vars
    NumberLeaves = AnchorMatrix.shape[1]
    UnionAnchors = AnchorMatrix.any(axis=1)
    UnionCounts = Counts[UnionAnchors].sum(axis=0)
    UnionCounts = np.asarray(UnionCounts).ravel()
    UnionLibrary = Libraries[UnionAnchors].sum()
    if UnionLibrary == 0:
        UnionLibrary = 1
        UnionCounts = np.ones(NumberGenes) / NumberGenes

    # Pools counts and divides by library size to get the prior rate
    PriorRate = UnionCounts / UnionLibrary
    Signatures = np.zeros((NumberGenes, NumberLeaves), dtype=np.float32)
    for Leaf in range(NumberLeaves):
        LeafMask = AnchorMatrix[:, Leaf]
        LeafLibrary = Libraries[LeafMask].sum()

        if LeafMask.sum() == 0:
            Signature = PriorRate * TargetLib
            Signatures[:, Leaf] = Signature
            continue

        # Pools counts and divides by library size to get the posterior rate
        LeafCounts = Counts[LeafMask].sum(axis=0)
        LeafCounts = np.asarray(LeafCounts).ravel()
        # Gets the mean library size of the anchor pool
        MeanLibrary = Libraries[LeafMask].mean()
        if MeanLibrary == 0:
            MeanLibrary = 1
        # Create pseudo-spots and add back to real counts to get the posterior rate
        Numerator = LeafCounts + PriorPseudoSpots * PriorRate * MeanLibrary
        Denominator = LeafLibrary + PriorPseudoSpots * MeanLibrary
        if Denominator == 0:
            Denominator = 1
        Rate = Numerator / Denominator
        # Rescale
        Signature = Rate * TargetLib
        Signatures[:, Leaf] = Signature

    return Signatures


def anchorphase(adata, MinAnchors=15, Dominance=1.5, ParentPi=0.5, ParentHist=0.3, NullPanels=50, NullQuantile=0.99):
    # Main function to select anchors for each subtype and super type
    ad = adata.copy()
    Anchors = subtypeanchors(ad, Dominance=Dominance, ParentPi=ParentPi, ParentHist=ParentHist)
    Ranks = rankspots(ad, MaxRank=1500)

    LeafNames = []
    LeafToSuper = []
    AnchorMasks = []
    Merged = []

    for Supertype in SUPERTYPES:
        FailedSubtype = False
        for Subtype in SUBTYPES:
            if SUBTYPETOSUPER[Subtype] != Supertype:
                continue
            # Checks if the subtype has enough anchors and is present
            AnchorMask = Anchors[Subtype]
            EnoughAnchors = AnchorMask.sum() >= MinAnchors
            Present = presencetest(ad, Subtype, AnchorMask, Ranks, NullPanels=NullPanels, NullQuantile=NullQuantile)
            # If the subtype has enough anchors and is present, add it to the leaves
            if EnoughAnchors and Present:
                LeafNames.append(Subtype)
                LeafToSuper.append(SUPERTYPES.index(Supertype))
                AnchorMasks.append(AnchorMask)
            else:
                FailedSubtype = True
                Merged.append(Subtype)

        # If the subtype has failed, add a residual anchor
        if FailedSubtype:
            ResidualName = Supertype + "_residual"
            ResidualMask = residualanchors(ad, Supertype, Dominance=Dominance, ParentPi=ParentPi, ParentHist=ParentHist)
            LeafNames.append(ResidualName)
            LeafToSuper.append(SUPERTYPES.index(Supertype))
            AnchorMasks.append(ResidualMask)

    AnchorMatrix = np.array(AnchorMasks)
    AnchorMatrix = AnchorMatrix.T
    AnchorMatrix = AnchorMatrix.astype(bool)
    Signatures = buildsignatures(ad, AnchorMatrix)

    # Get the number of anchors for each leaf
    AnchorCounts = []
    for Leaf in range(AnchorMatrix.shape[1]):
        Count = AnchorMatrix[:, Leaf].sum()
        AnchorCounts.append(int(Count))

    ad.obsm["anchors"] = AnchorMatrix
    ad.varm["signatures"] = Signatures
    ad.uns["leaf_names"] = LeafNames
    ad.uns["leaf_to_supertype"] = np.array(LeafToSuper, dtype=np.int32)
    ad.uns["anchor_counts"] = AnchorCounts
    ad.uns["merged_subtypes"] = Merged

    return ad
