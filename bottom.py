from __future__ import annotations

import numpy as np
from scipy import sparse


def selectgenes(Signatures, MaxGenes=2000):
    # Use sub-type contrasting genes
    Positive = np.maximum(Signatures, 0)
    GeneSum = Positive.sum(axis=1)
    Keep = GeneSum > 0
    GeneMax = Positive.max(axis=1)
    GeneMean = Positive.mean(axis=1)
    Contrast = GeneMax / np.maximum(GeneMean, 1e-6)
    Contrast[~Keep] = -np.inf

    Order = np.argsort(-Contrast)
    NumberGenes = min(MaxGenes, Keep.sum())
    NumberGenes = max(NumberGenes, 1)
    GeneIndex = Order[:NumberGenes]

    return GeneIndex


def preparedata(adata, Signatures, GeneIndex, TargetLib=1e4):
    Counts = adata.layers["counts"]
    Counts = Counts.tocsr() if sparse.issparse(Counts) else sparse.csr_matrix(Counts)
    SelectedCounts = Counts[:, GeneIndex]
    SelectedCounts = SelectedCounts.toarray()
    Libraries = np.asarray(Counts.sum(axis=1)).ravel()
    Libraries = np.maximum(Libraries, 1)
    Cp10k = SelectedCounts * (TargetLib / Libraries[:, None])
    X = np.log1p(Cp10k)

    SelectedSignatures = Signatures[GeneIndex, :]
    SelectedSignatures = np.maximum(SelectedSignatures, 0)
    S = np.log1p(SelectedSignatures)
    X = X.astype(np.float32)
    S = S.astype(np.float32)
    return X, S


def projectsimplex(Vector, Total):
    if Total <= 0:
        Projection = np.zeros(Vector.shape[0])
        return Projection

    Sorted = np.sort(Vector)
    Sorted = Sorted[::-1]
    RunningSum = 0
    Rho = 0
    Theta = 0

    # Projects a vector onto the simplex
    for i in range(len(Sorted)):
        RunningSum = RunningSum + Sorted[i]
        CandidateTheta = (RunningSum - Total) / (i + 1)
        CandidateValue = Sorted[i] - CandidateTheta
        if CandidateValue > 0:
            Rho = i + 1
            Theta = CandidateTheta

    if Rho == 0:
        Projection = np.zeros(Vector.shape[0])
        return Projection

    Projection = Vector - Theta
    Projection = np.maximum(Projection, 0)
    return Projection


def projectparents(Beta, Pi, LeafToSuper):
    # Project the subtypes of each spot onto the parents
    Projected = np.zeros(Beta.shape)
    NumberSpots = Beta.shape[0]
    NumberSupertypes = Pi.shape[1]

    for Spot in range(NumberSpots):
        for Supertype in range(NumberSupertypes):
            LeafIndex = np.where(LeafToSuper == Supertype)[0]
            Values = Beta[Spot, LeafIndex]
            Total = Pi[Spot, Supertype]
            ProjectedValues = projectsimplex(Values, Total)
            Projected[Spot, LeafIndex] = ProjectedValues

    return Projected


def startbeta(Pi, LeafToSuper):
    # Start with a uniform distribution for each spot
    NumberSpots = Pi.shape[0]
    NumberLeaves = LeafToSuper.shape[0]
    NumberSupertypes = Pi.shape[1]
    Beta = np.zeros((NumberSpots, NumberLeaves), dtype=np.float64)

    for Supertype in range(NumberSupertypes):
        LeafIndex = np.where(LeafToSuper == Supertype)[0]
        NumberChildren = len(LeafIndex)
        if NumberChildren == 0:
            continue
        Share = Pi[:, Supertype] / NumberChildren
        for Leaf in LeafIndex:
            Beta[:, Leaf] = Share

    return Beta


def solvebeta(X, S, Pi, LeafToSuper, Ridge=1e-3, MaxIter=300, Tolerance=1e-6):
    # Solve for the bottom-level subtype fractions
    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Pi = np.asarray(Pi, dtype=np.float64)
    LeafToSuper = np.asarray(LeafToSuper, dtype=np.int32)

    NumberLeaves = S.shape[1]
    StS = S.T @ S
    StS = StS + Ridge * np.eye(NumberLeaves)
    XtS = X @ S
    Eigenvalues = np.linalg.eigvalsh(StS)
    LargestEigenvalue = Eigenvalues.max()
    if LargestEigenvalue <= 0:
        Beta = startbeta(Pi, LeafToSuper)
        return Beta.astype(np.float32)

    StepSize = 1 / (2 * LargestEigenvalue + 1e-8)
    Beta = startbeta(Pi, LeafToSuper)
    Y = Beta.copy()
    T = 1

    for Step in range(MaxIter):
        Gradient = Y @ StS
        Gradient = Gradient - XtS
        Gradient = 2 * Gradient

        Candidate = Y - StepSize * Gradient
        NewBeta = projectparents(Candidate, Pi, LeafToSuper)
        NewT = 1 + 4 * T * T
        NewT = np.sqrt(NewT)
        NewT = 1 + NewT
        NewT = NewT / 2

        Momentum = (T - 1) / NewT
        Y = NewBeta + Momentum * (NewBeta - Beta)
        Difference = np.linalg.norm(NewBeta - Beta)
        Denominator = np.linalg.norm(Beta)
        Denominator = max(Denominator, 1e-12)
        RelativeChange = Difference / Denominator

        Beta = NewBeta
        T = NewT

        if RelativeChange < Tolerance:
            break

    return Beta.astype(np.float32)


def bottomphase(adata, MaxGenes=2000, Ridge=1e-3, MaxIter=300, Tolerance=1e-6):
    ad = adata.copy()

    Signatures = np.asarray(ad.varm["signatures"])
    Pi = np.asarray(ad.obsm["pi"])
    LeafToSuper = np.asarray(ad.uns["leaf_to_supertype"])
    GeneIndex = selectgenes(Signatures, MaxGenes=MaxGenes)
    X, S = preparedata(ad, Signatures, GeneIndex)
    Beta = solvebeta(X, S, Pi, LeafToSuper, Ridge=Ridge, MaxIter=MaxIter, Tolerance=Tolerance)

    ad.obsm["beta"] = Beta
    ad.uns["bottom_gene_index"] = GeneIndex
    ad.uns["bottom_params"] = {
        "MaxGenes": MaxGenes,
        "Ridge": Ridge,
        "MaxIter": MaxIter,
        "Tolerance": Tolerance,
    }

    return ad
