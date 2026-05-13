from __future__ import annotations

import numpy as np

from .markers import SUBTYPES, SUBTYPETOSUPER, SUPERTYPES


def subtypeparents():
    ParentIndex = []
    for Subtype in SUBTYPES:
        Supertype = SUBTYPETOSUPER[Subtype]
        SuperIndex = SUPERTYPES.index(Supertype)
        ParentIndex.append(SuperIndex)

    ParentIndex = np.array(ParentIndex, dtype=np.int32)
    return ParentIndex


def childindices(Supertype):
    Children = []
    for SubtypeIndex, Subtype in enumerate(SUBTYPES):
        Parent = SUBTYPETOSUPER[Subtype]
        if Parent == Supertype:
            Children.append(SubtypeIndex)

    Children = np.array(Children, dtype=np.int32)
    return Children


def splitresidual(Mass, Weights):
    RowSums = Weights.sum(axis=1, keepdims=True)
    ZeroRows = RowSums[:, 0] == 0
    if np.any(ZeroRows):
        Weights = Weights.copy()
        Weights[ZeroRows, :] = 1
        RowSums = Weights.sum(axis=1, keepdims=True)

    Fractions = Weights / RowSums
    Split = Mass[:, None] * Fractions
    return Split


def expandbeta(adata, BetaKey="beta_smooth", OutputKey="beta_subtype", UseMarkerWeights=True):
    ad = adata.copy()
    Beta = np.asarray(ad.obsm[BetaKey], dtype=np.float64)
    LeafNames = list(ad.uns["leaf_names"])
    LeafToSuper = np.asarray(ad.uns["leaf_to_supertype"], dtype=np.int32)
    if UseMarkerWeights:
        SubScores = np.asarray(ad.obsm["marker_scores_sub"], dtype=np.float64)
    else:
        SubScores = None
    Expanded = np.zeros((ad.n_obs, len(SUBTYPES)), dtype=np.float64)

    for LeafIndex, LeafName in enumerate(LeafNames):
        LeafMass = Beta[:, LeafIndex]
        if LeafName in SUBTYPES:
            SubtypeIndex = SUBTYPES.index(LeafName)
            Expanded[:, SubtypeIndex] = Expanded[:, SubtypeIndex] + LeafMass
            continue

        SuperIndex = LeafToSuper[LeafIndex]
        Supertype = SUPERTYPES[SuperIndex]
        Children = childindices(Supertype)
        if len(Children) == 0:
            continue

        if UseMarkerWeights:
            Weights = SubScores[:, Children]
        else:
            Weights = np.ones((ad.n_obs, len(Children)), dtype=np.float64)

        Split = splitresidual(LeafMass, Weights)
        Expanded[:, Children] = Expanded[:, Children] + Split

    ad.obsm[OutputKey] = Expanded.astype(np.float32)
    ad.uns[OutputKey + "_names"] = list(SUBTYPES)
    ad.uns[OutputKey + "_to_supertype"] = subtypeparents()
    ad.uns[OutputKey + "_use_marker_weights"] = bool(UseMarkerWeights)
    return ad
