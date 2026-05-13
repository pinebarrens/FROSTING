"""Top-level supertype proportions."""

from __future__ import annotations

import numpy as np


def toppartition(adata, MarkerWeight=1.0, CnvWeight=0.5, TumorIndex=0, MaxIter=100, Tolerance=1e-10):
    ad = adata.copy()

    # Get marker scores and CNV
    MarkerScores = np.asarray(ad.obsm["marker_scores_super"], dtype=np.float64)
    CnvPhi = ad.obs["cnv_phi"].to_numpy(dtype=np.float64)

    NumberSpots = MarkerScores.shape[0]
    NumberSupertypes = MarkerScores.shape[1]

    # Matrix combines marker scores and CNV weights
    D = np.ones((NumberSpots, NumberSupertypes), dtype=np.float64)
    D = D * MarkerWeight
    D[:, TumorIndex] = D[:, TumorIndex] + CnvWeight

    # Linear target for each spot
    B = MarkerScores * MarkerWeight
    B[:, TumorIndex] = B[:, TumorIndex] + CnvWeight * CnvPhi

    # Defines lower and upper bounds
    Low = -B.max(axis=1)
    High = (D - B).max(axis=1) + 1e-6

    # Optimizes by solving a quadratic program through bisection search
    for Step in range(MaxIter):
        Middle = (Low + High) / 2
        RawPi = B + Middle[:, None]
        RawPi = RawPi / D
        RawPi = np.maximum(RawPi, 0)
        RowSums = RawPi.sum(axis=1)
        TooSmall = RowSums < 1
        Low = np.where(TooSmall, Middle, Low)
        High = np.where(TooSmall, High, Middle)

        Width = High - Low
        BiggestWidth = Width.max()
        if BiggestWidth < Tolerance:
            break

    Theta = (Low + High) / 2
    Pi = B + Theta[:, None]
    Pi = Pi / D
    Pi = np.maximum(Pi, 0)

    # Renormalize to ensure row sums to 1
    RowSums = Pi.sum(axis=1, keepdims=True)
    RowSums = np.where(RowSums > 0, RowSums, 1)
    Pi = Pi / RowSums

    ad.obsm["pi"] = Pi.astype(np.float32)
    ad.uns["top_params"] = {
        "MarkerWeight": MarkerWeight,
        "CnvWeight": CnvWeight,
        "TumorIndex": TumorIndex,
    }

    return ad
