from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .bottom import projectparents


def spatedges(adata, Neighbors=6):
    # Creates spatial graph 
    Coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    NumberSpots = Coordinates.shape[0]
    if NumberSpots < 2:
        Edges = np.zeros((0, 2), dtype=np.int64)
        return Edges

    Neighbors = min(Neighbors, NumberSpots - 1)
    Tree = cKDTree(Coordinates)
    Distances, Indices = Tree.query(Coordinates, k=Neighbors + 1)
    NeighborIndex = Indices[:, 1:]

    EdgeList = []
    for Spot in range(NumberSpots):
        for Neighbor in NeighborIndex[Spot]:
            Edge = [Spot, Neighbor]
            Edge.sort()
            EdgeList.append(Edge)

    UniqueEdges = []
    SeenEdges = set()
    for Edge in EdgeList:
        EdgeTuple = tuple(Edge)
        if EdgeTuple in SeenEdges:
            continue
        SeenEdges.add(EdgeTuple)
        UniqueEdges.append(Edge)

    Edges = np.array(UniqueEdges, dtype=np.int64)
    return Edges


def tvsmooth(Beta, Pi, LeafToSuper, Edges, Lambda=0.05, Iterations=200, Tolerance=1e-5):
    Beta = np.asarray(Beta, dtype=np.float64)
    Pi = np.asarray(Pi, dtype=np.float64)
    LeafToSuper = np.asarray(LeafToSuper, dtype=np.int32)
    Edges = np.asarray(Edges, dtype=np.int64)
    if Edges.shape[0] == 0:
        Smoothed = projectparents(Beta, Pi, LeafToSuper)
        return Smoothed.astype(np.float32)

    NumberSpots = Beta.shape[0]
    Source = Edges[:, 0]
    Target = Edges[:, 1]
    Degrees = np.zeros(NumberSpots)
    for Edge in range(Edges.shape[0]):
        Degrees[Source[Edge]] = Degrees[Source[Edge]] + 1
        Degrees[Target[Edge]] = Degrees[Target[Edge]] + 1
    MaxDegree = Degrees.max()
    GraphNorm = 2 * MaxDegree
    if GraphNorm == 0:
        Smoothed = projectparents(Beta, Pi, LeafToSuper)
        return Smoothed.astype(np.float32)

    Step = 1 / np.sqrt(GraphNorm)
    X = Beta.copy()
    XBar = X.copy()
    Z = np.zeros((Edges.shape[0], Beta.shape[1]), dtype=np.float64)

    for Iteration in range(Iterations):
        Difference = XBar[Source, :] - XBar[Target, :]
        NewZ = Z + Step * Difference
        NewZ = np.clip(NewZ, -Lambda, Lambda)
        # Compute the backward difference
        Backward = np.zeros(X.shape, dtype=np.float64)
        np.add.at(Backward, Source, NewZ)
        np.add.at(Backward, Target, -NewZ)
        # Compute the candidate for the new values
        Candidate = X - Step * Backward
        Candidate = Candidate + Step * Beta
        Candidate = Candidate / (1 + Step)
        # Project the candidate onto the simplex so the sum doesn't violate parent constraints 
        NewX = projectparents(Candidate, Pi, LeafToSuper)
        XBar = 2 * NewX - X
        Change = np.linalg.norm(NewX - X)
        Denominator = np.linalg.norm(X)
        Denominator = max(Denominator, 1e-12)
        RelativeChange = Change / Denominator

        X = NewX
        Z = NewZ

        if RelativeChange < Tolerance:
            break

    return X.astype(np.float32)


def smoothphase(adata, Lambda=0.05, Iterations=200, Neighbors=6):
    ad = adata.copy()
    Beta = np.asarray(ad.obsm["beta"])
    Pi = np.asarray(ad.obsm["pi"])
    LeafToSuper = np.asarray(ad.uns["leaf_to_supertype"])
    Edges = spatedges(ad, Neighbors=Neighbors)

    BetaSmooth = tvsmooth(Beta, Pi, LeafToSuper, Edges, Lambda=Lambda, Iterations=Iterations)

    ad.obsm["beta_smooth"] = BetaSmooth
    ad.uns["smooth_edges"] = Edges
    ad.uns["smooth_params"] = {
        "Lambda": Lambda,
        "Iterations": Iterations,
        "Neighbors": Neighbors,
    }

    return ad
