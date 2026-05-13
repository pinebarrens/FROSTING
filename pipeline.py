from __future__ import annotations

import scanpy as sc

from .anchors import anchorphase
from .bottom import bottomphase
from .cnv import cnvscore
from .expand import expandbeta
from .hist import histfeatures
from .io import loadhires, loadspatial
from .markers import markerscore
from .qc import qcfilter
from .smooth import smoothphase
from .top import toppartition


def runpipeline(SpatialData, TissuePositions, ScaleFactors, SpatialFolder, HiresImage, GtfFile, SampleID="sample", ReferenceBeta=False, UseMarkerWeights=True):
    adata = loadspatial(SpatialData, TissuePositions, ScaleFactors, SpatialFolder, SampleID=SampleID)
    adata = qcfilter(adata, MinCounts=500, MinGenes=200, MaxMito=15)
    Image, HiresScale = loadhires(HiresImage, ScaleFactors)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_cpm"] = adata.X.copy()

    adata = markerscore(adata, MaxRank=1500)
    adata = histfeatures(adata, Image, HiresScale, PixelScale=1.0, Radius=None)
    adata = cnvscore(
        adata,
        GtfFile,
        TumorQuantile=0.65,
        WindowSize=100,
        MaxSubclones=10,
        MinClusterSpots=15,
    )
    adata = toppartition(adata, MarkerWeight=1.0, CnvWeight=0.5, TumorIndex=0)
    adata = anchorphase(adata, MinAnchors=15, Dominance=1.5, ParentPi=0.5, ParentHist=0.3)
    adata = bottomphase(adata, MaxGenes=2000, Ridge=1e-3, MaxIter=300, Tolerance=1e-6)
    adata = smoothphase(adata, Lambda=0.05, Iterations=200, Neighbors=6)

    if ReferenceBeta:
        adata = expandbeta(
            adata,
            BetaKey="beta_smooth",
            OutputKey="beta_reference",
            UseMarkerWeights=UseMarkerWeights,
        )

    return adata
