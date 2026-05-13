from __future__ import annotations

import numpy as np
from skimage.color import rgb2hed


def histfeatures(adata, Image, HiresScale, PixelScale=1.0, Radius=None):
    ad = adata.copy()
    # Create input appropriate for rgb2hed
    Image = np.asarray(Image)
    if Image.dtype != np.uint8:
        Image = np.clip(Image, 0, 255)
        Image = Image.astype(np.uint8)

    HedImage = rgb2hed(Image)
    HImage = HedImage[:, :, 0].astype(np.float32)
    EImage = HedImage[:, :, 1].astype(np.float32)

    # Define threshold for Hematoxylin
    HLow = np.percentile(HImage, 5)
    HHigh = np.percentile(HImage, 95)
    HRange = HHigh - HLow
    HThreshold = HLow + 0.20 * HRange
    ScaleFactors = ad.uns["scalefactors"]
    SpotDiameter = ScaleFactors["spot_diameter_fullres"]

    # Define radius for each spot
    if Radius is None:
        Radius = SpotDiameter * HiresScale * PixelScale * 0.6 / 2
        Radius = round(Radius)
        Radius = int(Radius)
        Radius = max(6, Radius)

    # Line up the pixel coordinates of each spot with the image
    RowPixels = ad.obs["pxl_row_in_fullres"].to_numpy() * HiresScale * PixelScale
    ColPixels = ad.obs["pxl_col_in_fullres"].to_numpy() * HiresScale * PixelScale
    HMean = np.zeros(ad.n_obs, dtype=np.float32)
    EMean = np.zeros(ad.n_obs, dtype=np.float32)
    HCoverage = np.zeros(ad.n_obs, dtype=np.float32)

    ImageHeight = Image.shape[0]
    ImageWidth = Image.shape[1]

    # Square patch is a tighter circle around each spot - potential for size refinement
    for Spot in range(ad.n_obs):
        Row = RowPixels[Spot]
        Col = ColPixels[Spot]
        CenterY = int(round(Row))
        CenterX = int(round(Col))
        XStart = CenterX - Radius
        XEnd = CenterX + Radius
        YStart = CenterY - Radius
        YEnd = CenterY + Radius
        XStart = max(0, XStart)
        YStart = max(0, YStart)
        XEnd = min(ImageWidth, XEnd)
        YEnd = min(ImageHeight, YEnd)
        HPatch = HImage[YStart:YEnd, XStart:XEnd]
        EPatch = EImage[YStart:YEnd, XStart:XEnd]
        HMean[Spot] = HPatch.mean()
        EMean[Spot] = EPatch.mean()
        HCoverage[Spot] = np.mean(HPatch > HThreshold)

    HistFeatures = np.stack([HMean, EMean, HCoverage], axis=1)

    StromaRaw = EMean - HMean
    StromaMean = StromaRaw.mean()
    StromaStd = StromaRaw.std()
    if StromaStd == 0:
        StromaStd = 1
    StromaZ = (StromaRaw - StromaMean) / StromaStd
    StromaScore = 1 / (1 + np.exp(-StromaZ))

    CoverageLow = np.percentile(HCoverage, 5)
    CoverageHigh = np.percentile(HCoverage, 95)
    CoverageRange = CoverageHigh - CoverageLow
    if CoverageRange == 0:
        CoverageRange = 1
    CellScore = (HCoverage - CoverageLow) / CoverageRange
    CellScore = np.clip(CellScore, 0, 1)

    # Histology criteria for anchor spot
    HistFractions = np.zeros((ad.n_obs, 4), dtype=np.float32)
    HistFractions[:, 0] = CellScore / 3
    HistFractions[:, 1] = CellScore / 3
    HistFractions[:, 2] = StromaScore
    HistFractions[:, 3] = CellScore / 3

    RowSums = HistFractions.sum(axis=1, keepdims=True)
    RowSums = np.where(RowSums > 0, RowSums, 1)
    HistFractions = HistFractions / RowSums

    ad.obsm["hist_features"] = HistFeatures.astype(np.float32)
    ad.obsm["histology_fractions"] = HistFractions
    ad.obs["hist_s"] = StromaScore.astype(np.float32)
    ad.obs["hist_c"] = CellScore.astype(np.float32)
    ad.uns["hist_feature_names"] = ["H_mean", "E_mean", "H_coverage"]

    return ad
