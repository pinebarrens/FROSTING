from __future__ import annotations

import argparse
from pathlib import Path

from hierdecon import runpipeline


def parseargs():
    Parser = argparse.ArgumentParser(description="Run HierDecon on a Visium sample.")
    Parser.add_argument("--spatial-data", required=True)
    Parser.add_argument("--tissue-positions", required=True)
    Parser.add_argument("--scale-factors", required=True)
    Parser.add_argument("--spatial-folder", required=True)
    Parser.add_argument("--hires-image", required=True)
    Parser.add_argument("--gtf-file", required=True)
    Parser.add_argument("--sample-id", default="sample")
    Parser.add_argument("--reference-beta", action="store_true")
    Args = Parser.parse_args()
    return Args


def main() -> int:
    Args = parseargs()
    SpatialData = Path(Args.spatial_data)
    TissuePositions = Path(Args.tissue_positions)
    ScaleFactors = Path(Args.scale_factors)
    SpatialFolder = Path(Args.spatial_folder)
    HiresImage = Path(Args.hires_image)
    GtfFile = Path(Args.gtf_file)

    print("Running HierDecon")
    adata = runpipeline(
        SpatialData,
        TissuePositions,
        ScaleFactors,
        SpatialFolder,
        HiresImage,
        GtfFile,
        SampleID=Args.sample_id,
        ReferenceBeta=Args.reference_beta,
        UseMarkerWeights=True,
    )
    print("Spots after QC:", adata.n_obs)
    print("Genes:", adata.n_vars)
    print("CNV subclones:", len(adata.uns["cnv_subclones"]))
    print("Leaves:", len(adata.uns["leaf_names"]))
    if Args.reference_beta:
        print("Reference beta columns:", len(adata.uns["beta_reference_names"]))

    print("Pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
