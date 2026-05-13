from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def saveresult(OutFile, ObsNames, CellTypes, Proportions, Pi, SuperNames, Method, RefTag="refFree", SampleID="", Beta=None, LeafNames=None, LeafToSupertype=None, Meta=None):
    OutFile = Path(OutFile)
    OutFile.parent.mkdir(parents=True, exist_ok=True)

    Proportions = np.asarray(Proportions, dtype=np.float32)
    Pi = np.asarray(Pi, dtype=np.float32)
    if Beta is None:
        Beta = np.zeros((Proportions.shape[0], 0), dtype=np.float32)
    else:
        Beta = np.asarray(Beta, dtype=np.float32)

    if LeafNames is None:
        LeafNames = []
    if LeafToSupertype is None:
        LeafToSupertype = np.zeros(0, dtype=np.int32)
    else:
        LeafToSupertype = np.asarray(LeafToSupertype, dtype=np.int32)

    np.savez_compressed(
        OutFile,
        obs_names=np.array(list(ObsNames), dtype=object),
        cell_types=np.array(list(CellTypes), dtype=object),
        proportions=Proportions,
        pi=Pi,
        super_names=np.array(list(SuperNames), dtype=object),
        beta=Beta,
        leaf_names=np.array(list(LeafNames), dtype=object),
        leaf_to_supertype=LeafToSupertype,
        method=str(Method),
        ref_tag=str(RefTag),
        sample_id=str(SampleID),
        meta=np.array([dict(Meta or {})], dtype=object),
    )
    return OutFile


def readresult(ResultFile):
    ResultFile = Path(ResultFile)
    Data = np.load(ResultFile, allow_pickle=True)
    Result = {}
    for Key in Data.files:
        Value = Data[Key]
        if Key in ("method", "ref_tag", "sample_id"):
            Value = str(Value.tolist())
        elif Key == "meta":
            Value = dict(Value.item())
        Result[Key] = Value

    return Result


def collectresults(ResultsFolder):
    Rows = []
    ResultsFolder = Path(ResultsFolder)
    for ResultFile in sorted(ResultsFolder.rglob("*.npz")):
        Result = readresult(ResultFile)
        Row = {
            "path": str(ResultFile),
            "method": Result["method"],
            "ref_tag": Result["ref_tag"],
            "sample_id": Result["sample_id"],
            "n_spots": int(Result["pi"].shape[0]),
            "has_beta": bool(Result["beta"].shape[1] > 0),
        }
        Rows.append(Row)

    Results = pd.DataFrame(Rows)
    return Results
