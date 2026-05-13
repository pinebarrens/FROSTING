from __future__ import annotations

import numpy as np
import pandas as pd

from hierdecon import expandbeta
from hierdecon.markers import SUBTYPES, SUPERTYPES

from .schema import collectresults, readresult, saveresult


def buildgroundtruth(adata, DataFolder):
    from trial_pipeline3 import build_ground_truth_official

    GroundTruth = build_ground_truth_official(adata, janesick_dir=DataFolder)
    return GroundTruth


def exporthierdecon(adata, OutFile, SampleID="10x_human_breast", UseReferenceBeta=True, UseMarkerWeights=True):
    BetaNative = np.asarray(adata.obsm["beta_smooth"], dtype=np.float32)
    LeafNames = list(adata.uns["leaf_names"])
    LeafToSupertype = np.asarray(adata.uns["leaf_to_supertype"], dtype=np.int32)

    if UseReferenceBeta:
        Expanded = expandbeta(
            adata,
            BetaKey="beta_smooth",
            OutputKey="beta_reference",
            UseMarkerWeights=UseMarkerWeights,
        )
        Beta = np.asarray(Expanded.obsm["beta_reference"], dtype=np.float32)
        BetaNames = list(SUBTYPES)
        BetaToSupertype = np.asarray(Expanded.uns["beta_reference_to_supertype"], dtype=np.int32)
    else:
        Beta = BetaNative
        BetaNames = LeafNames
        BetaToSupertype = LeafToSupertype

    ResultFile = saveresult(
        OutFile,
        ObsNames=adata.obs_names.astype(str),
        CellTypes=LeafNames,
        Proportions=BetaNative,
        Pi=adata.obsm["pi"],
        SuperNames=SUPERTYPES,
        Beta=Beta,
        LeafNames=BetaNames,
        LeafToSupertype=BetaToSupertype,
        Method="hierdecon",
        RefTag="refFree",
        SampleID=SampleID,
        Meta={
            "use_reference_beta": UseReferenceBeta,
            "use_marker_weights": UseMarkerWeights,
        },
    )
    return ResultFile


def scorehumanbreast(adata, ResultFile, GroundTruth):
    from benchmarks.eval_benchmarks import score_janesick

    Scores = score_janesick(adata, ResultFile, GroundTruth)
    return Scores


def scorefolder(adata, ResultsFolder, GroundTruth):
    Manifest = collectresults(ResultsFolder)
    ScoreTables = {}

    for _, Row in Manifest.iterrows():
        Scores = scorehumanbreast(adata, Row["path"], GroundTruth)
        for Name, Table in Scores.items():
            if Name not in ScoreTables:
                ScoreTables[Name] = []
            ScoreTables[Name].append(Table)

    Combined = {}
    for Name, Tables in ScoreTables.items():
        Combined[Name] = pd.concat(Tables, ignore_index=True)

    return Manifest, Combined


def rownormalize(Matrix):
    Matrix = np.asarray(Matrix, dtype=np.float64)
    RowSums = Matrix.sum(axis=1, keepdims=True)
    RowSums = np.where(RowSums > 0, RowSums, 1)
    Normalized = Matrix / RowSums
    return Normalized


def meanfractiontables(Manifest, GroundTruth):
    PiRows = []
    BetaRows = []
    Mask = np.asarray(GroundTruth["gt_mask"], dtype=bool)

    GroundTruthPi = rownormalize(GroundTruth["Y_super"])[Mask]
    GroundTruthPiMean = GroundTruthPi.mean(axis=0)
    for CellType, Value in zip(SUPERTYPES, GroundTruthPiMean):
        PiRows.append({"method": "ground_truth", "celltype": CellType, "mean_pi": Value})

    GroundTruthBeta = rownormalize(GroundTruth["Y_subtype"])[Mask]
    GroundTruthBetaMean = GroundTruthBeta.mean(axis=0)
    for Subtype, Value in zip(SUBTYPES, GroundTruthBetaMean):
        BetaRows.append({"method": "ground_truth", "subtype": Subtype, "mean_beta": Value})

    for _, Row in Manifest.iterrows():
        Result = readresult(Row["path"])
        Method = Result["method"]
        if Result["ref_tag"]:
            Method = Method + "@" + Result["ref_tag"]

        Pi = rownormalize(Result["pi"])[Mask]
        PiMean = Pi.mean(axis=0)
        for CellType, Value in zip(Result["super_names"], PiMean):
            PiRows.append({"method": Method, "celltype": CellType, "mean_pi": Value})

        Beta = Result["beta"]
        if Beta.shape[1] == 0:
            continue
        Beta = rownormalize(Beta)[Mask]
        BetaMean = Beta.mean(axis=0)
        for Subtype, Value in zip(Result["leaf_names"], BetaMean):
            BetaRows.append({"method": Method, "subtype": Subtype, "mean_beta": Value})

    PiTable = pd.DataFrame(PiRows).pivot(index="celltype", columns="method", values="mean_pi")
    BetaTable = pd.DataFrame(BetaRows).pivot(index="subtype", columns="method", values="mean_beta")
    PiTable = PiTable.round(3)
    BetaTable = BetaTable.round(3)

    return PiTable, BetaTable


def runtimetable(Manifest):
    Rows = []
    RuntimeKeys = ("runtime_seconds", "train_seconds", "elapsed_seconds", "seconds")

    for _, Row in Manifest.iterrows():
        Result = readresult(Row["path"])
        Meta = Result.get("meta", {})
        Runtime = np.nan
        RuntimeKey = ""
        for Key in RuntimeKeys:
            if Key in Meta:
                Runtime = Meta[Key]
                RuntimeKey = Key
                break

        Method = Result["method"]
        if Result["ref_tag"]:
            Method = Method + "@" + Result["ref_tag"]

        Rows.append({
            "method": Method,
            "runtime_seconds": Runtime,
            "runtime_minutes": Runtime / 60 if pd.notna(Runtime) else np.nan,
            "runtime_key": RuntimeKey,
        })

    Table = pd.DataFrame(Rows)
    Table = Table.sort_values("runtime_seconds", na_position="last")
    return Table


def detectionmetrics(Prediction, Truth, Mask, Names, PresenceThreshold=0.0):
    from sklearn.metrics import average_precision_score, roc_auc_score

    Prediction = np.asarray(Prediction, dtype=np.float64)
    Truth = np.asarray(Truth, dtype=np.float64)
    Mask = np.asarray(Mask, dtype=bool)
    Rows = []

    for Index, Name in enumerate(Names):
        Y = Truth[Mask, Index] > PresenceThreshold
        Score = Prediction[Mask, Index]
        Positive = int(Y.sum())
        Total = int(Y.shape[0])
        if Positive == 0 or Positive == Total:
            Auprc = np.nan
            Auroc = np.nan
        else:
            Auprc = average_precision_score(Y, Score)
            Auroc = roc_auc_score(Y, Score)

        Rows.append({
            "celltype": Name,
            "auprc": Auprc,
            "auroc": Auroc,
            "n_pos": Positive,
            "n_total": Total,
        })

    Table = pd.DataFrame(Rows)
    return Table


def detectiontables(adata, Manifest, GroundTruth):
    TargetObs = list(adata.obs_names.astype(str))
    Mask = np.asarray(GroundTruth["gt_mask"], dtype=bool)
    TruthPi = rownormalize(GroundTruth["Y_super"])
    TruthSubtype = pd.DataFrame(
        rownormalize(GroundTruth["Y_subtype"]),
        index=TargetObs,
        columns=SUBTYPES,
    )
    PiTables = []
    BetaTables = []

    for _, Row in Manifest.iterrows():
        Result = readresult(Row["path"])
        Method = Result["method"]
        if Result["ref_tag"]:
            Method = Method + "@" + Result["ref_tag"]

        ResultObs = [str(Name) for Name in Result["obs_names"]]
        Pi = pd.DataFrame(
            rownormalize(Result["pi"]),
            index=ResultObs,
            columns=Result["super_names"],
        ).reindex(TargetObs)
        PiMask = Mask & Pi.notna().any(axis=1).to_numpy()
        PiTable = detectionmetrics(Pi.fillna(0).to_numpy(), TruthPi, PiMask, Result["super_names"])
        PiTable.insert(0, "method", Method)
        PiTables.append(PiTable)

        Beta = Result["beta"]
        if Beta.shape[1] == 0:
            continue
        LeafNames = list(Result["leaf_names"])
        PredBeta = pd.DataFrame(rownormalize(Beta), index=ResultObs, columns=LeafNames).reindex(TargetObs)
        TrueBeta = pd.DataFrame(0.0, index=TargetObs, columns=LeafNames)
        for LeafName in LeafNames:
            if LeafName in TruthSubtype.columns:
                TrueBeta[LeafName] = TruthSubtype[LeafName]
        TrueBeta = rownormalize(TrueBeta.to_numpy())
        BetaMask = Mask & PredBeta.notna().any(axis=1).to_numpy()
        BetaTable = detectionmetrics(PredBeta.fillna(0).to_numpy(), TrueBeta, BetaMask, LeafNames)
        BetaTable.insert(0, "method", Method)
        BetaTables.append(BetaTable)

    PiDetection = pd.concat(PiTables, ignore_index=True)
    BetaDetection = pd.concat(BetaTables, ignore_index=True)
    return PiDetection, BetaDetection
