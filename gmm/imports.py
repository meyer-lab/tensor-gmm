""" Methods for data import and normalization. """

import numpy as np
import pandas as pd


def smallDF(fracCells):
    # FracCells = Amount of cells per experiment
    flowDF = importflowDF()
    gVars = ["Time", "Dose", "Date", "Ligand"]
    # Columns that should be trasformed
    transCols = ["Foxp3", "CD25", "CD45RA", "CD4"]

    # Data was measured for CD3/CD8/CD56 was not measured for non-Tregs/Thelpers
    # Also drop columns with missing values
    flowDF = flowDF.dropna(subset=["Foxp3"]).dropna(axis=1)
    experimentcells = flowDF.groupby(by=gVars).size()
    flowDF[transCols] = flowDF.groupby(by=gVars)[transCols].transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
    flowDF = flowDF.groupby(by=gVars).sample(n=fracCells).reset_index(drop=True)
    flowDF["Cell Type"] = flowDF["Cell Type"].apply(celltypetonumb)
    concDF = flowDF["Dose"].values
    ligDF = flowDF["Ligand"].values
    valDF = flowDF["Valency"].values
    timeDF = flowDF["Time"].values
    flowDF = flowDF.drop(columns=["Valency", "index", "Time", "Date", "Dose", "Ligand"])

    return flowDF, experimentcells, concDF, ligDF, valDF, timeDF


def celltypetonumb(typ):
    """Changing cell types to a number"""
    if typ == "None":
        return 1
    elif typ == "Treg":
        return 2
    else:  # Thelper
        return 3


def importflowDF():
    """Downloads all conditions, surface markers and cell types.
    Cells are labeled via Thelper, None, Treg, CD8 or NK"""
    return pd.read_feather("/opt/andrew/FlowDataGMM_Mon_Labeled.ftr")
