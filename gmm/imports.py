""" Methods for data import and normalization. """

import numpy as np
import pyarrow.parquet as pq
import xarray as xa


def smallDF(fracCells):
    """Creates DF of specific # of experiments
    Zscores all markers per experiment but pSTAT5 normalized over all experiments"""
    # fracCells = Amount of cells per experiment
    flowDF = importflowDF()
    gVars = ["Ligand", "Dose", "Time"]
    # Columns that should be trasformed
    tCols = ["Foxp3", "CD25", "CD4", "CD45RA"]
    transCols = ["Foxp3", "CD25", "CD4", "CD45RA", "pSTAT5"]

    flowDF["Ligand"] = flowDF["Ligand"] + "-" + flowDF["Valency"].astype(int).astype(str)
    flowDF.drop(columns=["Valency", "index", "Date"], axis=1, inplace=True)

    # Data was measured for CD3/CD8/CD56 was not measured for non-Tregs/Thelpers
    # Also drop columns with missing values
    flowDF = flowDF.dropna(subset=["Foxp3"]).dropna(axis=1)
    flowDF[tCols] = flowDF.groupby(by=gVars)[tCols].transform(lambda x: x / np.std(x))
    for mark in transCols:
        flowDF = flowDF[flowDF[mark] < flowDF[mark].quantile(.995)]
    flowDF = flowDF.groupby(by=gVars).sample(n=fracCells).reset_index(drop=True)
    flowDF["Cell Type"] = flowDF["Cell Type"].apply(celltypetonumb)
    flowDF["pSTAT5"] /= np.std(flowDF["pSTAT5"])
    flowDF.sort_values(by=gVars, inplace=True)
    totalcells = flowDF.shape[0]

    flowDF["Cell"] = np.tile(np.arange(1, fracCells + 1), int(flowDF.shape[0] / fracCells))
    flowDF = flowDF.drop(["Cell Type"], axis=1)
    flowDF = flowDF.set_index(["Cell", "Ligand", "Dose", "Time"]).to_xarray()
    flowDF = flowDF[transCols].to_array(dim="Marker")

    return flowDF, totalcells


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

    monomeric = pq.read_table("/opt/andrew/FlowDataGMM_Mon_NoSub.pq")
    monomeric = monomeric.to_pandas()

    return monomeric
