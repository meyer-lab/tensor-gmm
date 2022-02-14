
import numpy as np
import pandas as pd
from scipy import stats


def smallDF(fracCells):
    # FracCells = Amount of cells per experiment 
    flowDF = importflowDF()
    flowDFg = flowDF.groupby(by=["Time", "Dose", "Date", "Ligand"])

    # Columns that should be trasformed
    transCols = ["Foxp3", "CD25", "CD3", "CD8", "CD56", "CD45RA"]
    flowDF[transCols] = flowDFg[transCols].transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
    return flowDF


def importflowDF():
    """Downloads all conditions, surface markers and cell types.
    Cells are labeled via Thelper, None, Treg, CD8 or NK """
    return pd.read_feather('/opt/andrew/FlowDataGMM_Mon_Labeled.ftr')
