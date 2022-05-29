"""
This creates Figure 6.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.scImport import import_thompson_drug, geneNNMF

# from gmm import barcodes,features, meta


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

    cellperexp = 20
    genesDF = import_thompson_drug(cellperexp)
    print(genesDF)
    geneComponent, geneFactors = geneNNMF(genesDF, k = 20, verbose=0)

    return f


