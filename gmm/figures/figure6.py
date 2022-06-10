"""
This creates Figure 6.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.scImport import import_thompson_drug, geneNNMF, normalizeGenes, mu_sigma
from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# from gmm import barcodes,features, meta


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

    cellperexp = 200
    
    genesDF, geneNames = import_thompson_drug(cellperexp)
    print(genesDF)
    # genesDF.to_csv('output.csv')
    # genesDF = pd.read_csv('output.csv')
    # genesDF.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    # geneNames = genesDF.columns[0:-2].tolist()
    # geneDrugNames = genesDF.columns.tolist()
    mu_sigma(genesDF)[0]

    genesN = normalizeGenes(genesDF, geneNames)

    std = fg[geneNames].std(axis=0)
    mean = fg[geneNames].mean(axis=0)
    cv = np.divide(std, mean, out=np.zeros_like(std),where=mean!=0)

    nonzero = fg[geneNames][fg[geneNames] == 0].count()

    # geneComponent, geneFactors = geneNNMF(genesDF, k = 20, verbose=0)

    return f


