"""
This creates Figure 7.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.scImport import import_thompson_drug, geneNNMF, normalizeGenes, mu_sigma, gene_filter
import matplotlib.pyplot as plt


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))
    
    genesDF, geneNames = import_thompson_drug()
    # genesDF.to_csv('output_final.csv')
    # genesDF = pd.read_csv('output_total.csv')
    # genesDF.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    # geneNames = genesDF.columns[0:-2].tolist()

    genesN = normalizeGenes(genesDF, geneNames)

    filteredGeneDF, logmean, logstd = mu_sigma(genesDF, geneNames)
    finalDF, filtered_index = gene_filter(filteredGeneDF, logmean, logstd, offset_value = 1.3)

    ax[0].scatter(logmean,logstd)
    ax[1].scatter(logmean[filtered_index],logstd[filtered_index])

    geneComponent, geneFactors = geneNNMF(finalDF, k=20, verbose=0, maxiteration=2000)


    return f


