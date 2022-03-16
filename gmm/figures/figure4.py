"""
This creates Figure 3.
"""
import numpy as np
import pandas as pd
import seaborn as sns

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM, meanmarkerDF
from ..tensor import tensor_decomp, tensor_means


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 30), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    #smallDF(Amount of cells wanted per experiment
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)
    maxcluster = 5
    nk, means, covar = probGMM(zflowDF, maxcluster, cellperexp)
    meansDF, markerslist = meanmarkerDF(zflowDF,cellperexp,means,nk,maxcluster)

    tMeans = tensor_means(meansDF,markerslist)
    
    rank = 5

    factors_NNP = tensor_decomp(tMeans, rank, "partialT")
 
    for i in range(len(factors_NNP)):
        sns.heatmap(data=factors_NNP[i],ax=ax[i])

    return f
