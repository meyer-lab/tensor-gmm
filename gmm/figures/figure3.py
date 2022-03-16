"""
This creates Figure 3.
"""
import seaborn as sns

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM, meanmarkerDF
from ..tensor import tensor_decomp, tensor_means, tensor_covar


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 30), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)
    maxcluster = 5
    nk, means, covar = probGMM(zflowDF, maxcluster, cellperexp)
    meansDF, markerslist = meanmarkerDF(zflowDF, cellperexp, means, nk, maxcluster)

    tMeans = tensor_means(meansDF, markerslist)
    tCovar = tensor_covar(meansDF, markerslist, covar)

    rank = 5

    factors_NNP = tensor_decomp(tMeans, rank, "NNparafac")

    for i in range(len(factors_NNP)):
        sns.heatmap(data=factors_NNP[i], ax=ax[i])

    return f
