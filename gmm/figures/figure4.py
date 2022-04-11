"""
This creates Figure 4.
"""
import numpy as np
import xarray as xa
from scipy.optimize import least_squares
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, maxloglik, leastsquaresguess


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 20
    zflowDF, _ = smallDF(cellperexp)

    # probGM(DF,maximum cluster,cellsperexperiment): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 2
    nk, tMeans, tCovar = probGMM(zflowDF, maxcluster)

    # tensor_decomp(tensor means, rank, type of decomposition):
    # [DF,tensorfactors/weights] creates DF of factors for different
    # conditions and output of decomposition
    rank = 2
    # factors_NNP, factorinfo_NNP = tensor_decomp(tMeans, rank, "NNparafac")

    nk_tMeans_guess = leastsquaresguess(nk, tMeans, maxcluster)

    optimized = least_squares(maxloglik, nk_tMeans_guess, max_nfev=1, ftol=1e-2, args=(maxcluster, zflowDF, tMeans, tCovar))

    print("Optimized Parameters:", optimized)

    return f
