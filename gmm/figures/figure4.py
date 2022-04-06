"""
This creates Figure 4.
"""
import numpy as np
from scipy.optimize import least_squares, Bounds
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, comparingGMM, maxloglik


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
    maxcluster = 4
    nk, tMeans, tCovar = probGMM(zflowDF, maxcluster)

    # tensor_decomp(tensor means, rank, type of decomposition):
    # [DF,tensorfactors/weights] creates DF of factors for different
    # conditions and output of decomposition
    rank = 5
    factors_NNP, factorinfo_NNP = tensor_decomp(tMeans, rank, "NNparafac")

    nkCommon = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))  # nk is shared across conditions
    nkGuess = nkCommon
    optimized = least_squares(maxloglik, nkGuess, args=(zflowDF, tMeans, tCovar))

    # minbound = [1e-2,1e-2,1e-2,1e-2]
    # maxbound = [100,100,100,100]
    # nkBounds = Bounds(minbound,maxbound)
    # optimized = least_squares(maxloglik, nkGuess, bounds = (minbound,maxbound), args = (zflowDF,tMeans,tCovar))

    print("NK Guess:", nkGuess)
    print("Optimized NK:", optimized.x)

    return f
