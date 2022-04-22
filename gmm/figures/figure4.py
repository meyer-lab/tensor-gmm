"""
This creates Figure 4.
"""
import numpy as np
from scipy.optimize import minimize
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, cp_to_vector, markerslist, maxloglik


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 20
    zflowTensor, _ = smallDF(cellperexp)

    # probGM(DF,maximum cluster,cellsperexperiment): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 2
    nk, tMeans, tCovar = probGMM(zflowTensor, maxcluster)

    # tensor_decomp(tensor means, rank, type of decomposition):
    # [DF,tensorfactors/weights] creates DF of factors for different
    # conditions and output of decomposition

    rank = 2
    _, facInfo = tensor_decomp(tMeans, rank, "NNparafac")

    nkValues = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))
    cpVector = cp_to_vector(facInfo)

    optimized = minimize(
        maxloglik,
        cpVector,
        method="Nelder-Mead",
        args=(
            facInfo,
            tCovar,
            nkValues,
            zflowTensor),
        options={
            "disp": True,
            "maxiter": 10})

    print("Optimized Parameters:", optimized)

    return f
