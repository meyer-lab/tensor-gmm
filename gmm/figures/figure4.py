"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorly as tl
from scipy.optimize import minimize, Bounds
from jax.config import config
from jax import value_and_grad
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.GMM import probGMM
from gmm.tensor import tensor_decomp, minimize_func, vector_guess
from tensorly.cp_tensor import cp_normalize


config.update("jax_enable_x64", True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 100
    zflowTensor, _ = smallDF(cellperexp)

    # probGM(Xarray, max cluster): Xarray [nk, means, covar] while using estimation gaussian parameters
    # tMeans[Cluster, Marker, Time, Dose, Ligand]
    maxcluster = 6
    ranknumb = 3
    vectorGuess = vector_guess(zflowTensor, ranknumb, maxcluster)

    _, tMeans, _ = probGMM(zflowTensor, maxcluster)
    _, facInfo = tensor_decomp(tMeans, ranknumb)

    maximizedNK, maximizedFactors, ptNewCore = minimize_func(vectorGuess, facInfo, zflowTensor, tMeans)

    ax[0].bar(np.arange(1, maxcluster + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    for i in range(0, len(facInfo.shape)):
        heatmap = sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 1])

    return f
