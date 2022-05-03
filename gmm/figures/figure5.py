"""
This creates Figure 5.
"""
import numpy as np
import tensorly as tl
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from jax.config import config
from jax import value_and_grad
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, cp_to_vector, maxloglik, tensorcovar_decomp
from tensorly.decomposition import partial_tucker
from tensorly.tucker_tensor import tucker_to_tensor, tucker_to_vec


config.update("jax_enable_x64", True)



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 20
    zflowTensor, _ = smallDF(cellperexp)

   # probGM(Xarray, max cluster): Xarray [nk, means, covar] while using estimation gaussian parameters
    # tMeans[Cluster, Marker, Time, Dose, Ligand]
    maxcluster = 3
    nk, tMeans, tPrecision = probGMM(zflowTensor, maxcluster)

    # tensorcovar_decomp(precision, rank, nk):
    # [DF,core tensor, tensorfactors] creates DF of factors for different
    # conditions and output of decomposition

    ranknumb = 2
    covarFactorsDF, ptCore, ptFactors = tensorcovar_decomp(tPrecision, ranknumb, nk)

    for i in range(0, len(nk.dims)):
        heatmap = sns.heatmap(data= covarFactorsDF[i], ax=ax[i], cmap="Blues")

    # a = tucker_to_vec(ptCore,ptFactors)
    # b = tucker_to_tensor(ptCore,ptFactors)
    # print(a)
    # print(b)


    return f
