"""
This creates Figure 4.
"""
import numpy as np
import tensorly as tl
from scipy.optimize import minimize
from jax.config import config
from jax import value_and_grad
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.GMM import probGMM
from gmm.tensor import tensor_decomp, cp_to_vector, maxloglik
from tensorly.decomposition import partial_tucker

config.update("jax_enable_x64", True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 20
    zflowTensor, _ = smallDF(cellperexp)

    # probGM(Xarray, max cluster): Xarray [nk, means, covar] while using estimation gaussian parameters
    # tMeans[Cluster, Marker, Time, Dose, Ligand]
    maxcluster = 3
    nk, tMeans, tCovar = probGMM(zflowTensor, maxcluster)

    # tensor_decomp(tensor means, rank, type of decomposition):
    # [DF,tensorfactors/weights] creates DF of factors for different
    # conditions and output of decomposition

    ranknumb = 2
    _, facInfo = tensor_decomp(tMeans, ranknumb)

    nkValues = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))
    cpVector = cp_to_vector(facInfo)
    cpVector = np.concatenate((nkValues, cpVector))
    args = (facInfo, tCovar, zflowTensor)

    tl.set_backend("jax")

    func = value_and_grad(maxloglik)

    opt = minimize(func, cpVector, jac=True, method="L-BFGS-B", args=args, options={"iprint": 50})

    tl.set_backend("numpy")

    return f
