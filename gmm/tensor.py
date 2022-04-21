import pandas as pd
import numpy as np
import tensorly as tl
import xarray as xa
from sklearn.mixture import GaussianMixture

from tensorly.decomposition import non_negative_parafac, parafac
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """ Runs tensor decomposition on means tensor. """

    if tensortype == "NNparafac":
        fac = non_negative_parafac(
            np.nan_to_num(
                tensor.to_numpy()), mask=np.isfinite(
                tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(
            np.nan_to_num(
                tensor.to_numpy()), mask=np.isfinite(
                tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]
    fac = cp_normalize(fac)

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))

    return dfs, fac


def tensor_R2X(tensor: xa.DataArray, maxrank: int, tensortype):
    """ Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1, maxrank)
    varexpl = np.empty(len(rank))

    for i in range(len(rank)):
        _, facinfo = tensor_decomp(tensor, rank[i], tensortype)
        vTop, vBottom = 0.0, 0.0
        tMask = np.isfinite(tensor)
        vTop += np.sum(np.square(tl.cp_to_tensor(facinfo) * tMask - np.nan_to_num(tensor)))
        vBottom += np.sum(np.square(np.nan_to_num(tensor)))
        varexpl[i] = 1.0 - vTop / vBottom

    return rank, varexpl

def cp_to_vector(facinfo,zflowTensor: xa.DataArray,maxcluster: int):

    tensor = tl.cp_to_tensor(facinfo) 
    times = zflowTensor.coords["Time"]
    doses = zflowTensor.coords["Dose"]
    ligand = zflowTensor.coords["Ligand"]
    clustArray = np.arange(1, maxcluster + 1)
    commonSize = (len(times), len(doses), len(ligand))
    commonDims = {"Time": times, "Dose": doses, "Ligand": ligand}

    facMeans = xa.DataArray(tensor, coords={"Cluster": clustArray, "Markers": markerslist, **commonDims})
    facMeans = facMeans.stack(z=("Ligand", "Dose", "Time","Markers","Cluster")).to_numpy()

    return facMeans



def vector_to_cp(vectorIn: np.ndarray, rank: int, tMeans: xa.DataArray):

    nN = np.cumsum(np.array(tMeans.shape)*rank)
    A = np.reshape(vectorIn[:nN[0]], (tMeans.shape[0], rank))
    B = np.reshape(vectorIn[nN[0]:nN[1]], (tMeans.shape[1], rank))
    C = np.reshape(vectorIn[nN[1]:nN[2]], (tMeans.shape[2], rank))
    D = np.reshape(vectorIn[nN[2]:nN[3]], (tMeans.shape[3], rank))
    E = np.reshape(vectorIn[nN[3]:nN[4]], (tMeans.shape[4], rank))

    tFac = tl.cp_tensor.CPTensor((None, [A, B, C, D, E]))

    return tFac



def comparingGMM(zflowDF: xa.DataArray, tMeans: xa.DataArray, tPrecision: xa.DataArray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = 0.0

    tMeans = tMeans.to_numpy()
    tPrecision = tPrecision.to_numpy()
    X = zflowDF.to_numpy()

    it = np.nditer(tMeans[0, 0, :, :, :], flags=['multi_index', 'refs_ok'])
    for _ in it:  # Loop over indices
        i, j, k = it.multi_index

        Xcur = np.transpose(X[:, :, i, j, k])

        if np.all(np.isnan(X)):  # Skip if there's no data
            continue

        gmm = GaussianMixture(n_components=nk.size, covariance_type="full", means_init=tMeans[:, :, i, j, k],
                              weights_init=nk)
        gmm._initialize(Xcur, np.ones((X.shape[1], nk.size)))
        gmm.precisions_cholesky_ = tPrecision[:, :, :, i, j, k]
        loglik += np.sum(gmm.score_samples(Xcur))

    return loglik


def leastsquaresguess(nk, tMeans):
    nkCommon = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))  # nk is shared across conditions
    tMeans_vector = tMeans.values.flatten()
    return np.append(nkCommon, tMeans_vector)


def maxloglik(nk_tMeans_input, maxcluster, zflowDF, tMeans, tPrecision):
    nk_guess = nk_tMeans_input[0:maxcluster]

    tGuessMeans = tMeans.copy()
    assert len(nk_guess) == maxcluster
    tMeans_input = nk_tMeans_input[maxcluster:]
    assert len(tMeans_input) == len(tMeans.values.flatten())

    tGuessMeans.values = np.reshape(tMeans_input, tMeans.shape)

    ll = comparingGMM(zflowDF, tGuessMeans, tPrecision, nk_guess)
    # print(ll)

    return -ll
