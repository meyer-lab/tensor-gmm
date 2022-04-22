import pandas as pd
import numpy as np
import tensorly as tl
import xarray as xa
from scipy.special import logsumexp

from tensorly.decomposition import non_negative_parafac, parafac
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """ Runs tensor decomposition on means tensor. """

    if tensortype == "NNparafac":
        fac = non_negative_parafac(
            np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(
            np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

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


def cp_to_vector(facinfo: tl.cp_tensor.CPTensor):
    """ Converts from factors to a linear vector. """
    vec = []

    for fac in facinfo.factors:
        vec = np.append(vec, fac.flatten())

    return vec


def vector_to_cp(vectorIn: np.ndarray, rank: int, shape: tuple):
    """Converts linear vector to factors"""
    nN = np.cumsum(np.array(shape) * rank)
    nN = np.insert(nN, 0, 0)

    factors = [np.reshape(vectorIn[nN[ii]:nN[ii + 1]], (shape[ii], rank)) for ii in range(len(shape))]
    return tl.cp_tensor.CPTensor((None, factors))


def comparingGMM(zflowDF: xa.DataArray, tMeans: np.ndarray, tPrecision: xa.DataArray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = logsumexp(np.log(nk)) * tMeans.shape[2] * tMeans.shape[3] * tMeans.shape[4]

    Xp = np.einsum("jiklm,njoklm->nioklm", zflowDF, tPrecision)
    mp = np.einsum("ijklm,ijoklm->ioklm", tMeans, tPrecision)
    diff = np.square(Xp - mp[:, np.newaxis, :, :, :, :])
    n_features = mp.shape[1]
    diff_sum = -0.5 * (n_features * np.log(2 * np.pi) + np.sum(diff, axis=2))
    log_prob = np.swapaxes(diff_sum, 0, 1)

    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    prec_numpy = tPrecision.to_numpy()
    prec_numpy = prec_numpy.reshape(prec_numpy.shape[0], -1, prec_numpy.shape[3], prec_numpy.shape[4], prec_numpy.shape[5])
    prec_numpy = np.log(prec_numpy)

    log_det = np.sum(prec_numpy[:, :: n_features + 1, :, :, :], 1)

    ll = logsumexp(log_prob + log_det[np.newaxis, :, :, :, :], axis=(0, 1))
    return loglik + np.sum(ll)


def leastsquaresguess(nk, tMeans):
    nkCommon = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))  # nk is shared across conditions
    tMeans_vector = tMeans.values.flatten()
    return np.append(nkCommon, tMeans_vector)


def maxloglik(facVector: np.ndarray, facInfo: tl.cp_tensor.CPTensor, tPrecision: xa.DataArray, nk: np.ndarray, zflowTensor: xa.DataArray):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    factorsguess = vector_to_cp(facVector, facInfo.rank, facInfo.shape)
    rebuildMeans = tl.cp_to_tensor(factorsguess)

    ll = comparingGMM(zflowTensor, rebuildMeans, tPrecision, nk)
    return -ll
