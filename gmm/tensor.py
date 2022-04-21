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


def tensor_R2X(tensor, maxrank, tensortype):
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


def comparingGMM(zflowDF: xa.DataArray, tMeans: xa.DataArray, tPrecision: xa.DataArray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = logsumexp(np.log(nk)) * tMeans.shape[2] * tMeans.shape[3] * tMeans.shape[4]
    normTerm = n_features * np.log(2 * np.pi)

    tMeans = tMeans.to_numpy()
    tPrecision = tPrecision.to_numpy()
    X = zflowDF.to_numpy()

    Xp = np.einsum("jiklm,njoklm->nioklm", X, tPrecision)
    mp = np.einsum("ijklm,ijoklm->ioklm", tMeans, tPrecision)
    diff = np.square(Xp - mp[:, np.newaxis, :, :, :, :])
    n_features = mp.shape[1]

    it = np.nditer(tMeans[0, 0, :, :, :], flags=['multi_index', 'refs_ok'])
    for _ in it:  # Loop over indices
        i, j, k = it.multi_index

        precisions_chol = tPrecision[:, :, :, i, j, k]

        # The determinant of the precision matrix from the Cholesky decomposition
        # corresponds to the negative half of the determinant of the full precision matrix.
        # In short: det(precision_chol) = - det(precision) / 2
        log_det = np.sum(np.log(precisions_chol.reshape(mp.shape[0], -1)[:, :: n_features + 1]), 1)

        log_prob = np.sum(diff[:, :, :, i, j, k], axis=2).T

        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        loglik += logsumexp(-0.5 * (normTerm + log_prob) + log_det)

    return loglik


def leastsquaresguess(nk, tMeans):
    nkCommon = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))  # nk is shared across conditions
    tMeans_vector = tMeans.values.flatten()
    return np.append(nkCommon, tMeans_vector)


def maxloglik(nk_tMeans_input, maxcluster, zflowDF, tMeans, tCovar):
    nk_guess = nk_tMeans_input[0:maxcluster]

    tGuessMeans = tMeans.copy()
    assert len(nk_guess) == maxcluster
    tMeans_input = nk_tMeans_input[maxcluster:]
    assert len(tMeans_input) == len(tMeans.values.flatten())

    tGuessMeans.values = np.reshape(tMeans_input, tMeans.shape)

    ll = comparingGMM(zflowDF, tGuessMeans, tCovar, nk_guess)
    print(ll)

    return -ll
