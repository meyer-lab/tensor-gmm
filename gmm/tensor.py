import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.config import config
import tensorly as tl
import xarray as xa
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from jax import value_and_grad, jit

from scipy.optimize import minimize
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
config.update("jax_enable_x64", True)


def vector_to_cp_pt(vectorIn, rank: int, shape: tuple, enforceSPD=True):
    """Converts linear vector to factors"""
    vectorIn = jnp.exp(vectorIn)
    rebuildnk = vectorIn[0 : shape[0]]
    vectorIn = vectorIn[shape[0] : :]

    # Shape of tensor for means or precision matrix
    nN = np.cumsum(np.array(shape) * rank)
    nN = np.insert(nN, 0, 0)

    factors = [jnp.reshape(vectorIn[nN[ii] : nN[ii + 1]], (shape[ii], rank)) for ii in range(len(shape))]
    # Rebuidling factors and ranks

    precSym = jnp.zeros((shape[1], shape[1], rank))
    ai, bi = jnp.tril_indices(5)
    pVec = vectorIn[nN[-1] : :].reshape(-1, rank)
    precSym = precSym.at[ai, bi, :].set(pVec)
    precSym = (precSym + jnp.swapaxes(precSym, 0, 1)) / 2.0  # Enforce symmetry

    if enforceSPD:
        # Compute the symmetric polar factor of B. Call it H.
        # Clearly H is itself SPD.
        for ii in range(precSym.shape[2]):
            _, S, V = jnp.linalg.svd(precSym[:, :, ii], full_matrices=False)
            precSymH = V @ S @ V.T
        #     # get Ahat in the above formula
        #     precSym = precSym.at[:, :, ii].set((precSym[:, :, ii] + precSymH) / 2)

        precSym = (precSym + jnp.swapaxes(precSym, 0, 1)) / 2.0  # Enforce symmetry

    factors_pt = [factors[0], precSym, factors[2], factors[3], factors[4]]

    return rebuildnk, factors, factors_pt


def vector_guess(shape: tuple, rank: int):
    """Predetermines total vector that will be maximized for NK, factors and core"""
    factortotal = np.sum(shape) * rank + int(shape[1] * (shape[1] - 1) / 2 + shape[1]) * rank + shape[0]
    return np.random.normal(loc=-1.0, size=factortotal)


def comparingGMM(zflowDF: xa.DataArray, meanFact, tPrecision: np.ndarray, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    loglik = 0.0

    tMeans = tl.cp_to_tensor((None, meanFact))
    X = zflowDF.to_numpy()

    it = np.nditer(tMeans[0, 0, :, :, :], flags=["multi_index", "refs_ok"])
    for _ in it:  # Loop over indices
        i, j, k = it.multi_index

        Xcur = np.transpose(X[:, :, i, j, k])  # Cell Number per experiment x Marker

        if np.all(np.isnan(Xcur)):  # Skip if there's no data
            continue

        gmm = GaussianMixture(n_components=nk.size, covariance_type="full", means_init=tMeans[:, :, i, j, k], weights_init=nk)
        gmm._initialize(Xcur, np.ones((X.shape[1], nk.size)))  # Markers x Clusters
        gmm.precisions_cholesky_ = tPrecision[:, :, :, i, j, k]  # Cluster x Marker x Marker
        loglik += np.sum(gmm.score_samples(Xcur))

    return loglik


@jit
def comparingGMMjax(X, nk, meanFact: list, ptFact):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values
    to determine the max log-likelihood"""
    assert nk.ndim == 1
    n_markers = ptFact[1].shape[0]
    nkl = jnp.log(nk / jnp.sum(nk))

    mp = jnp.einsum("iz,jz,kz,lz,mz,ix,jox,kx,lx,mx->ioklm", *meanFact, *ptFact, optimize="greedy")
    Xp = jnp.einsum("jiklm,nx,jox,kx,lx,mx->inoklm", X, *ptFact, optimize="greedy")
    log_prob = jnp.square(jnp.linalg.norm(Xp - mp[jnp.newaxis, :, :, :, :, :], axis=2))
    log_prob = -0.5 * (n_markers * jnp.log(2 * jnp.pi) + log_prob)

    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    unrav = jnp.reshape(ptFact[1], (-1, ptFact[1].shape[2]))
    unrav = unrav[:: n_markers + 1, :]
    ppp = jnp.einsum("ak,ck,fk,hk,jk->acfhj", ptFact[0], unrav, ptFact[2], ptFact[3], ptFact[4], optimize="greedy")
    log_det = jnp.sum(jnp.log(ppp), 1)

    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    loglik = jnp.sum(jsp.logsumexp(log_prob + log_det[jnp.newaxis, :, :, :, :] + nkl[jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis], axis=1))
    return loglik


def maxloglik_ptnnp(facVector, shape: tuple, rank: int, X):
    """Function used to rebuild tMeans from factors and maximize log-likelihood"""
    parts = vector_to_cp_pt(facVector, rank, shape)
    # Creating function that we want to minimize
    return -comparingGMMjax(X, *parts)


def minimize_func(zflowTensor: xa.DataArray, rank: int, n_cluster: int, maxiter=20000, iprint=-1, x0=None):
    """Function used to minimize loglikelihood to obtain NK, factors and core of Cp and Pt"""
    meanShape = (n_cluster, zflowTensor.shape[0], zflowTensor.shape[2], zflowTensor.shape[3], zflowTensor.shape[4])

    args = (meanShape, rank, zflowTensor.to_numpy())

    func = jit(value_and_grad(maxloglik_ptnnp), static_argnums=(1, 2))

    if x0 is None:
        x0 = vector_guess(meanShape, rank)

    opts = {"maxls": 200, "maxcor": 30, "iprint": iprint, "maxiter": maxiter, "maxfun": 1e6, "ftol": 1e-10}
    opt = minimize(func, x0, jac=True, method="L-BFGS-B", args=args, options=opts)

    optNK, optCP, optPT = vector_to_cp_pt(opt.x, rank, meanShape)
    optLL = -opt.fun
    optCP = cp_normalize((None, optCP))
    optVec = opt.x

    return optNK, optCP, optPT, optLL, optVec


def tensorGMM_CV(X, numFolds: int, numClusters: int, numRank: int, maxiter=20000):
    """Runs Cross Validation for TensorGMM in order to determine best cluster/rank combo."""
    logLik = 0.0
    meanShape = (numClusters, len(markerslist), X.shape[2], X.shape[3], X.shape[4])

    kf = KFold(n_splits=numFolds)
    x0 = None

    # Start generating splits and running model
    for train_index, test_index in kf.split(X[:, :, 0, 0, 0].T):
        # Train
        _, _, _, _, x0 = minimize_func(X[:, train_index, :, :, :], numRank, numClusters, maxiter=maxiter, x0=x0)
        # Test
        test_ll = -maxloglik_ptnnp(x0, meanShape, numRank, X[:, test_index, :, :, :].to_numpy())
        logLik += test_ll

    return float(logLik)
