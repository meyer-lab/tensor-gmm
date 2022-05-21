import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.config import config
import tensorly as tl
import xarray as xa
from sklearn.mixture import GaussianMixture
from jax import value_and_grad, jit

from scipy.optimize import minimize
from tensorly.decomposition import non_negative_parafac
from tensorly.cp_tensor import cp_normalize
from gmm.imports import smallDF

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
config.update("jax_enable_x64", True)


def tensor_decomp(tensor: xa.DataArray, ranknumb: int):
    """Runs tensor decomposition on means tensor."""

    # Need to input the tMeans as numpy tensor
    fac = non_negative_parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]
    fac = cp_normalize(fac)  # Normalizing factors

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))
        # For each dimension in tensor, have a specific ranking for each parameter

    return dfs, fac


def tensor_R2X(tensor: xa.DataArray, maxrank: int):
    """Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1, maxrank + 1)
    varexpl = np.empty(len(rank))

    for i in range(len(rank)):
        _, facinfo = tensor_decomp(tensor, rank[i])
        vTop, vBottom = 0.0, 0.0
        tMask = np.isfinite(tensor)
        vTop += np.sum(np.square(tl.cp_to_tensor(facinfo) * tMask - np.nan_to_num(tensor)))
        # Need to rebuild tensor using factors and weights
        vBottom += np.sum(np.square(np.nan_to_num(tensor)))
        varexpl[i] = 1.0 - vTop / vBottom

    return rank, varexpl


def cp_pt_to_vector(nk: np.ndarray, facinfo: list, factors_pt: list):
    """Converts from factors to a linear vector."""
    vec = np.array([], dtype=float)
    vec = np.append(vec, nk)

    for fac in facinfo:
        vec = np.append(vec, fac.flatten())

    vec = np.append(vec, factors_pt[1].flatten())

    return np.log(vec)


def vector_to_cp_pt(vectorIn, rank: int, shape: tuple, enforceSPD=True):
    """Converts linear vector to factors"""
    vectorIn = jnp.exp(vectorIn)
    rebuildnk = vectorIn[0 : shape[0]]
    vectorIn = vectorIn[shape[0] : :]

    # Shape of tensor for means or precision matrix
    nN = np.cumsum(np.array(shape) * rank)
    nN = np.insert(nN, 0, 0)
    nN = np.append(nN, nN[-1] + shape[1] * shape[1] * rank)

    factors = [jnp.reshape(vectorIn[nN[ii] : nN[ii + 1]], (shape[ii], rank)) for ii in range(len(shape))]
    # Rebuidling factors and ranks

    precSym = vectorIn[nN[-2] : nN[-1]].reshape(shape[1], shape[1], rank)

    if enforceSPD:
        precSym = (precSym + jnp.swapaxes(precSym, 0, 1)) / 2.0  # Enforce symmetry

        # Compute the symmetric polar factor of B. Call it H.
        # Clearly H is itself SPD.
        for ii in range(precSym.shape[2]):
            _, S, V = jnp.linalg.svd(precSym[:, :, ii], full_matrices=False)
            precSymH = V @ S @ V.T
            # get Ahat in the above formula
            precSym.at[:, :, ii].set((precSym[:, :, ii] + precSymH) / 2)

        precSym = (precSym + jnp.swapaxes(precSym, 0, 1)) / 2.0  # Enforce symmetry

    factors_pt = [factors[0], precSym, factors[2], factors[3], factors[4]]

    return rebuildnk, factors, factors_pt


def vector_guess(shape: tuple, rank: int):
    """Predetermines total vector that will be maximized for NK, factors and core"""
    factortotal = np.sum(shape) * rank + shape[1] * shape[1] * rank + shape[0]
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


def minimize_func(zflowTensor: xa.DataArray, rank: int, n_cluster: int, maxiter=20000):
    """Function used to minimize loglikelihood to obtain NK, factors and core of Cp and Pt"""
    times = zflowTensor.coords["Time"]
    doses = zflowTensor.coords["Dose"]
    ligand = zflowTensor.coords["Ligand"]

    clustArray = np.arange(1, n_cluster + 1)
    meanShape = (n_cluster, len(markerslist), len(times), len(doses), len(ligand))
    commonDims = {"Time": times, "Dose": doses, "Ligand": ligand}
    coords = {"Cluster": clustArray, "Markers": markerslist, **commonDims}

    args = (meanShape, rank, zflowTensor.to_numpy())

    tl.set_backend("jax")

    func = jit(value_and_grad(maxloglik_ptnnp), static_argnums=(1, 2))

    x0 = vector_guess(meanShape, rank)
    opt = minimize(func, x0, jac=True, method="L-BFGS-B", args=args, options={"maxls": 200, "iprint": 90, "maxiter": maxiter})

    tl.set_backend("numpy")

    optNK, optCP, optPT = vector_to_cp_pt(opt.x, rank, meanShape)
    optLL = -opt.fun
    optCP = cp_normalize((None, optCP))
    optVec = opt.x

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    CPdf = [pd.DataFrame(optCP.factors[ii], columns=cmpCol, index=coords[key]) for ii, key in enumerate(coords)]

    return optNK, CPdf, optPT, optLL, optVec


def tensorGMM_CV(dataXArray, numFolds, numClusters, numRank):
    """Runs Cross Validation for TensorGMM in order to determine best cluster/rank combo."""
    logLik = 0
    times = dataXArray.coords["Time"]
    doses = dataXArray.coords["Dose"]
    ligands = dataXArray.coords["Ligand"]
    markers = dataXArray.coords["Marker"]

    foldSize = int(np.floor(dataXArray.Cell.size / numFolds))
    meanShape = (numClusters, len(markers), len(times), len(doses), len(ligands))

    # Start generating splits and running model
    for splitNum in np.arange(0, numFolds):
        # Generate Coordinates of test and training data
        trainCoords = np.concatenate((np.arange(0, splitNum * foldSize), np.arange((splitNum + 1) * foldSize, foldSize * numFolds)))
        testCoords = np.arange(splitNum * foldSize, (splitNum + 1) * foldSize)

        # Generate tensors of test and training data
        trainData = dataXArray.to_numpy()[:, trainCoords, :, :, :]
        testData = dataXArray.to_numpy()[:, testCoords, :, :, :]

        # Arrange into Xarrays
        trainXArray = xa.DataArray(
            trainData,
            dims=("Marker", "Cell", "Time", "Dose", "Ligand"),
            coords={"Marker": markers, "Cell": np.arange(0, trainData.shape[1]), "Time": times, "Dose": doses, "Ligand": ligands},
        )
        testXArray = xa.DataArray(
            testData,
            dims=("Marker", "Cell", "Time", "Dose", "Ligand"),
            coords={"Marker": markers, "Cell": np.arange(0, testData.shape[1]), "Time": times, "Dose": doses, "Ligand": ligands},
        )
        # Train
        _, _, _, _, optVec = minimize_func(trainXArray, numRank, numClusters, maxiter=1000)
        # Test
        logLik -= maxloglik_ptnnp(optVec, meanShape, numRank, testXArray.to_numpy())

    return float(logLik)
