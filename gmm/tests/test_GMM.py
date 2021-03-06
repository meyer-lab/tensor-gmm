"""
Test the data import.
"""
import pandas as pd
import numpy as np
import xarray as xa
import math
from ..imports import smallDF
from ..GMM import cvGMM
from ..scImport import import_thompson_drug, ThompsonDrugXA
from ..tensor import vector_to_cp_pt, comparingGMM, comparingGMMjax, vector_guess, maxloglik_ptnnp, minimize_func, tensorGMM_CV, covFactor_to_precisions, comparingGMMjax_NK

data_import, other_import = smallDF(10)
meanShape = (6, data_import.shape[0], data_import.shape[2], data_import.shape[3], data_import.shape[4])
dataPA_import, _, _ = ThompsonDrugXA(numCells=10, rank=10, maxit=20, runFacts=True)


def test_cvGMM():
    """Stub test."""
    gmmDF = cvGMM(data_import, 4, other_import[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_import():
    """Stub test."""
    dataTwo, _ = smallDF(data_import.shape[1] * 2)
    assert data_import.shape[0] == dataTwo.shape[0]
    assert 2 * data_import.shape[1] == dataTwo.shape[1]
    assert data_import.shape[2] == dataTwo.shape[2]
    assert data_import.shape[3] == dataTwo.shape[3]
    assert data_import.shape[4] == dataTwo.shape[4]


def test_cov_to_prec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    x0 = vector_guess(meanShape, rank=3)

    _, _, covFac = vector_to_cp_pt(x0, 3, meanShape)
    precBuild = covFactor_to_precisions(covFac)

    assert np.all(np.isfinite(precBuild))


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    x0 = vector_guess(meanShape, rank=3)

    built = vector_to_cp_pt(x0, 3, meanShape)

    # Check that we can get a likelihood
    ll = maxloglik_ptnnp(x0, meanShape, 3, data_import.to_numpy())

    assert np.isfinite(ll)


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    x0 = vector_guess(meanShape, rank=3)

    nk, meanFact, covFac = vector_to_cp_pt(x0, 3, meanShape)
    precBuild = covFactor_to_precisions(covFac)

    optimized1 = comparingGMM(data_import, meanFact, precBuild, nk)
    optimized2 = comparingGMMjax(data_import.to_numpy(), nk, meanFact, precBuild)
    np.testing.assert_allclose(optimized1, optimized2, rtol=1e-5)


def test_independence():
    """Test that conditions can be separately evaluated as expected."""
    x0 = vector_guess(meanShape, rank=3)
    data_numpy = data_import.to_numpy()

    nk, meanFact, covFac = vector_to_cp_pt(x0, 3, meanShape)
    precBuild = covFactor_to_precisions(covFac)

    ll1 = comparingGMM(data_import, meanFact, precBuild, nk)
    ll2 = comparingGMMjax(data_numpy, nk, meanFact, precBuild)
    np.testing.assert_allclose(ll1, ll2, rtol=1e-5)

    # Test that cells are independent
    ll3 = comparingGMMjax(data_numpy[:, :5, :, :, :], nk, meanFact, precBuild)
    ll3 += comparingGMMjax(data_numpy[:, 5:, :, :, :], nk, meanFact, precBuild)
    np.testing.assert_allclose(ll2, ll3, rtol=1e-5)

    # Test that ligands are independent
    # meanFactOne = deepcopy(meanFact)
    # meanFactOne[4] = meanFact[4][:5, :]
    # ptFactOne = deepcopy(ptFact)
    # ptFactOne[4] = ptFact[4][:5, :]
    # ll4 = comparingGMMjax(data_numpy[:, :, :, :, :5], nk, meanFactOne, ptFactOne)
    # meanFactTwo = deepcopy(meanFact)
    # meanFactTwo[4] = meanFact[4][5:, :]
    # ptFactTwo = deepcopy(ptFact)
    # ptFactTwo[4] = ptFact[4][5:, :]
    # ll4 += comparingGMMjax(data_numpy[:, :, :, :, 5:], nk, meanFactTwo, ptFactTwo)
    # np.testing.assert_allclose(ll2, ll4, rtol=1e-5)


def test_fit():
    """Test that fitting can run fine."""
    nk, fac, ptfac, ll, _, _ = minimize_func(data_import, 3, 10, maxiter=20, verbose=False)
    loglik = tensorGMM_CV(data_import, numFolds=3, numClusters=3, numRank=2, maxiter=20)
    assert isinstance(loglik, float)
    assert isinstance(ll, float)


def test_import_PopAlign():
    """Stub test."""
    dataPA_two, _, _ = ThompsonDrugXA(numCells=20, rank=20, maxit=20, runFacts=True)
    assert 2 * dataPA_import.shape[0] == dataPA_two.shape[0]
    assert 2 * dataPA_import.shape[1] == dataPA_two.shape[1]
    assert dataPA_import.shape[2] == dataPA_two.shape[2]
    assert dataPA_import.shape[3] == dataPA_two.shape[3]
    assert dataPA_import.shape[4] == dataPA_two.shape[4]


def test_finite_data():
    """Test that all values in tensor has no NaN"""

    assert np.isfinite(data_import.to_numpy()).all()
    assert np.isfinite(dataPA_import.to_numpy()).all()


def test_cov_fit():
    """Test that tensor-GMM method recreates covariance of data accurately"""
    cov = [[0.5, 0], [0, 2]]
    samples = np.transpose(np.random.multivariate_normal([3, 1], cov, 1000)).reshape((2, 1000, 1, 1, 1))
    samples = xa.DataArray(samples, dims=("Dim", "Point", "Throwaway 1", "Throwaway 2", "Throwaway 3"), coords={"Dim": ["X", "Y"], "Point": np.arange(0, 1000), "Throwaway 1": [1], "Throwaway 2": [1], "Throwaway 3": [1]})
    _, _, optPT, _, _, _ = minimize_func(samples, rank=6, n_cluster=1, maxiter=2000, verbose=False)
    cholCov = covFactor_to_precisions(optPT, returnCov=True)
    cholCov = np.squeeze(cholCov[:, :, :, 0, 0, 0])
    covR = cholCov @ cholCov.T

    assert math.isclose(cov[0][0], covR[0][0], abs_tol=0.3)
    assert math.isclose(cov[1][0], covR[1][0], abs_tol=0.2)
    assert math.isclose(cov[0][1], covR[0][1], abs_tol=0.2)
    assert math.isclose(cov[1][1], covR[1][1], abs_tol=0.3)


def test_loglikelihood_NK():
    """Testing to see if loglilihood is a number"""
    cluster = 6
    rank = 3
    markers = 5
    conditions = 4

    # Think data isn't organized correctly.
    X = np.random.rand(markers, 100, conditions)
    nkFact = np.random.rand(cluster, rank)
    meanFact = [np.random.rand(cluster, rank), np.random.rand(markers, rank), np.random.rand(conditions, rank)]
    precBuild = np.random.rand(cluster, markers, markers, conditions)

    ll = comparingGMMjax_NK(X, nkFact, meanFact, precBuild)
    assert np.isfinite(ll)
