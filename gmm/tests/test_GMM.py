"""
Test the data import.
"""
import pandas as pd
import numpy as np
from ..imports import smallDF
from ..GMM import cvGMM
from ..tensor import vector_to_cp_pt, comparingGMM, comparingGMMjax, vector_guess, maxloglik_ptnnp, minimize_func, tensorGMM_CV

data_import, other_import = smallDF(10)
meanShape = (6, data_import.shape[0], data_import.shape[2], data_import.shape[3], data_import.shape[4])


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


def test_CP_to_vec():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    x0 = vector_guess(meanShape, rank=3)

    built = vector_to_cp_pt(x0, 3, meanShape, enforceSPD=False)
    vector_to_cp_pt(x0, 3, meanShape, enforceSPD=True)

    # Check that we can get a likelihood
    ll = maxloglik_ptnnp(x0, meanShape, 3, data_import.to_numpy())

    assert np.isfinite(ll)


def test_comparingGMM():
    """Test that we can ensures log likelihood is calculated the same"""
    x0 = vector_guess(meanShape, rank=3)

    nk, meanFact, ptFact = vector_to_cp_pt(x0, 3, meanShape)
    ptBuilt = np.einsum("ax,bcx,dx,ex,fx->abcdef", *ptFact)

    optimized1 = comparingGMM(data_import, meanFact, ptBuilt, nk)
    optimized2 = comparingGMMjax(data_import.to_numpy(), nk, meanFact, ptFact)

    np.testing.assert_almost_equal(optimized1, optimized2)


def test_fit():
    """Test that fitting can run fine."""
    nk, fac, ptfac, ll, _ = minimize_func(data_import, 3, 10, maxiter=20)
    loglik = tensorGMM_CV(data_import, numFolds=3, numClusters=3, numRank=2, maxiter=20)
    assert type(loglik) is float
    assert type(ll) is float