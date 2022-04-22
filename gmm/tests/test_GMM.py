"""
Test the data import.
"""
import pandas as pd
from ..imports import smallDF
from ..GMM import cvGMM, probGMM
from ..tensor import tensor_decomp, cp_to_vector, vector_to_cp


def test_cvGMM():
    """Stub test."""
    dataTwo, other = smallDF(50)
    gmmDF = cvGMM(dataTwo, 4, other[1])
    assert isinstance(gmmDF, pd.DataFrame)


def test_GMMprob():
    """Test that we can construct a covariance matrix including pSTAT5."""
    cellperexp = 50
    dataTwo, _ = smallDF(cellperexp)
    maxcluster = 4
    nk, means, covari = probGMM(dataTwo, maxcluster)


def test_TensorVector():
    """Test that we can go from Cp to vector, and from vector to Cp without changing values."""
    cellperexp = 50
    dataTwo, _ = smallDF(cellperexp)
    maxcluster = 4
    nk, means, covari = probGMM(dataTwo, maxcluster)
    rank = 2
    factors_NNP, facinfo = tensor_decomp(means, rank, "NNparafac")
    cpVector = cp_to_vector(facinfo, dataTwo)
    vectorFac = vector_to_cp(cpVector, rank, means)

    assert vectorFac.factors[0].flatten().all() == factors_NNP[0].values.flatten().all()
