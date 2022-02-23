"""
Test the data import.
"""
from statistics import covariance
import pytest
import pandas as pd
from ..imports import smallDF
from ..GMM import cvGMM, probGMM


def test_import():
    """Stub test."""
    dataTwo, _, _, _, _, _ = smallDF(50)
    gmmDF = cvGMM(dataTwo, 4)
    assert isinstance(gmmDF, pd.DataFrame)


def test_GMMprob():
    """Test that we can construct a covariance matrix including pSTAT5."""
    cellperexp = 50
    dataTwo, _, _, _, _, _= smallDF(cellperexp)
    nk,means,covariance = probGMM(dataTwo, 4, cellperexp)
