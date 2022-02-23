"""
Test the data import.
"""
import pytest
from ..imports import smallDF


def test_import():
    """Stub test.""" 
    data, _, _, _, _, _ = smallDF(10)
    dataTwo, _, _, _, _, _ = smallDF(20)
    assert 2 * data.shape[0] == dataTwo.shape[0]
