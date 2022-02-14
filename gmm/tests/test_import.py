'''
Test the data import.
'''
import pytest
from ..imports import smallDF


def test_import():
    """ Stub test. """
    data = smallDF(10)
    print(data)
