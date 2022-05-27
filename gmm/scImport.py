import numpy as np
from sklearn.decomposition import NMF
from scipy.io import mmread


def import_thompson_drug():
    x = mmread("/opt/andrew/drugscreen.mtx")
    x = x.astype(np.int16)
    return x.todense()


def NNMF(X, k=14, verbose=0):
    model = NMF(n_components=k, verbose=verbose, max_iter=2000, tol=1e-6)
    W = model.fit_transform(X)
    return model.components_, W