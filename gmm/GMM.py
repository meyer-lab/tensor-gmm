import pandas as pd
import numpy as np
import xarray as xa
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from .tensor import markerslist


def LLscorer(estimator, X, _):
    """ Calculates the scores of the GMM vs. original predicted clusters"""
    return np.mean(estimator.score(X))


def cvGMM(zflowDF: xa.DataArray, maxcluster: int, true_types):
    """ Runs CV on GMM model with score and rand score for multiple clusters"""
    zflowDF = zflowDF.copy()
    zflowDF = zflowDF.drop_sel({"Marker": "pSTAT5"})
    X = np.reshape(zflowDF.to_numpy(), (-1, zflowDF.shape[0]))  # Creating matrix that will be used in GMM model

    cv = KFold(10, shuffle=True)
    GMM = GaussianMixture(covariance_type="full", tol=1e-6, max_iter=5000)

    scoring = {"LL": LLscorer, "rand": "rand_score"}
    grid = {'n_components': np.arange(1, maxcluster)}
    grid_search = GridSearchCV(GMM, param_grid=grid, scoring=scoring, cv=cv, refit=False, n_jobs=-1)
    grid_search.fit(X, true_types.values.flatten())
    results = grid_search.cv_results_

    return pd.DataFrame({"Cluster": results["param_n_components"],
                         "ll_score": results["mean_test_LL"],
                         "rand_score": results["mean_test_rand"]})


def probGMM(zflowDF, n_clusters: int):
    """Use the GMM responsibilities matrix to develop means and covariances for each experimental condition."""
    # Fit the GMM with the full dataset
    # Creating matrix that will be used in GMM model
    X = np.reshape(zflowDF.to_numpy(), (-1, zflowDF.shape[0]))
    GMM = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        max_iter=5000,
        verbose=20)
    GMM.fit(X)
    _, log_resp = GMM._estimate_log_prob_resp(X)  # Get the responsibilities

    # Reshape into tensor form for easy indexing
    log_resp = np.exp(log_resp)
    log_resp = np.reshape(log_resp, (-1, zflowDF.shape[1], zflowDF.shape[2], zflowDF.shape[3], zflowDF.shape[4]))

    times = zflowDF.coords["Time"]
    doses = zflowDF.coords["Dose"]
    ligand = zflowDF.coords["Ligand"]
    clustArray = np.arange(1, n_clusters + 1)
    commonSize = (len(times), len(doses), len(ligand))
    commonDims = {"Time": times, "Dose": doses, "Ligand": ligand}

    # Setup storage
    nk = xa.DataArray(np.full((n_clusters, *commonSize), np.nan),
                      coords={"Cluster": clustArray, **commonDims})
    means = xa.DataArray(np.full((n_clusters, len(markerslist), *commonSize), np.nan),
                         coords={"Cluster": clustArray, "Markers": markerslist, **commonDims})
    covariances = xa.DataArray(np.full((n_clusters, len(markerslist), len(markerslist), *commonSize), np.nan),
                               coords={"Cluster": clustArray, "Marker1": markerslist, "Marker2": markerslist, **commonDims})

    it = np.nditer(nk[0, :, :, :], flags=['multi_index', 'refs_ok'])
    for _ in it:
        i, j, k = it.multi_index

        output = _estimate_gaussian_parameters(np.transpose(zflowDF[:, :, i, j, k].values),
                                               np.transpose(log_resp[:, :, i, j, k]), 1e-6, "full")

        nk[:, i, j, k] = output[0]
        means[:, :, i, j, k] = output[1]
        covariances[:, :, :, i, j, k] = output[2]

    return nk, means, covariances
