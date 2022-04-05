import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters


def LLscorer(estimator, X, _):
    """ Calculates the scores of the GMM vs. original predicted clusters"""
    return np.mean(estimator.score(X))


def cvGMM(zflowDF, maxcluster: int):
    """ Runs CV on GMM model with score and rand score for multiple clusters"""
    X = zflowDF.drop(
        columns=["Cell Type", "pSTAT5", "index", "Time", "Date", "Dose", "Ligand"]
    )  # Creating matrix that will be used in GMM model

    cv = KFold(10, shuffle=True)
    GMM = GaussianMixture(covariance_type="full", tol=1e-6, max_iter=5000)

    scoring = {"LL": LLscorer, "rand": "rand_score"}
    grid = {'n_components': np.arange(1, maxcluster)}
    grid_search = GridSearchCV(GMM, param_grid=grid, scoring=scoring, cv=cv, refit=False, n_jobs=-1)
    grid_search.fit(X, zflowDF["Cell Type"].values)
    results = grid_search.cv_results_

    return pd.DataFrame({"Cluster": results["param_n_components"], "ll_score": results["mean_test_LL"], "rand_score": results["mean_test_rand"]})


def probGMM(zflowDF, n_clusters: int, cellperexp: int):
    """Use the GMM responsibilities matrix to develop means and covariances for each experimental condition.

    NOTE: This method currently assumes there is a constant number of samples per experiment.

    Args:
        zflowDF (pandas.DataFrame): DF w/z-scored epitopes values w/pSTAT5 and celltypes
        n_clusters (int): The number of clusters to run the analysis for.
        cellperexp (int): Amount of cells wanted for GMM for each experiment

    Returns:
        numpy.array: Matrix of data sample numbers across each condition.
        numpy.array: Matrix of means across each condition.
        numpy.array: Tensor of covariance matrices across each condition.
    """
    statDF = zflowDF.drop(columns=["Cell Type", "index", "Time", "Date", "Dose", "Ligand"])  # Creating matrix that includes pSTAT5
    markerDF = statDF.drop(columns=["pSTAT5"])  # Creating matrix that will be used in GMM model, only markers

    # Fit the GMM with the full dataset
    GMM = GaussianMixture(n_components=n_clusters, covariance_type="full", max_iter=5000, verbose=20)
    GMM.fit(markerDF)
    _, log_resp = GMM._estimate_log_prob_resp(markerDF)  # Get the responsibilities
    assert log_resp.shape[0] == zflowDF.shape[0]  # Check shapes

    # Setup storage
    nk = list()
    means = list()
    covariances = list()

    # Loop over separate conditions
    for i in range(0, markerDF.shape[0], cellperexp):
        endi = i + cellperexp
        indDF = statDF.iloc[i:endi]
        resp_ind = log_resp[i:endi, :]
        assert indDF.shape[0] == cellperexp  # Check my indexing
        assert indDF.shape[0] == resp_ind.shape[0]  # Check my indexing

        output = _estimate_gaussian_parameters(indDF.values, np.exp(resp_ind), reg_covar=1e-6, covariance_type="full")
        nk.append(output[0])
        means.append(output[1])
        covariances.append(output[2])

    return np.stack(nk), np.stack(means), np.stack(covariances)
