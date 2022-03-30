import pandas as pd
import numpy as np
import tensorly as tl
import xarray as xa
from sklearn.mixture import GaussianMixture

from tensorly.decomposition import non_negative_parafac, parafac
from tensorly.cp_tensor import cp_normalize

markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]


def get_conditions(df, X):
    """ Provides all unique conditions for a specific time, ligand, and concentration. """
    assert df.shape[0] % X.shape[0] == 0
    colNames = ["Ligand", "Time", "Dose"]
    cellsPerCondition = int(df.shape[0] / X.shape[0])
    conditions = df[colNames].iloc[::cellsPerCondition]
    conditions = conditions.set_index(colNames)
    return conditions


def tensor_means(meansDF: pd.DataFrame, means: np.ndarray):
    """ Turn the GMM mean results into a tensor form. """
    # Add valency to ligand name

    conditions = get_conditions(meansDF, means)

    xd = xa.DataArray(means, dims=["Conditions", "Cluster", "Marker"])
    xd = xd.assign_coords(Cluster=np.arange(1, means.shape[1] + 1))
    xd = xd.assign_coords(Marker=markerslist, Conditions=conditions.index)
    return xd.unstack("Conditions")


def tensor_covar(meansDF: pd.DataFrame, covar: np.ndarray):
    """ Turn the GMM covariance results into tensor form. """
    # Add valency to ligand name

    conditions = get_conditions(meansDF, covar)

    # covar is conditions x clusters x markers x markers
    xd = xa.DataArray(covar, dims=["Conditions", "Cluster", "Marker1", "Marker2"])
    xd = xd.assign_coords(Cluster=np.arange(1, covar.shape[1] + 1))
    xd = xd.assign_coords(Marker1=markerslist, Marker2=markerslist, Conditions=conditions.index)
    return xd.unstack("Conditions")


def tensor_decomp(tensor: xa.DataArray, ranknumb: int, tensortype):
    """ Runs tensor decomposition on means tensor. """

    if tensortype == "NNparafac":
        fac = non_negative_parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)
    else:
        fac = parafac(np.nan_to_num(tensor.to_numpy()), mask=np.isfinite(tensor.to_numpy()), rank=ranknumb)

    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]
    fac = cp_normalize(fac)

    dfs = []
    for ii, dd in enumerate(tensor.dims):
        dfs.append(pd.DataFrame(fac.factors[ii], columns=cmpCol, index=tensor.coords[dd]))

    return dfs, fac


def tensor_R2X(tensor, maxrank, tensortype):
    """ Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1, maxrank)
    varexpl = np.empty(len(rank))

    for i in range(len(rank)):
        _, facinfo = tensor_decomp(tensor, rank[i], tensortype)
        vTop, vBottom = 0.0, 0.0
        tMask = np.isfinite(tensor)
        vTop += np.sum(np.square(tl.cp_to_tensor(facinfo) * tMask - np.nan_to_num(tensor)))
        vBottom += np.sum(np.square(np.nan_to_num(tensor)))
        varexpl[i] = 1.0 - vTop / vBottom

    return rank, varexpl


def meanCP_to_DF(factorinfo_NNP, meansDF):
    """Converts output of factor decomposition into a dataframe"""
    newTens = tl.cp_to_tensor(factorinfo_NNP)
    shapeTens = np.shape(newTens)

    # Cluster x Marker x Ligand x Time x Concentration

    markDF = []

    for i in range(shapeTens[0]):
        for j, tim in enumerate(meansDF["Time"].unique()):
            for l, dose in enumerate(meansDF["Dose"].unique()):
                for k, ligand in enumerate(meansDF["Ligand"].unique()):
                    ave_mark = newTens[i, :, k, j, l]
                    markDF.append([ligand, dose, tim, i + 1, ave_mark[0], ave_mark[1], ave_mark[2], ave_mark[3], ave_mark[4]])

    markDF = pd.DataFrame(markDF, columns=["Ligand", "Dose", "Time", "Cluster", "Foxp3", "CD25", "CD4", "CD45RA", "pSTAT5"])

    return markDF


def covarTens_to_DF(meansDF, covar, markerslist):
    """Converts tensor of convariances into a matrix """
    covarDF = meansDF.copy()

    for i, mark in enumerate(markerslist):
        for j, marker in enumerate(markerslist):
            markers_covar = covar[:, :, i, j]
            covarDF[mark + "-" + marker] = markers_covar.flatten(order="F")

    return covarDF


def comparingGMM(zflowDF, meansDF, covarDF, nk: np.ndarray):
    """Obtains the GMM means, convariances and NK values along with zflowDF mean marker values"""
    assert nk.ndim == 1
    nk /= np.sum(nk)
    conditions = zflowDF.groupby(["Time", "Dose", "Ligand"])

    loglik = 0.0

    for name, cond_cells in conditions:
        # Means of GMM
        flow_mean = meansDF.loc[(meansDF["Time"] == name[0]) & (meansDF["Dose"] == name[1]) & (meansDF["Ligand"] == name[2])]
        flow_mean = flow_mean.sort_values(by=['Cluster'])
        flow_mean = flow_mean[markerslist]
        flow_covar = covarDF.loc[(covarDF["Time"] == name[0]) & (covarDF["Dose"] == name[1]) & (covarDF["Ligand"] == name[2])]
        print(flow_covar.size)
        print(name)
        assert flow_mean.shape[0] == flow_covar.shape[0] # Rows are clusters
        assert flow_mean.size > 0
        assert flow_covar.size > 0

        X = cond_cells[markerslist].to_numpy()

        gmm = GaussianMixture(n_components=nk.size, max_iter=2, covariance_type="full", means_init=flow_mean.to_numpy(), weights_init=nk)
        gmm.fit(X) # One iteration to make the covariance matrix
        # Still need to plugin the covariance matrix

        loglik += np.sum(gmm.score_samples(X))
        print(loglik)

    return
