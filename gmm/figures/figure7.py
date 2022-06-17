"""
This creates Figure 7.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xa 
from .common import subplotLabel, getSetup
from gmm.scImport import geneNNMF #, import_thompson_drug, normalizeGenes, mu_sigma, gene_filter
from gmm.tensor import minimize_func, gen_points_GMM
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sklearn.model_selection import KFold
from ..tensor import maxloglik_ptnnp



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

    jax_run()

    num = 10; fac = 2 
    drugXA = ThompsonDrugXA(numCells = num, rank = fac, maxit = 10)
    
    rank = 2; clust = 2
    maximizedNK, optCP, optPTfactors, _, _, preNormOptCP = minimize_func(drugXA, rank = rank, n_cluster= clust)

    ax[0].bar(np.arange(1, maximizedNK.size + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    cmpCol = [f"Fac. {i}" for i in np.arange(1, fac + 1)]
    rankCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    clustArray = [f"Clust. {i}" for i in np.arange(1, clust + 1)]
    coords = {"Cluster": clustArray, "Factor": cmpCol, "Drug": drugXA.coords["Drug"]}
    maximizedFactors = [pd.DataFrame(optCP.factors[ii], columns=rankCol, index=coords[key]) for ii, key in enumerate(coords)]

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 1], cmap="Greens")

    ranknumb = np.arange(3, 5)
    n_cluster = np.arange(2, 4)

    maxloglikDFcv = pd.DataFrame()
    for i in range(len(ranknumb)):
        row = pd.DataFrame()
        row["Rank"] = ["Rank:" + str(ranknumb[i])]
        for j in range(len(n_cluster)):
            loglik = geneGMM_CV(drugXA, numFolds=3, numClusters=n_cluster[j], numRank=ranknumb[i])
            row["Cluster:" + str(n_cluster[j])] = loglik

        maxloglikDFcv = pd.concat([maxloglikDFcv, row])

    maxloglikDFcv = maxloglikDFcv.set_index("Rank")
    sns.heatmap(data=maxloglikDFcv, ax=ax[4])
    ax[4].set(title="Cross Validation")

    return f


def ThompsonDrugXA(numCells: int, rank: int, maxit: int):
    finalDF = pd.read_csv('/opt/andrew/FilteredDrugs_Offset1.3.csv')
    finalDF.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    finalDF = finalDF.groupby(by="Drug").sample(n=numCells).reset_index(drop=True)

    geneComponent, geneFactors = geneNNMF(finalDF, k=rank, verbose=0, maxiteration= maxit)
    cmpCol = [f"Fac. {i}" for i in np.arange(1, rank + 1)]
        
    PopAlignDF = pd.DataFrame(data=geneFactors, columns=cmpCol)
    PopAlignDF["Drug"] = finalDF["Drug"].values
    PopAlignDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(PopAlignDF.shape[0] / numCells))
        
    PopAlignXA = PopAlignDF.set_index(["Cell", "Drug"]).to_xarray()
    PopAlignXA = PopAlignXA[cmpCol].to_array(dim="Factor")

    npPopAlign = np.reshape(PopAlignXA.to_numpy(), (PopAlignXA.shape[0],PopAlignXA.shape[1], -1, 1, 1 ))
    PopAlignXA = xa.DataArray(npPopAlign, dims=("Factor", "Cell", "Drug", "Throwaway 1", "Throwaway 2"),
                coords={"Factor": cmpCol, "Cell": np.arange(1, numCells + 1), "Drug": finalDF["Drug"].unique(), 
                "Throwaway 1": ["Throwaway"], "Throwaway 2": ["Throwaway"]})


    return PopAlignXA

def gene_import(offset):
    genesDF, geneNames = import_thompson_drug()
    genesN = normalizeGenes(genesDF, geneNames)
    filteredGeneDF, logmean, logstd = mu_sigma(genesDF, geneNames)
    finalDF, filtered_index = gene_filter(filteredGeneDF, logmean, logstd, offset_value = offset)
    return finalDF

def jax_run():
    # Global flag to set a specific platform, must be used at startup
    jax.config.update('jax_platform_name', 'cpu')
    x = jnp.square(2)
    print(repr(x.device_buffer.device()))  # CpuDevice(id=0)


def geneGMM_CV(X, numFolds: int, numClusters: int, numRank: int, maxiter=300):
    """Runs Cross Validation for TensorGMM in order to determine best cluster/rank combo."""
    logLik = 0.0
    meanShape = (numClusters, X.shape[0], X.shape[2], X.shape[3], X.shape[4])

    kf = KFold(n_splits=numFolds)
    x0 = None

    # Start generating splits and running model
    for train_index, test_index in kf.split(X[:, :, 0, 0, 0].T):
        # Train
        _, _, _, _, x0, _ = minimize_func(X[:, train_index, :, :, :], numRank, numClusters, maxiter=maxiter, x0=x0)
        # Test
        test_ll = -maxloglik_ptnnp(x0, meanShape, numRank, X[:, test_index, :, :, :].to_numpy())
        logLik += test_ll

    return float(logLik)


