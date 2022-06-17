"""
This creates Figure 7.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xa 
from .common import subplotLabel, getSetup
from gmm.scImport import geneNNMF #, import_thompson_drug, normalizeGenes, mu_sigma, gene_filter
import matplotlib.pyplot as plt


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))
    
    # genesDF, geneNames = import_thompson_drug()
    # geneNames = genesDF.columns[0:-2].tolist()
    # genesN = normalizeGenes(genesDF, geneNames)
    # filteredGeneDF, logmean, logstd = mu_sigma(genesDF, geneNames)
    # finalDF, filtered_index = gene_filter(filteredGeneDF, logmean, logstd, offset_value = 1.3)

    drugXA = ThompsonDrugXA(numCells = 290, rank = 2, maxit = 10)
    print(drugXA)

    return f


def ThompsonDrugXA(numCells: int, rank: int, maxit: int):
    finalDF = pd.read_csv('/opt/andrew/FilteredDrugs_Offset1.3.csv')
    finalDF.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    finalDF = finalDF.groupby(by="Drug").sample(n=numCells).reset_index(drop=True)

    geneComponent, geneFactors = geneNNMF(finalDF, k=rank, verbose=0, maxiteration= maxit)
    cmpCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
        
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

