import enum
from inspect import getinnerframes
import numpy as np
import pandas as pd
import csv
import xarray as xa
from sklearn.decomposition import NMF
from scipy.io import mmread
from scipy.stats import linregress
import tensorly as tl


def import_thompson_drug():
    """Imports cell readings from scRNA-seq data from PBMCs from PopAlign paper"
    -Description of each file-
    drugScreeen : str Path to a sparse matrix
    barcodes : str Path to a .tsv 10X barcodes file
    metafile : str Path to a metadata file. Must contains `cell_barcodes` and `sample_id` columns
    genes : str Path to a .tsv 10X gene file"""

    metafile = pd.read_csv("gmm/data/meta.csv")  # Cell barcodes, sample id of treatment and sample number (33482,3)
    drugScreen = mmread("/opt/andrew/drugscreen.mtx").toarray()  # Sparse matrix of each cell/genes (32738,33482)-(Genes,Cell)
    drugScreen = drugScreen.astype(np.int32)
    barcodes = np.array([row[0] for row in csv.reader(open("gmm/data/barcodes.tsv"), delimiter="\t")])  # Cell barcodes(33482)
    genes = np.array([row[1].upper() for row in csv.reader(open("gmm/data/features.tsv"), delimiter="\t")])  # Gene Names (32738)

    bc_idx = {}
    for i, bc in enumerate(barcodes):  # Attaching each barcode with an index
        bc_idx[bc] = i

    namingList = np.append(genes, ["Drug"])  # Forming column name list
    totalGenes = pd.DataFrame()
    drugNames = []
    for i, cellID in enumerate(metafile["sample_id"].dropna().unique()):  # Enumerating each experiment/drug
        sample_bcs = metafile[metafile.sample_id == cellID].cell_barcode.values  # Obtaining cell bar code values for a specific experiment
        idx = [bc_idx[bc] for bc in sample_bcs]  # Ensuring barcodes match metafile for an expriment
        geneExpression = drugScreen[:, idx].T  # Obtaining all cells associated with a specific experiment (Cells, Gene)
        cellIdx = np.repeat(cellID, len(sample_bcs)) # Connecting drug name with cell
        drugNames = np.append(drugNames, cellIdx)
        totalGenes = pd.concat([totalGenes, pd.DataFrame(data=geneExpression)])  # Setting in a DF

    totalGenes.columns = genes # Attaching gene name to each column
    # totalGenes= totalGenes.fillna(0)
    totalGenes["Drug"] = drugNames # Attaching drug name to each cell
    
    return totalGenes.reset_index(drop=True), genes

def mu_sigma_normalize(geneDF):
    """Calculates the mu and sigma for every gene and returns means, sigmas, and dataframe filtered for genes expressed in > 0.1% of cells"""
    drugNames = geneDF["Drug"].values
    filtDF = geneDF.drop("Drug", axis=1)
    
    print(filtDF)
    
    assert np.isnan(filtDF.to_numpy()).any() == True
    assert np.isfinite(filtDF.to_numpy()).any() == False
    
    inplaceDF = filtDF.where(filtDF >= 0, 1, inplace=False)
    filteredGenes = filtDF[filtDF.columns[inplaceDF.mean(axis=0) > .001]]
    
    print(np.shape(filteredGenes))

    sumGenes = filteredGenes.sum(axis=0).values
    justDF = filteredGenes.to_numpy()
    
   
    
    assert np.isnan(justDF).any() == True
    assert np.isfinite(justDF).any() == False
    assert np.isnan(sumGenes).any() == True
    assert np.isfinite(sumGenes).any() == False
    assert sumGenes.any() == 0 
    
    divDF = np.divide(justDF, sumGenes)
    print(sumGenes)
    print(divDF)
    print(np.shape(justDF))
    print(np.shape(divDF))
    assert np.isnan(divDF).any() == True
    assert np.isfinite(divDF).any() == False

    

    # normG = filteredGenes.div(sumGenes, axis=1)
  
    means = filteredGenes.mean(axis=0).to_numpy()
    std = filteredGenes.std(axis=0).to_numpy()
    cv = np.divide(std, means, out=np.zeros_like(std), where=means != 0)
    
    filteredGenes["Drug"] = drugNames
    
    return filteredGenes, np.log10(means+1e-10), np.log10(cv+1e-10)


def gene_filter(geneDF, mean, std, offset_value=1.0):
    """Filters genes whos variance are higher than woudl be predicted by a Poisson distribution"""
    slope, intercept, _, _, _ = linregress(mean, std)
    inter = intercept + np.log10(offset_value)

    above_idx = np.where(std > mean * slope + inter)
    finalDF = geneDF.iloc[:, np.append(np.asarray(above_idx).flatten(), geneDF.shape[1] - 1)]

    return finalDF, above_idx


def geneNNMF(X, k=14, verbose=0, maxiteration=2000):
    """Turn gene expression into cells X components"""
    model = NMF(n_components=k, verbose=verbose, max_iter=maxiteration, tol=1e-6)
    X = X.drop("Drug", axis=1)
    W = model.fit_transform(X.to_numpy())

    sse_error = model.reconstruction_err_

    return model.components_, W, sse_error


def gene_import(offset):
    """Imports gene data from PopAlign and perfroms gene filtering process"""
    genesDF, geneNames = import_thompson_drug()
    filteredGeneDF, logmean, logstd = mu_sigma_normalize(genesDF)
    finalDF, filtered_index = gene_filter(filteredGeneDF, logmean, logstd, offset_value=offset)
    print(finalDF)
    return finalDF


def ThompsonDrugXA(numCells: int, rank: int, maxit: int, r2x=False):
    """Converts DF to Xarray given number of cells, factor number, and max iter: Factor, CellNumb, Drug, Empty, Empty"""
    # finalDF = pd.read_csv("/opt/andrew/FilteredDrugs_Offset1.3.csv")
    # finalDF = pd.read_csv("final.csv")
    finalDF = pd.read_csv('newdiv.csv')
    finalDF.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
    finalDF = finalDF.groupby(by="Drug").sample(n=numCells).reset_index(drop=True)


    _, geneFactors, _ = geneNNMF(finalDF, k=rank, verbose=0, maxiteration=maxit)
    if r2x == True:
        rank_vector, varexpl_NMF = tensor_R2X(finalDF, rank, maxit)
    else:
        rank_vector, varexpl_NMF = 0,0
        
    cmpCol = [f"Fac. {i}" for i in np.arange(1, rank + 1)]
    PopAlignDF = pd.DataFrame(data=geneFactors, columns=cmpCol)
    PopAlignDF["Drug"] = finalDF["Drug"].values
    PopAlignDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(PopAlignDF.shape[0] / numCells))

    PopAlignXA = PopAlignDF.set_index(["Cell", "Drug"]).to_xarray()
    PopAlignXA = PopAlignXA[cmpCol].to_array(dim="Factor")

    npPopAlign = np.reshape(PopAlignXA.to_numpy(), (PopAlignXA.shape[0], PopAlignXA.shape[1], -1, 1, 1))
    PopAlignXA = xa.DataArray(
        npPopAlign,
        dims=("Factor", "Cell", "Drug", "Throwaway 1", "Throwaway 2"),
        coords={"Factor": cmpCol,
            "Cell": np.arange(1, numCells + 1),
            "Drug": finalDF["Drug"].unique(),
            "Throwaway 1": ["Throwaway"],
            "Throwaway 2": ["Throwaway"],},)

    return PopAlignXA, rank_vector, varexpl_NMF

def tensor_R2X(tensor, maxrank, maxit):
    """ Calculates the R2X value even where NaN values are present"""
    rank = np.arange(1,maxrank)
    varexpl = np.empty(len(rank))
    tensor_nodrug = tensor.drop("Drug", axis=1)

    for i in range(len(rank)):
        _, _, sse_error = geneNNMF(tensor, k=rank[i], verbose=0, maxiteration=maxit)
        # print(sse_error)
        vTop = 0.0
        # print(np.sum(np.square(np.nan_to_num(tensor_nodrug.to_numpy()))))
        vTop += abs(np.sum(np.square(np.nan_to_num(tensor_nodrug.to_numpy())))-sse_error)
        varexpl[i] = 1-(vTop/np.sum(np.square(np.nan_to_num(tensor_nodrug.to_numpy()))))
        print(varexpl[i])
    return rank, varexpl