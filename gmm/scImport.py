import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import NMF
from scipy.io import mmread


def import_thompson_drug(numCells):
    """ Description of each file
    drugScreeen : str Path to a sparse matrix
    barcodes : str Path to a .tsv 10X barcodes file
    metafile : str Path to a metadata file. Must contains `cell_barcodes` and `sample_id` columns
    genes : str Path to a .tsv 10X gene file"""

    metafile = pd.read_csv('gmm/data/meta.csv') # Cell barcadoes, sample id of treatment and sample number (33482,3)
    drugScreen = mmread('/opt/andrew/drugscreen.mtx').toarray() # Sparse matrix of cell/genes (32738,33482) (Genes,Cell)
    drugScreen = drugScreen.astype(np.int16)
    barcodes = np.array([row[0] for row in csv.reader(open('gmm/data/barcodes.tsv'), delimiter="\t")]) # barcdoes of cells (33482)
    genes = np.array([row[1].upper() for row in csv.reader(open('gmm/data/features.tsv'), delimiter="\t")]) # Gene names (32738)
    
    # print(genes)
    bc_idx = {}
    for i, bc in enumerate(barcodes):
        bc_idx[bc] = i

    namingList = np.append(genes,["Drug"])
    totalGenes = pd.DataFrame()
    for i, cellID in enumerate(metafile['sample_id'].dropna().unique()):
        sample_bcs = metafile[metafile.sample_id == cellID].cell_barcode.values
        idx = [bc_idx[bc] for bc in sample_bcs]
        geneExpression = drugScreen[:,idx].T
        cellIdx = np.reshape(np.repeat(cellID,len(sample_bcs)),(-1,1))
        geneswithbars = np.hstack([geneExpression,cellIdx])
        totalGenes = pd.concat([totalGenes, pd.DataFrame(data = geneswithbars)])
        # Only running a few drugs at time to see if works
        if cellID == "Etodolac":
            break


    # Might take out depending if want to use all cells that are in each experiment
    # totalcells = np.arange(1, numCells + 1)
    totalGenes.columns = namingList
    # totalGenes = totalGenes.groupby(by="Drug").sample(n=numCells).reset_index(drop=True)
    # totalGenes["Cell"] = np.tile(np.arange(1, numCells + 1), int(totalGenes.shape[0]/numCells))

    return totalGenes, genes

def normalizeGenes(totalGenes, geneNames):
    # totalGenes[] = totalGenes.groupby(by="Drug").sample(n=numCells).reset_index(drop=True)
    totalGenes[geneNames] = totalGenes.groupby(by="Drug")[geneNames].transform(lambda x: x / np.sum(x))
    return  totalGenes
    # flowDF[tCols] = flowDF.groupby(by=gVars)[tCols].transform(lambda x: x / np.std(x)) 

    # sums = np.array(M.sum(axis=0)).flatten() # compute sums of all columns (cells)
	# # M.data = M.data.astype(float) # convert type from int to float prior to division
	
	# for i in range(len(M.indptr)-1): # for each column i
	# 	rr = range(M.indptr[i], M.indptr[i+1]) # get range rr
	# 	M.data[rr] = M.data[rr]/sums[i] # divide data values by matching column sum



def geneNNMF(X, k=14, verbose=0):
    """Turn gene expression into cells X components"""
    model = NMF(n_components=k, verbose=verbose, max_iter=2000, tol=1e-6)
    X.drop(columns=["Drug","Cell"], axis=1, inplace=True)
    W = model.fit_transform(X)

    return model.components_, W