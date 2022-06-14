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
    sumGenes = totalGenes[geneNames].sum(axis=0).tolist()
    newGene = pd.DataFrame(data=np.reshape(sumGenes,(1,-1)),columns=geneNames)

    normG = totalGenes[geneNames].div(newGene,axis=1)
    normG = normG[geneNames].replace(np.nan,0)

    drugs = totalGenes.iloc[:,-1].tolist()
    drugs = np.reshape(drugs,(-1,1))
    normalizeGenesDF = pd.concat([normG, pd.DataFrame(data = drugs,columns = ["Drug"])],axis=1)

    return  normalizeGenesDF


def mu_sigma(geneDF):
    """Calculates the mu and sigma for every gene and returns means, sigmas, and dataframe filtered for genes expressed in > 0.1% of cells"""
    noDrugDF = geneDF.drop("Drug", axis=1).astype("float64")
    filtDF = noDrugDF.where(noDrugDF >= 0, 1, inplace=False)
    drugCol = geneDF
    noDrugDF = noDrugDF[noDrugDF.columns[filtDF.mean(axis=0) > 0.01]]
    geneDF = pd.concat([noDrugDF, drugCol], axis=1)
    means = noDrugDF.mean(axis=0).to_numpy()
    std = noDrugDF.std(axis=0).to_numpy())
    return geneDF, means, std


def geneNNMF(X, k=14, verbose=0):
    """Turn gene expression into cells X components"""
    model = NMF(n_components=k, verbose=verbose, max_iter=2000, tol=1e-6)
    X.drop(columns=["Drug","Cell"], axis=1, inplace=True)
    W = model.fit_transform(X)

    return model.components_, W