"""
This creates Figure 4.
"""
import pandas as pd 
import seaborn as sns
import numpy as np
import tensorly as tl 


from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM, meanmarkerDF
from ..tensor import tensor_decomp, tensor_means, tensor_covar, tensor_R2X


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 10
    zflowDF, _ = smallDF(cellperexp)


    # probGM(DF,maximum cluster,cellsperexperiment): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 7
    nk, means, covar = probGMM(zflowDF, maxcluster, cellperexp)

    meansDF, markerslist = meanmarkerDF(zflowDF, cellperexp, means, nk, maxcluster)

    # tensor_means(DF,means of markers): [tensor form of means] converts DF into tensor 
    tMeans = tensor_means(zflowDF, means)

    # tensor_covar((DF,covariance of markers): [tensor form of covarinaces] converts DF into tensor 
    _ = tensor_covar(zflowDF, covar)

    # tensor_decomp(tensor means, rank, type of decomposition): [DF,tensor factors/weights] creates DF of factors for different conditions and output of decomposition 
    rank = 5
    factors_NNP,factorinfo_NNP = tensor_decomp(tMeans,rank,"NNparafac")

    newTens = tl.cp_to_tensor(factorinfo_NNP)
    shapeTens = np.shape(newTens)

    # Cluster x Marker x Ligand x Time x Concentration

    markDF = []

    for i in range(maxcluster):
        for j, tim in enumerate(zflowDF.Time.unique()): 
            for l, dose in enumerate(zflowDF.Dose.unique()):
                for k, ligand in enumerate(zflowDF.Ligand.unique()):
                        ave_mark = newTens[i,:,k,j,l]
                        markDF.append([ligand, dose, tim, i+1, ave_mark[0], ave_mark[1], ave_mark[2], ave_mark[3], ave_mark[4]])


    markDF = pd.DataFrame(markDF, columns=["Ligand","Concentration", "Time","Cluster", "Foxp3", "CD25", "CD4", "CD45RA", "pSTAT5"])

    print(markDF)

    covarDF = meansDF.copy()

    for i, mark in enumerate(markerslist):
        for j,marker in enumerate(markerslist):
            markers_covar = covar[:, :, i, j]
            covarDF[mark + "-" + marker] = markers_covar.flatten(order="F")


    covarDF["Ligand"] = covarDF["Ligand"] + "-" + covarDF["Valency"].astype(str)

    covarDF = covarDF.drop(columns = np.append(markerslist,["NK","Valency"]))

    print(covarDF)

    return f