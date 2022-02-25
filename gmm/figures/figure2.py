"""
This creates Figure 2.
"""
import enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from pathlib import Path


from .common import subplotLabel, getSetup
from ..imports import importflowDF, smallDF
from ..GMM import cvGMM, runPCA, probGMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 3))

    # ax[5].axis("off")

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment); 336 conditions in total
    cellperexp = 6000
    uniq = 336
    zflowDF, experimentalcells = smallDF(cellperexp)

    ax[0].hist(experimentalcells, bins=20)
    xlabel = "Number of Cells per Experiment"
    ylabel = "Events"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # runPCA(Runs PCA on DF with output [PCs,VarianceExplained])
    components, vcexplained = runPCA(zflowDF)

    ax[1].scatter(components, vcexplained)
    xlabel = "Principal Components"
    ylabel = "Variance"
    ax[1].set(xlabel=xlabel, ylabel=ylabel)

    maxcluster = 18

    # probGMM(Runs PCA on DF with output [PCs,VarianceExplained])
    nk, means, covariances = probGMM(zflowDF, maxcluster, cellperexp)

    statDF = []

    for i in range(uniq):
        concentration = zflowDF["Dose"].iloc[i*cellperexp]
        ligand = zflowDF["Ligand"].iloc[i*cellperexp]
        valency = zflowDF["Valency"].iloc[i*cellperexp]
        time = zflowDF["Time"][i*cellperexp]
        for j in range(maxcluster):
            ave_stat = means[i,j,:]
            statDF.append([time,ligand,valency,concentration,j+1,ave_stat[0],ave_stat[1],ave_stat[2],ave_stat[3],ave_stat[4]])

    statDF = pd.DataFrame(statDF,columns=["Time","Ligand","Valency","Concentration",
             "Cluster", "Foxp3","CD25","CD4","CD45RA","pSTAT5"])

    sns.scatterplot(data=statDF,x="Concentration",y="pSTAT5", hue="Cluster",ax=ax[4], style="Ligand")
    ax[2].legend(title = "Cluster", loc = 'best')
    xlabel = "Concentration"
    ylabel = "pSTAT5"
    ax[2].set(xlabel=xlabel, ylabel=ylabel,xscale="log")

    # statDF.to_csv('output.csv')
    return f
