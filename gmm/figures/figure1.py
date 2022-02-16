"""
This creates Figure 1.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from pathlib import Path

from .common import subplotLabel, getSetup
from ..imports import importflowDF, smallDF
from ..GMM import GMMpca, runPCA


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 3))

    ax[5].axis("off")

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment)
    zflowDF = smallDF(100)

    # PCA(Runs PCA on dataframe with output [PCs,VarianceExplained])
    components, vcexplained = runPCA(zflowDF)

    ax[0].scatter(components, vcexplained)
    xlabel = "Principal Components"
    ylabel = "Variance"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # Determining rand_score for GMM with dataframe

    # GMMpca(scoretype, modeltype, zflowDF, maxcluster+1, ksplit)
    gmmDF_rand = GMMpca("RandScore", "GMM", zflowDF, 21, 20)
    gmmDF_score = GMMpca("Score", "GMM", zflowDF, 21, 20)
    gmmDF_rand_pom = GMMpca("RandScore", "Pomengranate", zflowDF, 21, 20)
    gmmDF_score_pom = GMMpca("Score", "Pomengranate", zflowDF, 21, 20)

    gmmDF = [gmmDF_rand, gmmDF_score, gmmDF_rand_pom, gmmDF_score_pom]
    xlabel = "Cluster Number"
    ylabel = "Score"

    for i in range(len(gmmDF)):
        for j in range(len(components)):
            cvDF = gmmDF[i].loc[gmmDF[i].Component == components[j]]
            # print(cvDF)
            ax[i + 1].plot(cvDF.Cluster.values, cvDF.Score.values, label=components[j])
        ax[i + 1].legend(title="Component Number", loc='best')
        ax[i + 1].set(xlabel=xlabel, ylabel=ylabel)


    # filepath = Path('gmm/output/figure1.csv')
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # zflowDF.to_csv(filepath)

    return f
