"""
This creates Figure 2.
"""
import numpy as np
import pandas as pd
import seaborn as sns

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM, meanmarkerDF, heatmapmeansDF


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 30), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment)
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)

    maxcluster = 5
    nk, means, covar = probGMM(zflowDF, maxcluster, cellperexp)

    meansDF, markerslist = meanmarkerDF(zflowDF, cellperexp, means, nk, maxcluster)

    sns.scatterplot(data=meansDF, x="Dose", y="pSTAT5", hue="Cluster", ax=ax[0], style="Ligand")
    ax[0].set(xscale="log")

    sns.scatterplot(data=meansDF, x="Dose", y="NK", hue="Cluster", ax=ax[1], style="Ligand")
    ax[1].set(xscale="log")

    heatmapDF = heatmapmeansDF(meansDF)
    sns.heatmap(heatmapDF, ax=ax[2])

    ax[3].hist(zflowDF["pSTAT5"].values, bins=1000, color='r')
    xlabel = "Event"
    ylabel = "pSTAT Signal"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    wtntermDF = meansDF.loc[meansDF["Ligand"] == "WT C-term"]

    for i, mark in enumerate(markerslist):
        sns.lineplot(data=wtntermDF, x="Dose", y=mark, hue="Cluster", ax=ax[i + 4], palette='pastel', ci=None)
        ax[i + 4].set(xscale="log")

    return f
