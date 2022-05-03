"""
This creates Figure 1.
"""
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import cvGMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 50
    zflowDF, experimentalcells = smallDF(cellperexp)

    ax[0].hist(experimentalcells[0], bins=20)
    xlabel = "Number of Cells per Experiment"
    ylabel = "Events"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # scoreDF(Xarray, maxcluster, cell types): DF(Cluster #,Score, Rand Score)
    # Determines rand_score/score for GMM
    maxcluster = 18
    scoreDF = cvGMM(zflowDF, maxcluster, experimentalcells[1])

    for i in range(maxcluster):
        ax[1].plot(scoreDF.Cluster.values, scoreDF.rand_score.values)
        ax[2].plot(scoreDF.Cluster.values, scoreDF.ll_score.values)

    xlabel = "Cluster Number"
    ylabel = "Score"
    ax[1].set(xlabel=xlabel, ylabel=ylabel)
    ax[2].set(xlabel=xlabel, ylabel=ylabel)


    return f
