"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, markerslist


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 200
    zflowTensor, _ = smallDF(cellperexp)
    rank = 7

    maximizedNK, maximizedFactors, optPTfactors, _, _ = minimize_func(zflowTensor, rank=rank, n_cluster=6)
    ptMarkerPatterns = optPTfactors[1]

    for i in range(3):
        dff = pd.DataFrame(ptMarkerPatterns[:, :, i], columns=markerslist, index=markerslist)
        sns.heatmap(data=dff, ax=ax[i])

    ax[3].bar(np.arange(1, maximizedNK.size + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 4])

    return f
