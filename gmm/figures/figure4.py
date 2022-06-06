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
    rank = 3
    n_cluster = 3

    maximizedNK, optCP, optPTfactors, _, _ = minimize_func(zflowTensor, rank=rank, n_cluster=n_cluster)
    ptMarkerPatterns = optPTfactors[1]
    ptMarkerPatterns += np.swapaxes(ptMarkerPatterns, 0, 1)

    for i in range(3):
        dff = pd.DataFrame(ptMarkerPatterns[:, :, i], columns=markerslist, index=markerslist)
        sns.heatmap(data=dff, ax=ax[i])

    ax[3].bar(np.arange(1, maximizedNK.size + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    # CP factors
    cmpCol = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    commonDims = {"Time": zflowTensor.coords["Time"], "Dose": zflowTensor.coords["Dose"], "Ligand": zflowTensor.coords["Ligand"]}
    clustArray = np.arange(1, n_cluster + 1)
    coords = {"Cluster": clustArray, "Markers": markerslist, **commonDims}
    maximizedFactors = [pd.DataFrame(optCP.factors[ii], columns=cmpCol, index=coords[key]) for ii, key in enumerate(coords)]

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 4])

    return f
