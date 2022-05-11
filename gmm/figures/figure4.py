"""
This creates Figure 4.
"""
import numpy as np
import xarray as xa
import seaborn as sns
from jax.config import config
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, vector_guess
from gmm.tensor import markerslist


config.update("jax_enable_x64", True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 200
    zflowTensor, _ = smallDF(cellperexp)

    # probGM(Xarray, max cluster): Xarray [nk, means, covar] while using estimation gaussian parameters
    # tMeans[Cluster, Marker, Time, Dose, Ligand]
    maxcluster = 6
    ranknumb = 3
    vectorGuess = vector_guess(zflowTensor, ranknumb, maxcluster)

    times = zflowTensor.coords["Time"]
    doses = zflowTensor.coords["Dose"]
    ligand = zflowTensor.coords["Ligand"]

    clustArray = np.arange(1, maxcluster + 1)
    commonSize = (len(times), len(doses), len(ligand))
    commonDims = {"Time": times, "Dose": doses, "Ligand": ligand}

    tMeans = xa.DataArray(np.full((maxcluster, len(markerslist), *commonSize), np.nan),
                         coords={"Cluster": clustArray, "Markers": markerslist, **commonDims})

    maximizedNK, maximizedFactors, _ = minimize_func(vectorGuess, ranknumb, zflowTensor, tMeans)

    ax[0].bar(np.arange(1, maxcluster + 1), maximizedNK)
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    for i in range(0, len(maximizedFactors)):
        sns.heatmap(data=maximizedFactors[i], vmin=0, ax=ax[i + 1])

    return f
