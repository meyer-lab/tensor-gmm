"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, gen_points_GMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 200
    zflowTensor, _ = smallDF(cellperexp)
    flowDF = smallDF(400, True)
    doses = flowDF.Dose.unique()
    rank = 4
    n_cluster = 3
    time = 1.0
    ligand = "WT N-term-2"

    timei = np.where(zflowTensor.Time.values == time)[0][0]
    ligandi = np.where(zflowTensor.Ligand.values == ligand)[0]
    flowDF = flowDF.loc[(flowDF.Ligand == ligand) & (flowDF.Time == time)]

    maximizedNK, _, optPTfactors, _, _, preNormOptCP = minimize_func(zflowTensor, rank=rank, n_cluster=n_cluster)

    for dose in range(0, 6):
        points = gen_points_GMM(maximizedNK, preNormOptCP, optPTfactors, timei, dose+6, ligandi)
        pointsDF = pd.DataFrame({"Cluster": points[1],'Foxp3': points[0][:, 0], 'CD25': points[0][:, 1], 'CD45RA': points[0][:, 2], 'CD4': points[0][:, 3], 'pSTAT5': points[0][:, 4]})
        flowDF_dose = flowDF.loc[flowDF.Dose == doses[dose+6]]
        sns.histplot(data=pointsDF, x="CD4", ax=ax[dose], bins=100, binrange=[0, 7])
        sns.histplot(data=flowDF_dose, x="CD4", ax=ax[dose+6], bins=100, binrange=[0, 7])
        #sns.scatterplot(data=pointsDF, x="CD25", y="pSTAT5", hue="Cluster", palette="tab10", ax=ax[dose])
        #sns.scatterplot(data=flowDF_dose, x="CD25", y="pSTAT5", hue="Cell Type", palette="tab10", ax=ax[dose + 6])
        #ax[dose].set(xlim=(0, 6), ylim=(-5, 10), title=ligand + " at time " + str(time) + " at nM=" + str(zflowTensor.Dose.values[dose]))
        #ax[dose + 6].set(xlim=(0, 6), ylim=(-5, 10), title=ligand + " at time " + str(time) + " at nM=" + str(zflowTensor.Dose.values[dose]))

    return f
