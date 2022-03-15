"""
This creates Figure 2.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorly as tl

from tensorly.decomposition import parafac, non_negative_parafac, partial_tucker

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 40), (8, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment)
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)

    maxcluster = 5
    nk, means, covar = probGMM(zflowDF, maxcluster, cellperexp)

    meansDF = zflowDF.iloc[::cellperexp, :]  # Subset to one row per expt
    meansDF = meansDF[["Time", "Ligand", "Valency", "Dose"]]  # Only keep descriptive rows
    meansDF = pd.concat([meansDF] * maxcluster, ignore_index=True)  # Duplicate for each cluster
    markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
    covarDF = meansDF.copy()

    for i, mark in enumerate(markerslist):
        markers_means = means[:, :, i]
        meansDF[mark] = markers_means.flatten(order="F")
        for j, marker in enumerate(markerslist):
            markers_covar = covar[:, :, i, j]
            covarDF[mark + "-" + marker] = markers_covar.flatten(order="F")

    meansDF["Cluster"] = np.repeat(np.arange(1, maxcluster + 1), repeats=markers_means.shape[0])  # Track clusters
    meansDF["NK"] = nk.flatten(order="F")

    covarDF["Cluster"] = np.repeat(np.arange(1, maxcluster + 1), repeats=markers_means.shape[0])  # Track clusters

    sns.scatterplot(data=meansDF, x="Dose", y="pSTAT5", hue="Cluster", ax=ax[0], style="Ligand")
    ax[0].set(xscale="log")

    sns.scatterplot(data=meansDF, x="Dose", y="NK", hue="Cluster", ax=ax[1], style="Ligand")
    ax[1].set(xscale="log")

    heatmap = meansDF
    heatmapDF = pd.DataFrame()

    for ligand in heatmap.Ligand.unique():
        for dose in heatmap.Dose.unique():
            row = pd.DataFrame()
            row["Ligand/Dose"] = [ligand + " - " + str(dose) + " (nM)"]
            for tim in heatmap.Time.unique():
                for clust in heatmap.Cluster.unique():
                    entry = heatmap.loc[(heatmap.Ligand == ligand) & (heatmap.Dose == dose) & (heatmap.Cluster == clust) & (heatmap.Time == tim)]
                    row["Cluster:" + str(clust) + " - " + str(tim) + " hrs"] = entry.pSTAT5.to_numpy()

            heatmapDF = pd.concat([heatmapDF, row])

    heatmapDF = heatmapDF.set_index("Ligand/Dose")
    sns.heatmap(heatmapDF, ax=ax[2])

    ax[3].hist(zflowDF["pSTAT5"].values, bins=1000, color='r')
    xlabel = "Event"
    ylabel = "pSTAT Signal"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    wtntermDF = meansDF.loc[meansDF["Ligand"] == "WT C-term"]

    for i, mark in enumerate(markerslist):
        sns.lineplot(data=wtntermDF, x="Dose", y=mark, hue="Cluster", ax=ax[i + 4], palette='pastel', ci=None)
        ax[i + 4].set(xscale="log")

    ligands = meansDF.Ligand.unique()
    doses = meansDF.Dose.unique()
    times = meansDF.Time.unique()
    clusters = meansDF.Cluster.unique()
    extrainfo = ["Time", "Ligand", "Valency", "Dose", "Cluster"]
    covar_values = covarDF.drop(columns=extrainfo)
    covarlist = covar_values.columns

    tensor_means = np.empty((len(times), len(ligands), len(doses), len(clusters), len(markerslist)))
    tensor_covar1 = np.empty((len(times), len(ligands), len(doses), len(clusters), len(covarlist)))
    tensor_covar2 = np.empty((len(times), len(ligands), len(doses), len(clusters), len(markerslist), len(markerslist)))

    for i, time in enumerate(times):
        for j, ligand in enumerate(ligands):
            for k, dose in enumerate(doses):
                for l, clust in enumerate(clusters):
                    entry = meansDF.loc[(meansDF.Ligand == ligand) & (meansDF.Dose == dose) & (meansDF.Cluster == clust) & (meansDF.Time == time)]
                    # Now have all  markers for a specific condition and cluster
                    for m, mark in enumerate(markerslist):
                        tensor_means[i, j, k, l, m] = entry[mark].to_numpy()

    for i, time in enumerate(times):
        for j, ligand in enumerate(ligands):
            for k, dose in enumerate(doses):
                for l, clust in enumerate(clusters):
                    # entry = covarDF.loc[(covarDF.Ligand == ligand) & (covarDF.Dose == dose) & (covarDF.Cluster == clust) & (covarDF.Time == time)]
                    # Now have all  markers for a specific condition and cluster
                    for m, mark in enumerate(markerslist):
                        for n, marker in enumerate(markerslist):
                            markers_covar = covar[:, :, m, n]
                            covarian = markers_covar.flatten(order="F")
                            tensor_covar2[i, j, k, l, m, n] = covarian[n + m]

    _, fact = parafac(tensor_means, rank=4)
    _, factors2 = non_negative_parafac(tensor_means, rank=4)
    core, factors3 = partial_tucker(tensor_covar2, modes=[0, 1, 2, 3])
    decomp = ["parafac", "NNparafac", "partialT"]

    allfactors = [fact, factors2, factors3]

    for i in range(len(allfactors)):
        factor = allfactors[i]
        decomposition = decomp[i]
        fac_time = pd.DataFrame(factor[0], columns=[f"Cmp. {i}" for i in np.arange(1, factor[0].shape[1] + 1)], index=times)
        fac_ligand = pd.DataFrame(factor[1], columns=[f"Cmp. {i}" for i in np.arange(1, factor[1].shape[1] + 1)], index=ligands)
        fac_dose = pd.DataFrame(factor[2], columns=[f"Cmp. {i}" for i in np.arange(1, factor[2].shape[1] + 1)], index=doses)
        fac_clust = pd.DataFrame(factor[3], columns=[f"Cmp. {i}" for i in np.arange(1, factor[3].shape[1] + 1)], index=clusters)

        sns.heatmap(data=fac_time, ax=ax[i + 9])
        ax[i + 9].set_title(decomposition)
        sns.heatmap(data=fac_ligand, ax=ax[i + 12])
        ax[i + 12].set_title(decomposition)
        sns.heatmap(data=fac_dose, ax=ax[i + 15])
        ax[i + 15].set_title(decomposition)
        sns.heatmap(data=fac_clust, ax=ax[i + 18])
        ax[i + 18].set_title(decomposition)

    return f
