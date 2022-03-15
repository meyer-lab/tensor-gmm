import pandas as pd
import numpy as np
import tensorly as tl 

from tensorly.decomposition import non_negative_parafac, partial_tucker


def tensor_means(meansDF,markerslist):
    # meansDF = meansDF.drop(["Valency","NK"], axis=1)
    # meansDF = meansDF.set_index(["Time", "Ligand", "Dose", "Cluster"])
    # xtensor = meansDF.to_xarray()
    # tensor = xtensor.to_array(dim="Marker")
    
    # return tensor 


    ligands = meansDF.Ligand.unique()
    doses = meansDF.Dose.unique()
    times = meansDF.Time.unique()
    clusters = meansDF.Cluster.unique()

    tensor = np.empty((len(markerslist),len(times), len(ligands), len(doses), len(clusters)))


    for i, time in enumerate(times):
        for j, ligand in enumerate(ligands):
            for k, dose in enumerate(doses):
                for l, clust in enumerate(clusters):
                    entry = meansDF.loc[(meansDF.Ligand == ligand) & (meansDF.Dose == dose) & (meansDF.Cluster == clust) & (meansDF.Time == time)]
                    # Now have all  markers for a specific condition and cluster
                    for m,mark in enumerate(markerslist):
                        tensor[m, i, j, k, l] = entry[mark].to_numpy()

    return tensor


def tensor_covar(meansDF,markerslist,covar):

    ligands = meansDF.Ligand.unique()
    doses = meansDF.Dose.unique()
    times = meansDF.Time.unique()
    clusters = meansDF.Cluster.unique()
    
    tensor = np.empty((len(markerslist),len(markerslist),len(times), len(ligands), len(doses), len(clusters)))

    for i, time in enumerate(times):
        for j, ligand in enumerate(ligands):
            for k, dose in enumerate(doses):
                for l, clust in enumerate(clusters):
                    # Now have all markers for a specific condition and cluster
                    for m,mark in enumerate(markerslist):
                        for n, marker in enumerate(markerslist):
                            markers_covar = covar[:, :, m, n]
                            covarian_flat = markers_covar.flatten(order="F")
                            tensor[m, n, i, j, k, l] = covarian_flat[m+n]

    return tensor


def tensor_decomp(meansDF,tensor,ranknumb,tensortype):

    ligands = meansDF.Ligand.unique()
    doses = meansDF.Dose.unique()
    times = meansDF.Time.unique()
    clusters = meansDF.Cluster.unique()

    if tensortype == "NNparafac":
        _, factor = non_negative_parafac(tensor,rank=ranknumb)
        tensorlist = np.arange(1,5)

    else: 
        _, factor = partial_tucker(tensor, modes=[2,3,4,5],rank = ranknumb)
        tensorlist = np.arange(0,4)

    fac_time = pd.DataFrame(factor[tensorlist[0]], columns=[f"Cmp. {i}" for i in np.arange(1,factor[tensorlist[0]].shape[1] + 1)], index=times)
    fac_ligand = pd.DataFrame(factor[tensorlist[1]], columns=[f"Cmp. {i}" for i in np.arange(1,factor[tensorlist[1]].shape[1] + 1)], index=ligands)
    fac_dose = pd.DataFrame(factor[tensorlist[2]], columns=[f"Cmp. {i}" for i in np.arange(1,factor[tensorlist[2]].shape[1] + 1)], index=doses)
    fac_clust = pd.DataFrame(factor[tensorlist[3]], columns=[f"Cmp. {i}" for i in np.arange(1,factor[tensorlist[3]].shape[1] + 1)], index=clusters)

    factor_total =  [fac_time, fac_ligand, fac_dose, fac_clust]

    return factor_total