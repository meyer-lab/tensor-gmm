

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from sklearn.model_selection import KFold, cross_val_score
from pathlib import Path
from sklearn.mixture import GaussianMixture
from pomegranate import *


def GMMpca(scoretype, modeltype, zflowDF, maxcluster, ksplit):
    matrixsize = np.shape(zflowDF)
    arr = np.arange(1, matrixsize[1]-2, 1)
    celltypelist = zflowDF.CellType.values
    totalDF = zflowDF.drop(columns=['CellType', 'pSTAT5'])  # Creating matrix that will be used in GMM model
    clusternumb = np.arange(2, maxcluster)  # Amount of clusters
    zflowDF = zflowDF['CellType'].values  # Obtaining celltypes

    pcagmmDF = pd.DataFrame(columns=["Component", "Cluster", "Score"])

    for jj in range(len(arr)):
        # Iterating through different number of components to determine which is best
        pca = PCA(n_components=arr[jj])
        pcaDF = pca.fit_transform(totalDF)
        kf = KFold(n_splits=ksplit)  # Cross validation for amount of splits
        gmm_labels = np.zeros([len(pcaDF)])
        bestguess = np.zeros([len(clusternumb)])
        for kk in range(len(clusternumb)):
            if modeltype == 'GMM':
                GMM = GaussianMixture(n_components=clusternumb[kk], covariance_type='full', tol=.001, max_iter=5000,
                                      reg_covar=1e-3)

                # Running GMM model on PCA dataset
                if scoretype == "RandScore":
                    # Comparing the cell type with the GMM predicted
                    best_rand = cross_val_score(GMM, pcaDF, celltypelist, cv=kf, scoring='rand_score')
                    pcagmmDF = pcagmmDF.append(pd.DataFrame({"Component": [arr[jj]], "Cluster": [clusternumb[kk]], "Score": [np.mean(best_rand)]}))

                else:
                    # Score
                    best_score = cross_val_score(GMM, pcaDF, celltypelist, cv=kf)
                    pcagmmDF = pcagmmDF.append(pd.DataFrame({"Component": [arr[jj]], "Cluster": [clusternumb[kk]], "Score": [np.mean(best_score)]}))

            else:
                # Pomegranate
                GMM = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=clusternumb[kk], X=pcaDF)

                if scoretype == "RandScore":
                    gmm_rand = []
                    for train_index, test_index in kf.split(pcaDF):
                        trainX = pcaDF[train_index, :]
                        GMM.fit(trainX)
                        gmm_labels[test_index] = GMM.predict(pcaDF[test_index, :])
                        gmm_rand.append(rand_score(celltypelist, gmm_labels))  # Comparing the cell type with the GMM predicted

                    pcagmmDF = pcagmmDF.append(pd.DataFrame({"Component": [arr[jj]], "Cluster": [clusternumb[kk]], "Score": [np.mean(gmm_rand)]}))

                else:  # Score
                    gmm_score = []
                    for train_index, test_index in kf.split(pcaDF):
                        trainX = pcaDF[train_index, :]
                        GMM.fit(trainX)
                        gmm_score.append(GMM.log_probability(pcaDF[test_index, :]))

                    pcagmmDF = pcagmmDF.append(pd.DataFrame({"Component": [arr[jj]], "Cluster": [clusternumb[kk]], "Score": [np.mean(gmm_score)]}))

    return pcagmmDF


def runPCA(dataDF):

    matrixsize = np.shape(dataDF)
    arr = np.arange(1, matrixsize[1]-2, 1)
    totalvar = np.zeros([len(arr)])
    celltypelist = dataDF.CellType.values
    totalDF = dataDF.drop(columns=['CellType', 'pSTAT5'])

    # Determining variance explained
    for a in range(len(arr)):
        pca = PCA(n_components=arr[a])
        newform = pca.fit_transform(totalDF)
        totalvar[a] = sum(pca.explained_variance_ratio_)

    return arr, totalvar
