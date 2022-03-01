"""
This creates Figure 3.
"""
import pandas as pd
import numpy as np
import seaborn as sns


from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import cvGMM



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment); 336 conditions in total
    cellperexp = 6000
    zflowDF, experimentalcells = smallDF(cellperexp)

    time = zflowDF["Time"].unique()
    ligand = ["V91K C-term","WT C-term","N88D C-term"]
    k=0

    # Investigating distribution of pSTAT5
    for i,lig in enumerate(ligand):
        for j,tim in enumerate(time):
            ligDF = zflowDF.loc[(zflowDF["Ligand"] == lig) & (zflowDF["Time"] == tim) ]
            if i >= 1:
                k = i*len(time)
            else: 
                pass

            sns.scatterplot(data=ligDF, x="Dose", y="pSTAT5", ax=ax[j+k])
            ax[k+j].set(xscale="log",ylim=(0, 40000))
            ax[k+j].set_title("%s : %s hrs" % (lig, tim))
            
    return f
