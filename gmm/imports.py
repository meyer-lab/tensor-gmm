""" Methods for data import and normalization. """

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def smallDF(numCells: int):
    """Creates Xarray of a specific # of experiments
    Zscores all markers per experiment but pSTAT5 normalized over all experiments
    Outputs amount of experiments and cell types as an Xarray"""
    # numCells = Amount of cells per experiment
    flowArrow = importflowDF()
    gVars = ["Time", "Dose", "Ligand", "Valency"]
    # Columns that should be trasformed
    tCols = ["Foxp3", "CD25", "CD45RA", "CD4"]
    transCols = tCols + ["pSTAT5"]

    # Data was measured for CD3/CD8/CD56 was not measured for non-Tregs/Thelpers
    # Also drop columns with missing values
    flowArrow = flowArrow.filter(pa.compute.is_finite(flowArrow["Foxp3"]))
    flowArrow = flowArrow.select(transCols + gVars + ["Cell Type"])
    flowDF = flowArrow.to_pandas()

    # Group and subset
    experimentcells = flowDF.groupby(by=gVars).size()
    flowDF[tCols] = flowDF.groupby(by=gVars)[tCols].transform(lambda x: x / np.std(x))  # Dividing by std per experiement
    for mark in transCols:
        flowDF = flowDF[flowDF[mark] < flowDF[mark].quantile(0.995)]  # Getting rid of outlier values
    flowDF = flowDF.groupby(by=gVars).sample(n=numCells).reset_index(drop=True)

    # Add valency to the name
    flowDF["Ligand"] = flowDF["Ligand"] + "-" + flowDF["Valency"].apply(lambda x: f'{x:.0f}')
    flowDF.drop(columns=["Valency"], axis=1, inplace=True)

    flowDF["Cell Type"] = flowDF["Cell Type"].replace({"None": 1, "Treg": 2, "Thelper": 3})
    flowDF["pSTAT5"] /= np.std(flowDF["pSTAT5"])  # For pSTAT5 only, dividing my std of all experiments
    flowDF.sort_values(by=["Time", "Dose", "Ligand"], inplace=True)

    flowDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(flowDF.shape[0] / numCells))
    flowDF = flowDF.set_index(["Cell", "Time", "Dose", "Ligand"]).to_xarray()
    cell_type = flowDF["Cell Type"]
    flowDF = flowDF.drop_vars(["Cell Type"])
    flowDF = flowDF[transCols].to_array(dim="Marker")
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]

    return flowDF, (experimentcells, cell_type)


def importflowDF():
    """Downloads all conditions, surface markers and cell types.
    Cells are labeled via Thelper, None, Treg, CD8 or NK"""
    monomeric = pq.read_table("/opt/andrew/FlowDataGMM_Mon_NoSub.pq")
    dimeric = pq.read_table("/opt/andrew/FlowDataGMM_DimWT_NoSub.pq")
    schema = monomeric.schema
    schema = schema.set(5, pa.field('Foxp3', pa.float64()))
    schema = schema.set(6, pa.field('CD25', pa.float64()))
    schema = schema.set(7, pa.field('CD4', pa.float64()))
    schema = schema.set(8, pa.field('CD45RA', pa.float64()))
    schema = schema.set(9, pa.field('pSTAT5', pa.float64()))
    schema = schema.set(12, pa.field('CD56', pa.float64()))
    schema = schema.set(13, pa.field('CD3', pa.float64()))
    schema = schema.set(14, pa.field('CD8', pa.float64()))
    monomeric = monomeric.cast(schema)
    return pa.concat_tables([monomeric, dimeric], promote=True)
