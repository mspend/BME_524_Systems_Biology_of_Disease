# Hypergeometric enrichment to identify the statistically significant overlap of transcription factors identified in 5 different experimental methods

import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats import multitest

#### Load in data ####

### Dataset 1 ###
## CRISPR-Cas9 knock-out TF hits from Toledo, et al. 2015
# Knock-out performed in two patient-derived cell lines: 0131 and 0827
df_crispr = pd.read_csv('data/CRISPR_TFs.csv')
print(df_crispr.head())

# Use the CRISPR full dataframe to identify all transcription factors
# Sets are ideal, compared to lists, because items in a set are unique
all_tfs = set(df_crispr['Entrez ID'])
print(len(all_tfs))

# Filter out significant CRISPR-Cas9 hits using the corrected p-value <0.05
# Grab the Entrez ID of the significant hits
hits_0131 = set(df_crispr.loc[df_crispr['0131_FDR']<0.05,'Entrez ID'])
hits_0827 = set(df_crispr.loc[df_crispr['0827_FDR']<0.05,'Entrez ID'])
# Hint: sets cannot be concatenated by addition in Python, only lists
# this provides a set of all the Entrez IDs
crispr_tfs = set(list(hits_0131)+list(hits_0827))
print(len(crispr_tfs))

### Dataset 2 ###
## CRISPR-Cas9 knock-out gene hits from the Broad Institute's Dependency Map
#  primarily immortalized cell lines growing on plastic for decades
df_depmap = pd.read_csv('data/DepMap_gbm.csv', header=None, names=['Entrez ID'], index_col=None)
# create a set of all the Entrez IDs
depmap_tfs = set(df_depmap.iloc[:,0])

# Subset to the known TFs in GBM using the Entrez IDs from CRISPR_TFs.csv
# Use the set method intersection, the overlap
# Order doesn't matter with this method
depmap_tfs_new = depmap_tfs.intersection(all_tfs)
print(len(depmap_tfs_new))

# Another way to do it by appending to lists:
# depmap_tfs_new = []
# print(len(depmap_tfs_new))
# for gene in depmap_tfs:
#     if gene in all_tfs:
#         depmap_tfs_new.append(gene)
# print(len(depmap_tfs_new))
