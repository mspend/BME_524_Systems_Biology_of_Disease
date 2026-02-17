##########################################################
## BME 494/598:  singleCell_BasicProcessingTemplate.py  ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed by: Plaisier Lab                          ##
##   (https://plaisierlab.engineering.asu.edu/)         ##
##   Arizona State University                           ##
##   242 ISTB1, 550 E Orange St                         ##
##   Tempe, AZ  85281                                   ##
## @Author:  Chris Plaisier                             ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################

# on command line
# cd Dropbox\ \(ASU\)/pbmc

# pip install scanpy
# pip install leidenalg

######################################
## Restart the kernel!              ##
## Under Consoles -> Restart kernel ##
######################################


####################
## Load libraries ##
####################

import numpy as np
import pandas as pd
import scanpy as sc
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#------------------------
# Load data
#-----------------------
# change working directory in python
#os.getcwd()
#os.chdir("/Users/samanthaoconnor/Dropbox (ASU)/pbmc")

# Build scanpy data object
adata = sc.read_10x_mtx(
    'data/filtered_gene_bc_matrices/hg19/',   # the directory with the `.mtx` file
    var_names='gene_symbols',                 # use gene symbols for the variable names (variables-axis index)
    cache=True)

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
adata
adata.shape # see dimensions of adata object (cells,genes)
# adata is an AnnData object that can be slices like a dataframe

#------------------------
# Preprocessing
#-----------------------

# Basic filtering
sc.pp.filter_cells(adata, min_genes=200) # filter out cells that have less than 200 genes
sc.pp.filter_genes(adata, min_cells=3) # filter out genes that are detected in less than 3 cells
adata.shape

# Generate quality control metrics
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata
# High proportions of mito genes are indicative of poor-quality cells (Islam et al. 2014; Ilicic et al. 2016), possibly because of loss of cytoplasmic RNA from perforated cells
adata.var[adata.var['mt']] # look at the mitochrondrial genes
adata.obs['pct_counts_mt']
adata.obs['pct_counts_mt'].describe()

# Violin plots of qc metrics
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save= '.pdf') # can also save as png

# Scatter plots of qc metrics
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save ='_total_counts_pct_mt.pdf')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', save='_total_counts_genes.pdf')

# Visualize optimal cutoff values prior to filtering
with PdfPages('figures/scatter_total_counts_pct_mt_with_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=12, color='red', linestyle='--')
  ax1.axvline(x=500, color = 'red', linestyle='--')
  ax1.axvline(x=10000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Remove cells that have too many mitochondrial genes expressed or too many total counts
keep = (adata.obs['pct_counts_mt']> 0.001) & (adata.obs['pct_counts_mt'] < 12) & (adata.obs['total_counts'] > 500) & (adata.obs['total_counts'] < 10000)
print("Removed cells: %d"%(adata.n_obs - sum(keep)))

#keep = (adata.obs['pct_counts_mt'] < 5) & (adata.obs['total_counts'] < 10000) # show that total counts & genes are directly related; therefore, you really only need one cut-off
#print("Removed cells: %d"%(adata.n_obs - sum(keep)))

# Actually do the filtering
adata = adata[keep, :]
adata.shape

# Check to make sure filters worked
with PdfPages('figures/scatter_total_counts_pct_mt_after_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=12, color='red', linestyle='--')
  ax1.axvline(x=500, color = 'red', linestyle='--')
  ax1.axvline(x=10000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Normalize so that counts become comparable among cells
sc.pp.normalize_total(adata, target_sum=1e4)

# Logarithmize the data
sc.pp.log1p(adata)

# Identify and plot highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=6000) # set your own number of highly variable genes
sc.pl.highly_variable_genes(adata, save='.pdf')

# Save raw data prior to subsetting data to highly variable genes
adata.raw = adata

# Subset data for highly variable genes
adata = adata[:, adata.var.highly_variable]
adata

# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# Scale each gene to unit variance
sc.pp.scale(adata, max_value=10)


#----------------------------------------------
# PCA / clustering /  marker gene analysis
#----------------------------------------------

# PCA
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CD3D', save = '.pdf')
sc.pl.pca_variance_ratio(adata, log=True, save = '.pdf')
adata

# Compute the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Visualize cells in 2D
sc.tl.umap(adata)
sc.pl.umap(adata, color=['CD3D', 'NKG7', 'LST1','PPBP'], save ='.pdf')
sc.pl.umap(adata, color=['CD3D', 'NKG7', 'LST1','PPBP'], use_raw=False, save='_V2.pdf') # scaled and corrected gene expression values

# Clustering
sc.tl.leiden(adata, resolution=0.5) # scanpy recommends the Leiden graph-clustering method (community detection based on optimizing modularity)
sc.pl.umap(adata, color=['leiden', 'CD3D', 'NKG7'], save='_leiden.pdf')

# Finding marker genes
#sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
#sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='.pdf')

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_V2.pdf')
sc.pl.rank_genes_groups_heatmap(adata, save='_marker_genes.pdf')


# Write out CSV file wit clusters as different rows
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
dfs = []
for group in groups:
    tmp = pd.DataFrame({key: result[key][group] for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj']})
    tmp['Cluster'] = group
    dfs.append(tmp)

pd.concat(dfs, axis=0).to_csv('rank_genes_similar_to_Seurat.csv')


# Write out CSV file with clusters as different columns
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
dfs = []
for group in groups:
    tmp = pd.DataFrame({key+'_'+group: result[key][group] for key in ['names', 'logfoldchanges', 'pvals', 'pvals_adj']})
    tmp = tmp.sort_values('names_'+group)
    tmp.index = tmp['names_'+group]
    tmp = tmp.drop('names_'+group,axis=1)
    dfs.append(tmp)

pd.concat(dfs, axis=1).to_csv('rank_genes_similar_to_what_we_had_before.csv')


# Define a list of marker genes (literature markers)
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

sc.pl.umap(adata, color=marker_genes, save='_with_marker_genes.pdf')


# Show the 10 top ranked genes per cluster
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)

# Get a table with scores and results
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)

# Compare to a single cluster
sc.tl.rank_genes_groups(adata, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)

# More detailed
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8, save='marker_genes_violin.pdf')

# Compare genes across groups
sc.pl.violin(adata, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_three_genes.pdf')

# Identify cell types
new_cluster_names = [
    'T', 'CD14 Monocytes',
    'NK', 'B',
    'FCGR3A Monocytes',
    'Dendritic', 'Platelet']
adata.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf')

# other marker gene visualization
sc.pl.violin(adata, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_V2')
sc.pl.violin(adata, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', rotation=90, save='_V3')
sc.pl.violin(adata, marker_genes, groupby='leiden', rotation=90, save='_V4')

for gene in marker_genes[0:5]:
    sc.pl.violin(adata, gene, groupby='leiden', rotation=90, save = '_'+gene+'.pdf')

sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', swap_axes = True, save='.pdf')
sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', save='V2.pdf')

sc.pl.dotplot(adata, marker_genes, groupby='leiden', save='pdf')
sc.pl.heatmap(adata, marker_genes, groupby='leiden', save='.pdf')

marker_genes_dict = {'B-cell': ['CD79A', 'MS4A1'],
                     'T-cell': 'CD3D',
                     'T-cell CD8+': ['CD8A', 'CD8B'],
                     'NK': ['GNLY', 'NKG7'],
                     'Myeloid': ['CST3', 'LYZ'],
                     'Monocytes': ['FCGR3A'],
                     'Dendritic': ['FCER1A']}

sc.pl.heatmap(adata, marker_genes_dict, groupby='leiden', save='_V2.pdf')
