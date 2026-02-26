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

# Install required packages
#  !pip install scanpy
#  !pip install leidenalg

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

# Set up the cutoffs
# This will depend on your data
# 10X does a lot of filtering so the data already looks pretty good.
# Dr. Plaisier says that often he wants his lower threshold for total_counts to be 2,000 or 5,000
# However, this data 
mt_cutoffs = (0.001, 7)
total_counts_cutoffs = (500, 11000)

# Visualize optimal cutoff values prior to filtering
with PdfPages('figures/scatter_total_counts_pct_mt_with_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=mt_cutoffs[0], color='red', linestyle='--')
  ax1.axhline(y=mt_cutoffs[1], color='red', linestyle='--')
  ax1.axvline(x=total_counts_cutoffs[0], color = 'red', linestyle='--')
  ax1.axvline(x=total_counts_cutoffs[1], color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Remove cells that have too many mitochondrial genes expressed or too many total counts
keep = (adata.obs['pct_counts_mt'] > mt_cutoffs[0]) & (adata.obs['pct_counts_mt'] < mt_cutoffs[1]) & (adata.obs['total_counts'] > total_counts_cutoffs[0]) & (adata.obs['total_counts'] < total_counts_cutoffs[1])
print("Removed cells: %d"%(adata.n_obs - sum(keep)))

# Actually do the filtering
adata = adata[keep, :]
adata.shape

# Check to make sure filters worked
with PdfPages('figures/scatter_total_counts_pct_mt_after_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=mt_cutoffs[0], color='red', linestyle='--')
  ax1.axhline(y=mt_cutoffs[1], color='red', linestyle='--')
  ax1.axvline(x=total_counts_cutoffs[0], color = 'red', linestyle='--')
  ax1.axvline(x=total_counts_cutoffs[1], color = 'red', linestyle='--')
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
# sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

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
# a larger resolution = more clusters
# a smaller resolution = fewer clusters
sc.tl.leiden(adata, resolution=0.6) # scanpy recommends the Leiden graph-clustering method (community detection based on optimizing modularity)
sc.pl.umap(adata, color=['leiden', 'CD3D', 'IL7R', 'CCR7', 'CD14', 'LYZ', 'S100A4', 'MS4A1', 'CD8A', 'FCGR3A', 'NKG7','CST3','PPBP'], save='_leiden.pdf')





#------------------------
# Compute clustering
#------------------------

## Make output directory for marker genes
if not os.path.exists('markergenes'):
    os.mkdir('markergenes')

## Define a list of marker genes (literature markers)
genes_dict = {'B-cell': ['CD79A', 'MS4A1'],
                     'T-cell': ['CD3D'],
                     'T-cell CD8+': ['CD8A', 'CD8B'],
                     'NK': ['GNLY', 'NKG7'],
                     'Myeloid': ['CST3', 'LYZ'],
                     'Monocytes': ['FCGR3A'],
                     'Dendritic': ['FCER1A'],
                     'Platelet': ['PPBP']}

# Make sure they are in the top 6000 highly variable genes
genes_dict = {i:list(set(genes_dict[i]).intersection(adata.var_names)) for i in genes_dict}


## Do the clustering and plotting
df_all = []
sc.tl.leiden(adata, resolution = 0.6)
sc.pl.dotplot(adata, genes_dict, groupby = 'leiden', save = '_leiden.pdf', show=False)
sc.pl.umap(adata, color = ['leiden'], use_raw = False, show = False, save = '_leiden.pdf')
sc.tl.rank_genes_groups(adata, 'leiden', corr_method = 'benjamini-hochberg', method = 'wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes = 5, show = False, sharey = False, save = '.pdf')
with open('markergenes/markergenes_dataframe.csv', 'w') as outFile:
    for x in range(max([int(i) for i in list(adata.obs['leiden'])])+1):
        df = sc.get.rank_genes_groups_df(adata, str(x), pval_cutoff = 0.05, log2fc_min = 1).sort_values(by = 'logfoldchanges', ascending = False)
        df_all.append(df)
        df['cluster'] = [x]*df.shape[0]
        if x == 0:
            df.to_csv(outFile, header = True)
        else:
            df.to_csv(outFile, header = False)

# Concatenate the pandas DataFrames
df_complete = pd.concat(df_all)

# Subset DEGs to the marker genes
marker_genes = [j for i in genes_dict.values() for j in i]
df_complete.loc[df_complete['names'].isin(marker_genes)].to_csv('markergenes/marker_genes_overlap.csv')


#----------------------------------------------
# Determining the identity of cell types
#----------------------------------------------

# Identify cell types
new_cluster_names = [
    'T', 'CD14 Monocytes',
    'NK', 'B',
    'FCGR3A Monocytes',
    'Dendritic', 'Platelet']
adata.rename_categories('leiden', new_cluster_names)

# Make UMAP
sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf', show=False)

# Additional useful plots
sc.pl.dotplot(adata, marker_genes, groupby='leiden', save='_Final.pdf', show=False)
sc.pl.heatmap(adata, genes_dict, groupby='leiden', save='_Final.pdf', show=False)
