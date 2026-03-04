##########################################################
## BME 524:  singleCell_BasicProcessingTemplate.py      ##
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


## 1. (5pts) Set the 'sid' variable below with your student ID
#     - Relace the text and greater than and less than symbol with your student ID, important for grading
sid = ##########

## 2. (5pts) Load up the packages you will need to run
#     A. Import all the packages
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

## Make output directory for marker genes
if not os.path.exists('figures'):
    os.mkdir('figures')

## Make output directory for marker genes
if not os.path.exists('markergenes'):
    os.mkdir('markergenes')

## 3. (10pts) Load the scRNA-seq data from GSE162631 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162631)
#    A. Unzip the gbm_data.zip file
#    B. (5pts) Load up the data using scanpy into variable named 'adata'
#      - Note: There is a path difference between this data and the example as it doesn't have hg19 sub-folder. Just
#        need to be sure path points at folder containing three files:  1) 'barcodes.tsv.gz', 2) 'features.tsv.gz',
#        and 3) 'matrix.mtx.gz'
#    C. (5pts) Save shape of adata into a variable 'original_shape' as demonstration of correct loading of data
adata = sc.read_10x_mtx('data/filtered_gene_bc_matrices/', var_names = 'gene_symbols')
original_shape = adata.shape

## 4. (5pts) Add percent mitochondrial transcripts to metadata and use that alongside total_counts to filter cells
#    A. (5pts) Calculate QC metric of percent mitochondrial transcripts and save as 'mt' in meta data

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

## 5. (20pts) Add percent mitochondrial transcripts to meta data and use that alongside total_counts to filter cells
#    A. Set cutoff variables to something reasonable:
#       - (10pts) 'mt_cutoffs' = (<lower>, <upper>)
#           - Low should be between 1-2 %
#           - High should be set at the top of the distribution, try not to cut too deeply
#       - (10ots) 'total_counts_cutoffs' = (<lower>, <upper>)
#           - Low should be set to 2000 or 5000, (note not to use commas)
#           - High should be set to get rid of obvious outlier multiplets (again no commas in number)
#    B. Use scatterplot to determine the cutoffs for the lower and upper cutoffs for both total counts and percent
#       mitochondrial genes
#       - File name for scatter plot should be: 'figures/scatter_total_counts_pct_mt_with_cutoffs.pdf'
#       - Set them, and re-run to find a good cutoff
#    C. Filter the scRNA-seq data using these cutoffs
#    D. Save the shape of 'adata' into a variable called 'cleaned_shape'

# Set cutoffs appropriately
mt_cutoffs = (1, 20)
total_counts_cutoffs = (5000, 95000)

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
cleaned_shape = adata.shape

## 6. (15pts) Subset to the 6,000 most highly variable genes, run PCA, and compute the neighborhood graph
#    A. Compute the top 6,000 most highly variable genes, and subset 'adata' based on the gene list
#    B. (5pts) Save the shape of 'adata' into a variable called 'hvg_shape'
#    C. Run PCA on data using scanpy uysing the svd_solver='arpack'
#    D. (5pts) Compute the neighborhood graph, with n_neighbors = 5 and n_pcs=40
#       - Setting the k-nearest neighbors number correctly is important for the
#         neighborhood graph to detect smaller cluster sizes
#           - Smaller clusters need smaller n_neighbors, i.e., n_neighbors = 5
#           - If clusters are generally larger overall, then the default of
#             n_neighbors = 15 cells is reasonable
#    E. (5pts) Ensure that the marker genes in our 'genes_dict' are all in the highly variable genes (useful code below):
#       genes_dict = {i:list(set(genes_dict[i]).intersection(adata.var_names)) for i in genes_dict}

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
hvg_shape = adata.shape

# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# Scale each gene to unit variance
sc.pp.scale(adata, max_value=10)

# PCA
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CD3D', save = '.pdf')
sc.pl.pca_variance_ratio(adata, log=True, save = '.pdf')

# Compute the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=5, n_pcs=40)

# Visualize cells in 2D
sc.tl.umap(adata)

## Useful marker gene sets
genes_dict = {'Endothelial': ['VWF', 'CD34', 'PECAM1'],
              'B': ['CD79A','CD79B', 'CD19', 'IGHG1'],
              'T': ['CD3D', 'CD8A', 'CD8B'],
              'NK': ['NCAM1','NKG7','GZMB'],
              'Dendritic': ['HLA-DQA1', 'HLA-DPB1'],
              'Neutrophil': ['S100A9', 'IL1R2', 'FPR2'],
              'Macrophage': ['APOC1', 'S100A8'],
              'Proliferating_Macrophage': ['TOP2A','PCNA','MKI67'],
              'Macrophage_Microglia': ['F13A1','CD163'],
              'Microglia': ['P2RY12']}

# Make sure the marker genes are in the top 6000 highly variable genes
genes_dict = {i:list(set(genes_dict[i]).intersection(adata.var_names)) for i in genes_dict}


## 7. (15pts) Compute Leiden clustering for a range of clustering resolutions [0.1, 0.2, 0.3, 0.4, 0.5]
#   A. (15pts) Set up a for loop that computes the code below and writes out all the files with resolution
#      as part of the file name:
#      - (each file worth 3pts, per iteration) For example, 'dotplot__leiden_0.1.pdf', 'umap_leiden_0.1.pdf',
#        'rank_genes_groups_leiden_0.1.pdf', 'markergenes/markergenes_dataframe_0.1.csv',
#        and 'markergenes/marker_genes_overlap_0.1.csv'

## Compute clustering and write out useful plots and csv files of marker genes

resolutions = [0.1, 0.2, 0.3, 0.4, 0.5]
for res in resolutions:
    # Set up variables to hold data
    df_all = []
    # Cluster!
    sc.tl.leiden(adata, resolution = res)
    # Plot
    sc.pl.dotplot(adata, genes_dict, groupby = 'leiden', save = f'_leiden_{res}.pdf', show=False)
    sc.pl.umap(adata, color = ['leiden'], use_raw = False, show = False, save = f'_leiden_{res}.pdf')
    # Find differentially expressed genes per cluster
    sc.tl.rank_genes_groups(adata, 'leiden', corr_method = 'benjamini-hochberg', method = 'wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes = 5, show = False, sharey = False, save = f'_{res}.pdf')
    # Write out differentially expressed genes per cluster
    with open(f'markergenes/markergenes_dataframe_{res}.csv', 'w') as outFile:
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

    # Subset differentially expressed genes per cluster based on established marker
    # genes from cell types of interest
    marker_genes = [j for i in genes_dict.values() for j in i]
    df_complete.loc[df_complete['names'].isin(marker_genes)].to_csv(f'markergenes/marker_genes_overlap_{res}.csv')


## 8. (25pts) Choose a clustering resolution, redo clustering, relabel clusters, and write out final plots
#   A. (5pts) Set 'res1' variable
#   B. (5pts) Rerun clustering, and save the number of clusters into the variable 'final_cluster_num':
#        final_cluster_num = adata.obs['leiden'].describe()['unique']
#   C. (10pts) Set 'new_cluster_names' to relabel the clusters based on the mapping you have developed
#      - Use the final cell labels from the keys of 'gene_dict', e.g., 'Macrophage', 'Microglia', etc.
#      - It is expected that one cluster may not have an obvious mapping, label it as '???'
#   D. (5pts) Write out the final plots:
#     - UMAP with new labels: 'umap_new_idents.pdf'
#     - Dotplot with new labels: 'dotplot__Final.pdf'
#     - Heatmap with genes_dict: 'heatmap_Final.pdf'


# sc.pl.umap(adata, color=['S100A9', 'VWF', 'CD3D', 'GZMB' ], use_raw=False, save=None) # scaled and corrected gene expression values

res1 = 0.5
sc.tl.leiden(adata, resolution = res1)
# save the number of clusters into the variable 'final_cluster_num':
final_cluster_num = adata.obs['leiden'].describe()['unique']


# must be in order of cluster number
new_cluster_names = [
    'Macrophage',
    'Microglia',
    'Dendritic',
    'Dendritic',
    'Macrophage_Microglia',
    '???',
    'Proliferating_Macrophage',
    'Neutrophil',
    'Macrophage',
    'Neutrophil',
    'Proliferating_Macrophage',
    'Endothelial',
    'T',
    'NK',
    ]

adata.rename_categories('leiden', new_cluster_names)

# Plot
sc.pl.umap(adata, color = ['leiden'], use_raw = False, show = False, save = '_new_idents.pdf')
sc.pl.dotplot(adata, genes_dict, groupby = 'leiden', save = '__Final.pdf', show=False)
sc.pl.heatmap(adata, genes_dict, groupby='leiden', save='_Final.pdf', show=False)
