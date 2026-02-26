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

## Useful marker gene sets
genes_dict = {'Endothelial': ['VWF', 'CD34', 'PECAM1', 'VWF'],
              'B': ['CD79A','CD79B', 'CD19', 'IGHG1'],
              'T': ['CD3D', 'CD8A', 'CD8B'],
              'NK': ['NCAM1','NKG7','GZMB'],
              'Dendritic': ['HLA-DQA1', 'HLA-DPB1'],
              'Neutrophil': ['S100A9', 'IL1R2', 'FPR2'],
              'Macrophage': ['APOC1', 'S100A8'],
              'Proliferating_Macrophage': ['TOP2A','PCNA','MKI67'],
              'Macrophage_Microglia': ['F13A1','CD163'],
              'Microglia': ['P2RY12']}



## 1. (5pts) Set the 'sid' variable below with your student ID
#     - Relace the text and greater than and less than symbol with your student ID, important for grading
sid = 1233327240 



## 2. (5pts) Load up the packages you will need to run
#     A. Import all the packages
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

## 3. (10pts) Load the scRNA-seq data from GSE162631 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162631)
#    A. Unzip the gbm_data.zip file
#    B. (5pts) Load up the data using scanpy into variable named 'adata'
#      - Note: There is a path difference between this data and the example as it doesn't have hg19 sub-folder. Just
#        need to be sure path points at folder containing three files:  1) 'barcodes.tsv.gz', 2) 'features.tsv.gz',
#        and 3) 'matrix.mtx.gz'
#    C. (5pts) Save shape of adata into a variable 'original_shape' as demonstration of correct loading of data
adata = sc.read_10x_mtx('gbm_data/data/filtered_gene_bc_matrices/', var_names = 'gene_symbols')
original_shape = adata

## 4. (5pts) Add percent mitochondrial transcripts to meta data and use that alongside total_counts to filter cells
#    A. (5pts) Calculate QC metric of percent mitochondrial transcripts and save as 'mt' in meta data



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
mt_cutoffs = (1, 2)
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



## 7. (15pts) Compute Leiden clustering for a range of clustering resolutions [0.1, 0.2, 0.3, 0.4, 0.5]
#   A. (15pts) Set up a for loop that computes the code below and writes out all the files with resolution
#      as part of the file name:
#      - (each file worth 3pts, per iteration) For example, 'dotplot__leiden_0.1.pdf', 'umap_leiden_0.1.pdf',
#        'rank_genes_groups_leiden_0.1.pdf', 'markergenes/markergenes_dataframe_0.1.csv',
#        and 'markergenes/marker_genes_overlap_0.1.csv'

## Make output directory for marker genes, this only needs to be done once (not in for loop)
if not os.path.exists('markergenes'):
    os.mkdir('markergenes')


## Compute clustering and write out useful plots and csv files of marker genes

# Set up variables to hold data
df_all = []
# Cluster!
sc.tl.leiden(adata, resolution = 0.1)
# Plot
sc.pl.dotplot(adata, genes_dict, groupby = 'leiden', save = '_leiden.pdf', show=False)
sc.pl.umap(adata, color = ['leiden'], use_raw = False, show = False, save = '_leiden.pdf')
# Find differentially expressed genes per cluster
sc.tl.rank_genes_groups(adata, 'leiden', corr_method = 'benjamini-hochberg', method = 'wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes = 5, show = False, sharey = False, save = '_.pdf')
# Write out differentially expressed genes per cluster
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

# Subset differentially expressed genes per cluster based on established marker
# genes from cell types of interest
marker_genes = [j for i in genes_dict.values() for j in i]
df_complete.loc[df_complete['names'].isin(marker_genes)].to_csv('markergenes/marker_genes_overlap.csv')



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



