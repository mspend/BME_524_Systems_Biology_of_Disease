##########################################################
## OncoMerge:  learningPython.py                        ##
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


############
## Pandas ##
############

# https://www.kaggle.com/learn/pandas

# Import pandas with common alias pd
import pandas as pd


## De novo intialization of Pandas dataframes

# De novo Pandas DataFrame from nested lists
a = [['0h',10],['12h',20],['24h',18],['48h',8]]
df1 = pd.DataFrame(data = a, columns = ['Name','Expression'])
print(df1)

# De novo Pandas DataFrame from dictionary
a = {'Name':['0h','12h','24h','48h'],'Expression':[10,20,18,8]}
df1 = pd.DataFrame(data = a)
print(df1)

# De novo Pandas DataFrame from dictionary v2
a = [{'Name':'0h','Expression':10}, {'Name':'12h','Expression':20}, {'Name':'24h','Expression':18}, {'Name':'48h','Expression':8}]
df1 = pd.DataFrame(data = a)
print(df1)

# De novo Pandas DataFrame from dictionary v2
a = {'Expression':{'0h':10, '12h':20, '24h':18, '48h':8}}
df1 = pd.DataFrame(data = a)
print(df1)


## Load up real-world data into pandas
# Using data from GSE79731 on GEO:  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE79731
# A study of Mycobacterium tuberculosis infection

# Load phenotype data
phenos = pd.read_csv('data/phenos.csv', header = 0, index_col = 0)

## Selecting out a specific row or column

# What are the columns and row names?
print(phenos.columns)
print(phenos.index)

# What infection types exist in the data
print(phenos.loc['infection'])
print(set(phenos.loc['infection']))

# What time points exist in the data
print(phenos.loc['timepoints'])
print(set(phenos.loc['timepoints']))


## Loading transcriptomics data and translating gene ids to gene symbols

# Load gene expression data
gexp = pd.read_csv('data/GSE79731_series_matrix.csv', header = 0, index_col = 0)
print(gexp.columns)
print(gexp.index)
print(gexp.shape)

# Load gene info - separated by tabs not commas, and index_col is set to 1 not 0
# https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/
gene_info = pd.read_csv('data/Mus_musculus.gene_info', sep='\t', header = 0, index_col = 1)
print(gene_info.columns)
print(gene_info.index)
print(gene_info.shape)

# Translate Entrez gene ids to gene symbols
gexp2 = gexp.loc[gexp.index.isin(gene_info.index)]
gexp2.index = gene_info.loc[gexp2.index,'Symbol']
print(gexp2.index)

# Write out the gene expression data with gene symbols
gexp2.to_csv('data/GSE79731_series_matrix_symbols.csv')

# Select top most variant genes
top2000 = gexp2.var(axis=1).sort_values(ascending=False).index[range(2000)]



#########################
## Matplotlib - pyplot ##
#########################

### Quality control for BMM infected with TB (GSE79731)

# https://matplotlib.org/stable/tutorials/introductory/pyplot.html

import matplotlib.pyplot as plt

# Prinicipal components analysis of gene expression data
from sklearn.decomposition import PCA # We will discuss sklearn more on Thursday

# PCA for gene expression data top 2000 most variant genes
pca = PCA(n_components=2)
gexp2_pca = pca.fit(gexp2.loc[top2000].T).transform(gexp2.loc[top2000].T)
print(gexp2_pca.shape)
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

# Plot PCA
plt.figure()
colors = {'0h':'#fee5d9', '4h':'#fcae91', '8h':'#fb6a4a', '24h':'#de2d26', '48h':'#a50f15'}

# Plot scatter plot
for i in colors:
    plt.scatter(gexp2_pca[phenos.loc['timepoints']==i, 0], gexp2_pca[phenos.loc['timepoints']==i, 1], color=colors[i], alpha=.8, linewidths=0.75, label=i, edgecolors='black')

# Add axis labels
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# Add a line plot to show temporal ordering
gexp2_pca2 = pd.concat([pd.DataFrame(gexp2_pca, index = gexp2.columns, columns = ['PCA1','PCA2']), phenos.T], axis = 1)
plt.plot(gexp2_pca2.groupby('timepoints', sort = False).median(numeric_only=True)['PCA1'], gexp2_pca2.groupby('timepoints', sort = False).median(numeric_only=True)['PCA2'], alpha=0.5, color='grey')

# Add a legend and title
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA:  BMM infected with TB')

# Show the plot
plt.show()

# OR

# Save as a png
plt.savefig('PCA_plot.png')



#############
## Seaborn ##
#############

### More quaitly control for BMM infected with TB (GSE79731)

# https://www.kaggle.com/learn/data-visualization

import matplotlib.pyplot as plt
import seaborn as sns

## Correlate samples based on top 2000 most variant genes
c1 = gexp2.loc[top2000].corr()
c1.columns = phenos.loc['title']
c1.index = phenos.loc['title']


## Plot using heatmap
fig = plt.figure()
sns.heatmap(c1, cmap='Blues')
fig.tight_layout()
plt.show()


## Plot using clustermap
# Make column and row colors
colRow_colors = [colors[i] for i in phenos.loc['timepoints']]

# Plot clustermap
sns.clustermap(c1, cmap='Blues', col_colors=colRow_colors, vmin=0, vmax=1)
plt.show()


## Plot genes by patients using clustermap
# Make column and row colors
colRow_colors = [colors[i] for i in phenos.loc['timepoints']]

# Plot clustermap
sns.clustermap(gexp2.loc[top2000], cmap = sns.color_palette("vlag", n_colors = 33), col_colors=colRow_colors, col_cluster=False, standard_scale=0)
plt.show()



##############
## PdfPages ##
##############

# https://matplotlib.org/stable/gallery/misc/multipage_pdf.html

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('GSE79731.pdf') as pdf:

    # Plot PCA
    plt.figure()
    colors = {'0h':'#fee5d9', '4h':'#fcae91', '8h':'#fb6a4a', '24h':'#de2d26', '48h':'#a50f15'}
    for i in colors:
        plt.scatter(gexp2_pca[phenos.loc['timepoints']==i, 0], gexp2_pca[phenos.loc['timepoints']==i, 1], color=colors[i], alpha=.8, linewidths=0.75, label=i, edgecolors='black')
    gexp2_pca2 = pd.concat([pd.DataFrame(gexp2_pca, index = gexp2.columns, columns = ['PCA1','PCA2']), phenos.T], axis = 1)
    plt.plot(gexp2_pca2.groupby('timepoints', sort = False).median(numeric_only=True)['PCA1'], gexp2_pca2.groupby('timepoints', sort = False).median(numeric_only=True)['PCA2'], alpha=0.5, color='grey')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA:  BMM infected with TB')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Plot correlation clustermap
    sns.clustermap(c1, cmap='Blues', col_colors=colRow_colors, vmin=0, vmax=1)
    pdf.savefig()
    plt.close()

    # Plot gene expression clustermap
    sns.clustermap(gexp2.loc[top2000], cmap = sns.color_palette("vlag", n_colors = 33), col_colors=colRow_colors, col_cluster=False, standard_scale=0)
    pdf.savefig()
    plt.close()



##############
## Subplots ##
##############

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

import seaborn as sns
from scipy import stats
import statsmodels.stats.multitest as mt

# Subplots is useful for combining multiple figures into one larger figure.
res1 = pd.DataFrame(index = gexp2.index, columns = ['R','p-value','Adjusted.p-value'])

# Correlate gene expression versus time
for i in gexp2.index:
    res1.loc[i,['R','p-value']] = stats.pearsonr(gexp2.loc[i], [0,0,0,4,4,4,8,8,8,24,24,24,48,48,48])

# Correct p-values
res1['Adjusted.p-value'] = mt.multipletests(res1['p-value'], method='fdr_bh')[1]

# Plot top 4 most correlated genes
with PdfPages('topGenes.pdf') as pdf:
    fig1, ax1 = plt.subplots(3,3)
    topGenes = res1.sort_values('Adjusted.p-value').iloc[range(9)].index
    for i in range(len(topGenes)):
        ax1[int(i / 3)][i % 3].scatter(x=[int(i.rstrip('h')) for i in phenos.loc['timepoints']], y=gexp2.loc[topGenes[i]])
        ax1[int(i / 3)][i % 3].set_ylabel('Rel. Exp.')
        ax1[int(i / 3)][i % 3].set_xlabel('Time (h)')
        ax1[int(i / 3)][i % 3].set_title(topGenes[i])

    fig1.tight_layout()
    pdf.savefig()
    plt.close()


# Plot top 4 most correlated genes
with PdfPages('topGenes_box.pdf') as pdf:
    fig1, ax1 = plt.subplots(3,3)
    topGenes = res1.sort_values('Adjusted.p-value').iloc[range(9)].index
    for i in range(len(topGenes)):
        sns.boxplot(y=gexp2.loc[topGenes[i]], x=phenos.loc['timepoints'], ax = ax1[int(i / 3)][i % 3], linewidth = 0.7)
        #ax1[int(i / 2)][i % 2].set_title(topGenes[i])

    fig1.tight_layout()
    pdf.savefig()
    plt.close()

### NOTE:  some plots cannot be included in a subplot. For instance clustermap

