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


# Load libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np


################
## Clustering ##
################

## Load up real-world data into pandas
# Using data from GSE79731 on GEO:  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE79731
# A study of Mycobacterium tuberculosis infection

# Load phenotype data
phenos = pd.read_csv('data/phenos.csv', header = 0, index_col = 0)

# Load gene expression data
gexp = pd.read_csv('data/GSE79731_series_matrix.csv', header = 0, index_col = 0)

# Load gene info - separated by tabs not commas, and index_col is set to 1 not 0
# https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/
gene_info = pd.read_csv('data/Mus_musculus.gene_info', sep='\t', header = 0, index_col = 1)

# Translate Entrez gene ids to gene symbols
gexp2 = gexp.loc[gexp.index.isin(gene_info.index)]
gexp2.index = gene_info.loc[gexp2.index,'Symbol']

# Select top most variant genes
top2000 = gexp2.var(axis=1).sort_values(ascending=False).index[range(2000)]

# Return an eigengene for a gene expression data given a set of genes
def getEigengene(gexp, genes):
    pca = PCA(n_components=1)
    gexp_pca = pd.DataFrame(pca.fit(gexp.loc[genes].T).transform(gexp.loc[genes].T), index = gexp.columns)
    eigengene = gexp_pca[0]
    if sum([stats.pearsonr(gexp.loc[i],eigengene)[0] for i in genes])/len(genes) > 0:
        return eigengene
    else:
        return -eigengene

# Scaling
tmp = StandardScaler().fit_transform(gexp2.loc[top2000])

## Cluster using kMeans
sil_km = []
with PdfPages('km_silhouettes.pdf') as pdf:
    for i in range(2,10):
        n_clusters = i
        km1 = KMeans(n_clusters=i).fit(tmp)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        sil_km.append(silhouette_score(tmp, km1.labels_))

        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        #fig.set_size_inches(7, 7)

        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(tmp) + (n_clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tmp, km1.labels_)

        y_lower = 10
        for j in range(i):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            jth_cluster_silhouette_values = \
                sample_silhouette_values[km1.labels_ == j]

            jth_cluster_silhouette_values.sort()

            size_cluster_j = jth_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, jth_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=sil_km[i-2], color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Save figure to pdf
        pdf.savefig(fig)
        plt.close()

    # Save plots to pdf
    fig = plt.figure()
    plt.plot(range(2,10),sil_km)
    plt.xticks(range(2,10))
    plt.xlabel('Number of clsuters')
    plt.ylabel('Average sihouette score')
    pdf.savefig(fig)
    plt.close()

# Chose k = 6
km1 = KMeans(n_clusters = 6).fit(tmp)
print(km1.labels_)
eigengenes = pd.concat([getEigengene(gexp2.loc[top2000], top2000[km1.labels_==i]) for i in range(len(set(km1.labels_)))], axis = 1)
eigengenes.columns = range(len(set(km1.labels_)))

# Make column and row colors
colors = {'0h':'#fee5d9', '4h':'#fcae91', '8h':'#fb6a4a', '24h':'#de2d26', '48h':'#a50f15'}
colRow_colors = [colors[i] for i in phenos.loc['timepoints']]

# Plot clustermap
sns.clustermap(eigengenes.T, cmap = sns.color_palette("vlag",n_colors=33), col_colors=colRow_colors, col_cluster=False)
plt.show()

# Make it into a PDF
with PdfPages('GSE79731_clustered_KMeans_6.pdf') as pdf:
    # Plot clustermap
    sns.clustermap(eigengenes.T, cmap=sns.color_palette('vlag', n_colors=33), col_colors=colRow_colors, col_cluster=False)
    pdf.savefig()
    plt.close()


## Cluster using SpectralClustering
sil_gmm = []
with PdfPages('gmm_silhouettes.pdf') as pdf:
    for i in range(2,10):
        n_clusters = i
        gmm1 = GaussianMixture(n_components = i, covariance_type = 'full').fit(tmp)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        sil_gmm.append(silhouette_score(tmp, gmm1.predict(tmp)))

        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        #fig.set_size_inches(7, 7)

        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(tmp) + (n_clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(tmp, gmm1.predict(tmp))

        y_lower = 10
        for j in range(i):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            jth_cluster_silhouette_values = \
             sample_silhouette_values[gmm1.predict(tmp) == j]

            jth_cluster_silhouette_values.sort()

            size_cluster_j = jth_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, jth_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=sil_gmm[i-2], color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Save figure to pdf
        pdf.savefig(fig)
        plt.close()

    # Save plots to pdf
    fig = plt.figure()
    plt.plot(range(2,10),sil_gmm)
    plt.xticks(range(2,10))
    plt.xlabel('Number of clsuters')
    plt.ylabel('Average sihouette score')
    pdf.savefig(fig)
    plt.close()

# Chose k = 6
gmm1 = GaussianMixture(n_components = 6, covariance_type = 'full').fit(tmp)
print(gmm1.predict(tmp))
eigengenes = pd.concat([getEigengene(gexp2.loc[top2000], top2000[gmm1.predict(tmp)==i]) for i in range(len(set(gmm1.predict(tmp))))], axis = 1)
eigengenes.columns = range(len(set(gmm1.predict(tmp))))

# Make column and row colors
colors = {'0h':'#fee5d9', '4h':'#fcae91', '8h':'#fb6a4a', '24h':'#de2d26', '48h':'#a50f15'}
colRow_colors = [colors[i] for i in phenos.loc['timepoints']]

# Plot clustermap
sns.clustermap(eigengenes.T, cmap = sns.color_palette("vlag",n_colors=33), col_colors=colRow_colors, col_cluster=False)
plt.show()

# Make it into a PDF
with PdfPages('GSE79731_clustered_GaussianMixture_6.pdf') as pdf:
    # Plot clustermap
    sns.clustermap(eigengenes.T, cmap=sns.color_palette('vlag', n_colors=33), col_colors=colRow_colors, col_cluster=False)
    pdf.savefig()
    plt.close()



####################
## Classification ##
####################

## Breast Cancer Wisconsin (Diagnostic) Data Set
## Dataset description:  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
## Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

# Load up some helpful tools
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the breast cancer sklearn dataset
from sklearn.datasets import load_breast_cancer
d1 = load_breast_cancer()
imgData = d1['data']
imgData_scld = StandardScaler().fit_transform(imgData)
disState = d1['target_names'][d1['target']]

# Load up some classification algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

## Set up classification comparison
kFolds = 100
performanceResults = {'KNN':[],'SVM':[],'RF':[]}
for k in range(kFolds):
    # Split up data into train and test sets using random integer seed
    imgData_train, imgData_test, disState_train, disState_test = train_test_split(imgData_scld, disState, test_size=.4, random_state=random.randint(0,1000000))

    ## Train classifiers
    # k-nearest neighbors (KNN)
    knn_clf = KNeighborsClassifier(3)
    knn_clf.fit(imgData_train, disState_train)

    # Support vector machines (SVM)
    svm_clf = SVC(kernel='linear', C = 0.025)
    svm_clf.fit(imgData_train, disState_train)

    # Random Forest ensemble (RF)
    rf_clf = RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1)
    rf_clf.fit(imgData_train, disState_train)


    ## Test classifiers
    knn_pred = knn_clf.predict(imgData_test)
    performanceResults['KNN'].append(classification_report(disState_test, knn_pred, output_dict=True, zero_division=0))

    svm_pred = svm_clf.predict(imgData_test)
    performanceResults['SVM'].append(classification_report(disState_test, svm_pred, output_dict=True, zero_division=0))

    rf_pred = rf_clf.predict(imgData_test)
    performanceResults['RF'].append(classification_report(disState_test, rf_pred, output_dict=True, zero_division=0))


## What is a classification report
print(performanceResults['KNN'][0]) # First iteration of KNN classifier training and testing
print(performanceResults['KNN'][0].keys()) # Disease states plus a few more:  accuracy, macro avg, and weighted avg (weighted avg accounts for uneven class sampling)


## Build figure to describe classifiers: using 'f1-score'
# f1 score = 2/(1/recall+1/precision) = tp/(tp + 0.5 * (fp + fn))
# tp = true positive, fp = false positive, fn = false negative
results = {'KNN':{},'SVM':{},'RF':{}}
for disState in ['benign','malignant','weighted avg']: # ignoring macro average
    for clf in results.keys():
        results[clf][disState] = sum([performanceResults[clf][i][disState]['f1-score'] for i in range(kFolds)])/kFolds

# Print out the matrix of predictive power for classifiers
print(pd.DataFrame(results))

