##########################################################
## ccAFv2: Building ccAFv2                              ##
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
## @Author:  Chris Plaisier, Kara Ramirez               ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################


#----------------------
# Import pacakges
#----------------------

# Import pytorch
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Standard pacakges
import scanpy as sc
import pandas as pd
import numpy as np
import json

# For data handling
from scipy.sparse import issparse as isspmatrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# For plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages

# For metrics
from torchmetrics.classification import MulticlassAUROC
from torcheval.metrics import MulticlassConfusionMatrix


#----------------------
# Functions
#----------------------

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

# Scale data same way in R and Python
def _scale(df1):
    """
    scale takes in a pandas dataframe and applies scales the values into Z-scores across rows.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame of scRNA-seq data to be scaled.

    Returns
    -------
    pd.DataFrame
        DataFrame of scRNA-seq data that has been scaled.
    """
    return (df1.subtract(df1.mean(axis=1),axis=0)).div(df1.std(axis=1),axis=0)

# Prepare data (unchanged)
def prep_data_sets(data):
    data.var_names_make_unique()
    if isspmatrix(data.X):
        df = pd.DataFrame(data.X.todense(), index=data.obs_names, columns=data.var_names)
    else:
        df = pd.DataFrame(data.X, index=data.obs_names, columns=data.var_names)
    df_scaled = _scale(df)
    return df_scaled

# Compute ECE
def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE).
    """
    # Get the true probabilities and predicted probabilities for each bin
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Calculate the number of samples in each bin
    # np.histogram provides the counts (bin_counts) and the bin edges
    bin_counts = np.histogram(y_prob, bins=np.linspace(0.0, 1.0, n_bins + 1))[0]

    # Filter out bins with zero samples to avoid division by zero or incorrect weighting
    nonzero_bins = bin_counts > 0
    prob_true = prob_true[nonzero_bins]
    prob_pred = prob_pred[nonzero_bins]
    bin_counts = bin_counts[nonzero_bins]

    # Calculate the weighted absolute difference (ECE formula)
    # ECE = sum across all bins of |prob_true - prob_pred| * (samples in bin / total samples)
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts / len(y_true)))

    return ece

# Run model to predict labels later
def predict(model, X, label_encoder=None):
    model.eval()
    with torch.inference_mode():
        logits = model(X)
        preds_idx = torch.argmax(logits, dim=1)

    if label_encoder is not None:
        return label_encoder.inverse_transform(preds_idx.cpu().numpy())

    return preds_idx.cpu().numpy()


#----------------------
# Load data
#----------------------

sets = {'U5':'U5_final_states_meta_data.csv'}
savedir = 'output'


#-----------------------------------------------
# Load data and subset genes to top marker genes
#-----------------------------------------------

# create directory 'output' beforehand

datasets = {}
data_sets_scaled = {}
label_encoder = LabelEncoder()
labelsets = {}
for set1 in sets:
    # Load in normalized U5-hNSC data
    print('\nLoading '+set1+' scRNA-seq data...')
    datasets[set1] = sc.read_loom('data/'+set1+'_normalized_ensembl.loom')
    datasets[set1].obs['dataset'] = set1
    meta_data = pd.read_csv('data/'+sets[set1],header=0, index_col=0)
    datasets[set1].obs['labels'] = meta_data['final_cc_states']
    datasets[set1].obs['labels'] = [i if i!='Neural G0' else 'G0' for i in datasets[set1].obs['labels']]
    label_encoder.fit(datasets[set1].obs['labels'])
    label_mapping = {int(idx): str(cls) for idx, cls in enumerate(label_encoder.classes_)}
    with open(savedir + f'/label_encoder_mapping_{set1}_031626_tenpercent.json', 'w') as f:
        json.dump(label_mapping, f) # saving labels to use in R Torch later
    labelsets[set1] = label_encoder.transform(datasets[set1].obs['labels'])
    # Subset marker genes
    print('\nSubsetting data genes to marker genes...')
    mgenes = list(set(pd.read_csv('data/ccAFv2_genes.csv', header = 0, index_col = 0)['human_ensembl']))
    mg1 = mgenes
    # mg1 = list(set(mg1).intersection(datasets[set1].var_names))
    mg1 = [g for g in mgenes if g in datasets[set1].var_names] # preserves order of genes
    print(set1+' marker genes: '+str(len(mg1)))
    # Subset data to marker genes
    datasets[set1] = datasets[set1][:,mg1].copy()
    # Remove uninformative genes
    sc.pp.filter_genes(datasets[set1], min_cells=1)
    # Scale dataset
    data_sets_scaled[set1] = prep_data_sets(datasets[set1])

# Prep data for training
X = torch.from_numpy(data_sets_scaled['U5'].to_numpy()).type(torch.float)
y = torch.from_numpy(labelsets['U5']).type(torch.long)

# Split into test and train
#  - Using stratify = y, stratifies with same representation for each group
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify = y, random_state = 42)

# Print out the lengths for the training and test datasets
len(X_train), len(X_test), len(y_train), len(y_test)


#-----------------------------------
# Parameterize NN model
#-----------------------------------

# Replicate ccAFv2 with nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=861, out_features=861),
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=861, out_features=861),
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=861, out_features=7)
)

# Create a loss function
criterion = nn.CrossEntropyLoss()

# Create an optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Set random seed, to make sure we all get same answers.
#  - Note: the seed would not be set like this for real operations.
torch.manual_seed(42)

# Set the number of epochs
num_epochs = 100

# Dataloader for training data
train_dataset = TensorDataset(X_train, y_train.long())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# Help for choosing batch size and number of epochs:
#     https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model

# Testing data
X_test, y_test = X_test, y_test.long()


#-----------------------------------
# Build and train NN model
#-----------------------------------

# Capture training metrics as the are produced
training_metrics = []

for epoch in range(num_epochs):
    ### Training - Turn on training mode
    tmp1 = model.train()
    for batch_idx, (inputs_batch, targets_batch) in enumerate(train_loader):
        # 1. Zero gradients
        optimizer.zero_grad()
        # 2. Forward pass
        logits = model(inputs_batch)
        loss = criterion(logits, targets_batch)
        # 3. Backward
        loss.backward()
        # 4. Optimizer step
        optimizer.step()
        # 5. Compute metrics
        y_pred = torch.argmax(torch.softmax(logits, dim = 1), dim=1) # turn logits -> pred probs -> pred labels
        acc = accuracy_fn(y_true=targets_batch, y_pred=y_pred)
    ### Testing
    tmp2 = model.eval()
    with torch.inference_mode():
        # 1. Forward pass with test data
        test_logits = model(X_test)
        test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)
        # 2. Calculate loss/accuracy for test data
        test_loss = criterion(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        # 3. Print out what's happening every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%')
        # 4. Save out training for every epoch
        training_metrics.append({'Epoch': epoch+1, 'Training loss': float(loss), 'Training accuracy': acc, 'Test loss': float(test_loss), 'Test accuracy': test_acc})


#-----------------------------------
# Plot metrics
#-----------------------------------

# Plot the training metrics
tm_df = pd.DataFrame(training_metrics)
with PdfPages(savedir + '/' + 'plot_training_metrics.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8, 5))
    # 1. Plot the two lines with labels and styles
    ax.plot(tm_df['Epoch'], 100 - tm_df['Training accuracy'], label='Training loss', color='green', linestyle='-')
    ax.plot(tm_df['Epoch'], 100 - tm_df['Test accuracy'], label='Test loss', color='black', linestyle='-')
    ax.axhline(y=20, color='red', linestyle='--')
    # 2. Add labels, a title, and a legend for clarity
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('Training ccAFv2 model')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 100)
    # 3. Display the plot
    pdf.savefig()
    plt.close()

# Also write the training metrics to a file
pd.DataFrame(training_metrics).to_csv(savedir+ '/' + 'training_metrics.csv', index=False)


#-----------------------------------
# Save our trained NN model
#-----------------------------------

# Save weights & model
torch.save(model.state_dict(), savedir+ '/' + f'model_weights.pth')
torch.save(model, savedir+ '/' + f'model.pth')


#---------------------------------------
# Calculate metrics
#---------------------------------------

# Cell cycle states mapping to class numbers
cc_states = ['G0', 'G1', 'G2/M', 'Late G1', 'M/Early G1', 'S', 'S/G2']

# Compute raw logits once
with torch.inference_mode():
    test_logits = model(X_test)
    test_probas = torch.softmax(test_logits, dim=1) # turns model's confidence scores into probabilitites

# Confusion matrix before calibration
preds = torch.argmax(test_probas, dim=1) # argmax returns the indices with the max value- we convert our model outputs probabilities and then call for the highest probability i.e. our predicted class
confmat_metric = MulticlassConfusionMatrix(num_classes=7)
confmat_metric.update(preds, y_test)
confmat = confmat_metric.compute()

# Labels
row_labels = ['True: ' + c for c in cc_states]
col_labels = ['Pred: ' + c for c in cc_states]

# Confusion matrix
confmat_df = pd.DataFrame(
    confmat.numpy(),
    index=row_labels,
    columns=col_labels)
print(f"\nConfusion Matrix BEFORE calibration: ")
print(confmat_df)

pd.DataFrame(confmat_df).to_csv(savedir+ '/' + 'confusion_matrix_before_calib.csv')

# AUROC metric
auroc_metrics = MulticlassAUROC(num_classes=7, average=None)

# ---- Compute AUROC before calibration ----
aurocs = auroc_metrics(test_probas, y_test)
print(dict(zip(cc_states,aurocs)))


#-------------------------------
# UMAP
#-------------------------------

# ccAFv2 colors
ccAFv2_colors = {
    "G0": "#d9a428",
    "G1": "#f37f73",
    "Late G1": "#1fb1a9",
    "S": "#8571b2",
    "S/G2": "#db7092",
    "G2/M": "#3db270",
    "M/Early G1": "#6d90ca",
    "Unknown": "#D3D3D3",
    "G0/G1": "#FF6600"
    }

# Visualize cells in 2D
for set1 in datasets:

    print(f'\nPredicting labels for {set1}...')

    # Convert scaled data to torch tensor
    X_full = torch.from_numpy(
        data_sets_scaled[set1].to_numpy()
    ).float()

    # Predict the states
    preds_labels = predict(model, X_full, label_encoder)

    # Store predictions
    datasets[set1].obs['Predicted'] = preds_labels
    datasets[set1].obs['Predicted'] = datasets[set1].obs['ccAFv2'].astype('category')
    datasets[set1].obs['Original Labels'] = datasets[set1].obs['labels'].astype('category')

    # Run PCA/UMA{}
    sc.tl.pca(datasets[set1], svd_solver='arpack')
    sc.pp.neighbors(datasets[set1], n_neighbors=15, n_pcs=40)
    sc.tl.umap(datasets[set1])


    pdf_filename = savedir + '/' + f'UMAP_predicted_and_labels_{set1}_umap.pdf'

    with PdfPages(pdf_filename) as pdf:

        # Predicted
        sc.pl.umap(
            datasets[set1],
            color = ['Predicted','Original Labels'],
            palette=[ccAFv2_colors.get(cls, "#000000") for cls in datasets[set1].obs['ccAFv2'].cat.categories],
            show=False
            )
        pdf.savefig()
        plt.close()

