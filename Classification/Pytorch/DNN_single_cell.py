import sys
import scanpy as sc
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy import stats
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
#from keras.utils import np_utils

filename = 'filepath'
# load dataset
adata = sc.read(filename)
print(adata)
data = adata.X

dataframe = adata.to_df()
# load labels
labels_dataframe = adata.obs.reset_index()
#features_dataframe = adata.var['features-1']
dataframe['cell_type'] = dataframe.index

'''
print(dataframe['cell_type'])
# Convert to csc file
#data_csc_mat = data.tocsc()

# Convert sparse dataset
#dataframe = pd.DataFrame.sparse.from_spmatrix(data_csc_mat, columns = features_dataframe)
#data_csc_dense = data_csc_mat.toarray()
#dataframe = pd.DataFrame(data_csc_dense, columns = features_dataframe)
#print(dataframe.head())
#sys.exit()
#print(labels_dataframe['cell_type'])
#dataframe = pd.concat([dataframe, labels_dataframe['cell_type']], axis = 1)
#print(dataframe.head())
#sys.exit()
#print("Raw Matrix shape",dataframe.head())

#dataframe = dataframe.drop(columns= ['index','celltype','nFeature_RNA','nCount_RNA','barcode','barcodes', 'tissue','cluster','stage', 'batch','study' ])

print(dataframe.shape)
print(len(np.unique(dataframe['cell_type'])))
print(len(dataframe['cell_type'].value_counts().to_list()))
'''

print(len(np.unique(labels_dataframe['cell_type'])))

# preprocessing
def data_preprocessing(dataframe, checkNA= True ,  OHE = True):
  if checkNA:
    dataframe_col = np.shape(dataframe)[1]
    # drop columns with all zeros
    dataframe = dataframe.loc[:, (dataframe != 0).any()]
    # drop duplicates
    #dataframe = dataframe.drop_duplicates()
    # drop donor type and tissue type
    # dataframe = dataframe.drop(columns = ['Barcode'])
    # drop rows with NAN
    dataframe = dataframe[dataframe['cell_type'].notna()]
    new_dataframe_col = np.shape(dataframe)[1]
    print("{} columns dropped".format(( dataframe_col - new_dataframe_col)))
    # get label
    temp = dataframe.pop('cell_type')
    label = labels_dataframe['cell_type']

  if OHE:
    # Dictionary for key value pair of labels
    num_classes = np.unique(label)
    le = preprocessing.LabelEncoder()
    le.fit(label)
    # print(list(le.classes_))
    label = le.transform(label)
    # print("label:{}".format(label))

  return dataframe, label

dataframe, label = data_preprocessing(dataframe)
print(dataframe.shape)
print(label)
# Data normalization
# # Sklearn minmax scaler

def normal_dataframe(dataframe , norm_type  , normalize = False):
    if normalize == True:
        if norm_type =='min_max':
            min_max_scaler = preprocessing.MinMaxScaler()
            dataframe_scaled = min_max_scaler.fit_transform(dataframe)
            return dataframe_scaled

        elif norm_type == 'Stand_scaler':
            scaler = preprocessing.StandardScaler()
            dataframe_scaled = scaler.fit_transform(dataframe)
            return dataframe_scaled

        elif norm_type == 'zscore':
            # data_stats = dataframe.describe()
            # print(data_stats)
            # dataframe_scaled = (dataframe - data_stats['mean'])/ dataframe['std']
            dataframe_scaled = stats.zscore(dataframe)
            return dataframe_scaled

        elif norm_type == 'MaxAbsScaler':
            max_abs_scaler = preprocessing.MaxAbsScaler()
            dataframe_scaled = max_abs_scaler.fit_transform(dataframe)
            return dataframe_scaled

        elif norm_type == 'log_magnitude':
            dataframe_scaled = dataframe.apply(lambda x : np.log(x+1))
            return dataframe_scaled.to_numpy()

        elif norm_type == 'log_normalize':
            dataframe_scaled = dataframe.apply(lambda x : np.log(x+1))
            nlise = preprocessing.Normalizer()
            dataframe_normalized = nlise.fit_transform(dataframe_scaled)
            return dataframe_normalized

        elif norm_type == 'normalize':
            dataframe_scaled = preprocessing.normalize(dataframe)
            return dataframe
    else:
        return dataframe.to_numpy()

#dataframe_scaled = normal_dataframe(dataframe ,'MaxAbsScaler', True)
dataframe_scaled = dataframe
print("After normalization",dataframe_scaled.shape)
#sys.exit()

#dataframe = dataframe.toarray()
#Test train Split
# dataframe = dataframe.drop(['gene_id'],axis = 1)
print("Starting Spliting the data")
train, test , Y_train , Y_test = train_test_split(dataframe_scaled, label ,test_size = 0.1)
train , val , Y_train , Y_val = train_test_split(train ,Y_train , test_size = 0.1)
print("trainig data:{}, label:{}".format(np.shape(train), np.shape(Y_train)))
print("Validation data:{}, label:{}".format(np.shape(val), np.shape(Y_val)))
print("Testing data:{}, label:{}".format(np.shape(test), np.shape(Y_test)))
# print("Data Sample:{}".format(train[0]))

def label_encoder(label):
  # encode class values as integers
  # encoder = LabelEncoder()
  # encoder.fit(label)
  # encoded_Y = encoder.transform(label)
  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_y = np_utils.to_categorical(label, num_classes=224)
  return dummy_y


# creating label one hot encoding
#train_label_oh = label_encoder(Y_train)
#test_label_oh = label_encoder(Y_test)
#val_label_oh = label_encoder(Y_val)


# Model paramters
Epochs = 50
Batch_size = 32
Learning_rate = 0.0001

print("creating data loader")
# Data Loader


class Traindata(Dataset):
  def __init__(self, train, Y_train):
    self.X_data = train
    self.y_data = Y_train

  def __getitem__(self, index):
    return self.X_data[index], self.y_data[index]

  def __len__(self):
    return len(self.X_data)

class Testdata(Dataset):
  def __init__(self, test):
    self.X_data = test

  def __getitem__(self, index):
    return self.X_data[index]

  def __len__(self):
    return len(self.X_data)


train_data = Traindata(torch.FloatTensor(train.values), torch.FloatTensor(Y_train))
valid_data = Traindata(torch.FloatTensor(val.values), torch.FloatTensor(Y_val))
test_data = Testdata(torch.FloatTensor(test.values))

# Load dataset
train_loader = DataLoader(dataset = train_data, batch_size = Batch_size, shuffle = True)
valid_loader = DataLoader(dataset = valid_data, batch_size = 1, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = 1)


# Create Neural Net architecture
print("Starting loading model")

class SingleCellClassifier(nn.Module):
  def __init__(self):
    super(SingleCellClassifier, self).__init__()

    # Layer
    self.layer_input = nn.Linear(23166, 5000)
    self.layer_1 = nn.Linear(5000, 1000)
    self.layer_2 = nn.Linear(1000, 500)
    self.layer_out = nn.Linear(500, 729)
    self.relu = nn.ReLU()

    self.dropout = nn.Dropout(p = 0.01)

  def forward(self, inputs):
    x = self.relu(self.layer_input(inputs))
    x= self.relu(self.layer_1(x))
    x = self.dropout(x)
    x = self.relu(self.layer_2(x))
    x = self.layer_out(x)

    return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



model = SingleCellClassifier()
model.to(device)

print(model)

# loss and optimizer
criterion  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = Learning_rate)

# Evaluation
def multi_acc( y_pred, y_test):
  y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
  _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

  correct_pred = (y_pred_tags == y_test).float()
  acc = correct_pred.sum() / len(correct_pred)

  acc = torch.round(acc) * 100

  return acc

accuracy_stats = {'train':[],
                 'val':[]}

loss_stats = {'train':[],
              'val':[]}

# Training

print("Begin Training")

for epoch in tqdm(range(1, Epochs)):
  # Start
  train_epochs_loss = 0
  train_epochs_acc = 0

  model.train()

  for X_train_batch, y_train_batch in train_loader:
    X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
    optimizer.zero_grad()

    y_train_pred = model(X_train_batch)

    # optimizer
    train_loss = criterion(y_train_pred, y_train_batch.long())
    train_acc = multi_acc(y_train_pred, y_train_batch)

    train_loss.backward()
    optimizer.step()

    train_epochs_loss += train_loss.item()
    train_epochs_acc += train_acc.item()

  # Validation

  with torch.no_grad():

    val_epochs_loss= 0
    val_epochs_acc = 0

    model.eval()

    for X_val_batch, y_val_batch in valid_loader:
      X_val_batch, y_val_batch= X_val_batch.to(device), y_val_batch.to(device)
      y_val_pred = model(X_val_batch)

      val_loss = criterion(y_val_pred, y_val_batch.long())
      val_acc = multi_acc(y_val_pred, y_val_batch)

      val_epochs_loss += val_loss.item()
      val_epochs_acc += val_acc.item()

  loss_stats['train'].append(train_epochs_loss/len(train_loader))
  loss_stats['val'].append(val_epochs_loss/len(valid_loader))
  accuracy_stats['train'].append(train_epochs_acc/len(train_loader))
  accuracy_stats['val'].append(val_epochs_acc/len(valid_loader))

  print(f'Epoch {epoch+0:03}: | Train Loss: {train_epochs_loss/len(train_loader):.5f} | Val Loss: {val_epochs_loss/len(valid_loader):.5f} | Train Acc: {train_epochs_acc/len(train_loader):.3f}| Val Acc: {val_epochs_acc/len(valid_loader):.3f}')



# Testing  Model 
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
        # print(y_pred_list[0])
        # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

print(classification_report(Y_test, y_pred_list))

