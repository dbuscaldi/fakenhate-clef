# -*- coding: utf-8 -*-
import torch
from torch import nn, cuda
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import re
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil, sys
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix as mcm, classification_report
from utils import RobertaClassifier, HateSpeech_Dataset


df = pd.read_csv('/content/drive/MyDrive/Full_data_merged.csv')

df.columns

X = df[['concatenated_text']]
X.rename(columns={'concatenated_text':'Text'}, inplace=True)
Y = df[['Violence_dic', 'Discrimination_dic']]

labels = []
labels_name = []

for row in Y.values:
  if row[1] == 1 and row[0] == 0:
    labels.append(2)
    labels_name.append("Discrimination")
  else:
    labels.append(1)
    labels_name.append("Violence")

dic = {"Text":[], "Label names":[]}

for index, row in df.iterrows():
  text = row["concatenated_text"]
  violence = row["Violence_dic"]
  discrimination = row["Discrimination_dic"]

  if violence == 1:
    dic["Text"].append(text)
    dic["Label names"].append("Violence")

  if discrimination == 1:
    dic["Text"].append(text)
    dic["Label names"].append("Discrimination")

new_df = pd.DataFrame(dic)
print(new_df["Text"][0])

Y['Labels'] = labels

X['Labels'] = Y['Labels']
X['Label_names'] = labels_name
X

X["Label_names"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

T = X
T["Words Per Comment"] = T["Text"].str.split().apply(len)
T.boxplot("Words Per Comment", by="Label_names", grid=False, showfliers=False,
           color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()


# ValueError: Target size (torch.Size([32])) must be the same as input size (torch.Size([32, 3]))
# Need to do it after train/test split, otherwise, there will be data leakage

def one_hot_encode(target):
  encoder = OneHotEncoder(sparse=False)
  target_reshaped = [[t] for t in target]
  target_encoded = encoder.fit_transform(target_reshaped)

  return target_encoded

MAX_LEN = 16
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
device = 'cuda' if cuda.is_available() else 'cpu'

train_size = 0.8
train_dataset = new_df.sample(frac=train_size,random_state=200).reset_index(drop=False)
val_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
train_target_encoded = one_hot_encode(train_dataset['Label names'])
val_target_encoded = one_hot_encode(val_dataset['Label names'])

print("FULL Dataset: {}".format(X.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(val_dataset.shape))

training_set = HateSpeech_Dataset(train_dataset, train_target_encoded, tokenizer, MAX_LEN)
validation_set = HateSpeech_Dataset(val_dataset, val_target_encoded, tokenizer, MAX_LEN)

model = RobertaClassifier()
model.to(device)

def loss_fn(outputs, targets):
  return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

def load_ckp(checkpoint_fpath, model, optimizer):

    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training

    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])

    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']

    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **val_params)
val_targets = []
val_outputs = []

def save_ckp(state, is_best, checkpoint_path, best_model_path):

    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path

    torch.save(state, f_path)
    # if it is a best model, min validation loss

    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def train_model(start_epochs, n_epochs, valid_loss_min_input,
                training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path):

  # initialize tracker for minimum validation loss
  valid_loss_min = valid_loss_min_input
  writer = SummaryWriter()
  for epoch in range(start_epochs, n_epochs+1):
    train_loss = 0
    valid_loss = 0
    model.train()

    print('############# Epoch {}: Training Start   #############'.format(epoch))

    for batch_idx, data in enumerate(training_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

    print('############# Epoch {}: Training End     #############'.format(epoch))
    print('############# Epoch {}: Validation Start   #############'.format(epoch))

    ######################
    # validate the model #
    ######################

    model.eval()
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))

      # calculate average losses
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      writer.add_scalar("loss x epoch", train_loss, epoch)

      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))

      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }

      # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)

      # save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss
    print('############# Epoch {} Done   #############\n'.format(epoch))
  writer.close()
  return model

checkpoint_path = '/content/drive/MyDrive/UPV/ProjetStage/current_checkpoint.pt'

best_model = '/content/drive/MyDrive/UPV/ProjetStage/best_model.pt'

trained_model = train_model(1, 25, np.Inf, training_loader, validation_loader, model,
                      optimizer,checkpoint_path,best_model)

val_preds = (np.array(val_outputs) > 0.5).astype(int)
val_preds

accuracy = metrics.accuracy_score(val_targets, val_preds)
f1_score_micro = metrics.f1_score(val_targets, val_preds, average='micro')
f1_score_macro = metrics.f1_score(val_targets, val_preds, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")

cm_labels = ['Violence', 'Discrimination', 'Discrimination_and_violence']
cm = mcm(val_targets, val_preds)
print(classification_report(val_targets, val_preds))

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir runs
