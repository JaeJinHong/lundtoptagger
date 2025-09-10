import os
from typing import Union
import math

import yaml
import uproot
import awkward as ak
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from scipy.stats import entropy, gaussian_kde

from ..GNN_model_weight.models import mdn_loss, mdn_loss_new

class MyDataset(torch.utils.data.Dataset): # Lazyload the shuffled training set
    def __init__(self, data_dir='data_dir'):
        self.data_dir = data_dir
        self.data_files = sorted(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        # Construct the full path to the file
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        
        # Load the graph data from the file
        data = load_file(file_path)
        
        return data

    def __len__(self):
        return len(self.data_files)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def 

def train_multi(loader, model, device, optimizer, epoch):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        #print("batch_counter: ",batch_counter, end="\r")
        if len(data)<1024:
            continue
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        output = model(data)
        # Change labels to shape (batch_size, 1) for cross_entropy
        new_y = to_categorical(data.y, num_classes=4)
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        loss = F.cross_entropy(output, new_y, weight = new_w)
        loss.backward()
        loss_all += data.num_graphs * loss.item()

        optimizer.step()

    del data
    data = []
    torch.cuda.empty_cache()
    return loss_all / len(loader.dataset)


@torch.no_grad()
def get_accuracy_multi(loader, model, device):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data.to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        output = model(cl_data)
        pred = F.Softmax(output).max(dim=1)
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def test_multi(loader, model, device):
    model.eval()
    #print("init my_test()")
    #time.sleep(600)
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        #print("batch_counter: ",batch_counter, end="\r")
        data = data.to(device)
        output = model(data)
        # Change labels to shape (batch_size, 1) for cross_entropy
        new_y = to_categorical(data.y, num_classes=4)
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1))
        loss = F.binary_cross_entropy(output, new_y, weight=new_w)
        loss_all += data.num_graphs * loss.item()
    del data
    data = []
    torch.cuda.empty_cache()
    return loss_all/len(loader.dataset)

