import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import random
from torch.autograd import Variable
import copy
from torch import nn, optim
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv
import time
import math



class LocalDataset(torch.utils.data.Dataset):
  def __init__(self,dataset,worker_id):
    self.data = []
    self.target = []
    self.id = worker_id
    for i in range(len(dataset)):
      self.data.append(dataset[i][0][0])
      self.target.append(dataset[i][1][0])

  def __getitem__(self, index):
    return self.data[index],self.target[index]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index],self.target[index]

  def __len__(self):
    return len(self.data)



class UnlabeledDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.data = []
    self.target = None

  def __getitem__(self, index):
    return self.data[index],'unlabeled'

  def __len__(self):
    return len(self.data)



class GlobalDataset(torch.utils.data.Dataset):
  def __init__(self,federated_dataset):
    self.data = []
    self.target = []
    for dataset in federated_dataset:
      for (data,target) in dataset:
        self.data.append(data)
        self.target.append(target)

  def __getitem__(self, index):
    return self.data[index],self.target[index]

  def __len__(self):
    return len(self.data)





def get_dataset(args, Centralized=False,unlabeled_data=False):
    with open('../../data/federated_trainset_shakespeare.pickle', 'rb') as f:
        all_federated_trainset = pickle.load(f)
    with open('../../data/federated_testset_shakespeare.pickle', 'rb') as f:
        all_federated_testset = pickle.load(f)
    all_worker_num = len(all_federated_trainset)
    
    worker_id_list = random.sample(range(all_worker_num),args.worker_num)
    print(worker_id_list)
    federated_trainset = []
    federated_testset = []
    for i in worker_id_list:
        federated_trainset.append(all_federated_trainset[i])
        federated_testset.append(all_federated_testset[i])
        
    federated_valset = [None]*args.worker_num
    for i in range(args.worker_num):
        n_samples = len(federated_trainset[i])
        if n_samples==1:
            federated_valset[i] = copy.deepcopy(federated_trainset[i])
        else:
            train_size = int(len(federated_trainset[i]) * 0.7) 
            val_size = n_samples - train_size 
            federated_trainset[i],federated_valset[i] = torch.utils.data.random_split(federated_trainset[i], [train_size, val_size])     
   
    ## get global dataset
    if Centralized:
        global_trainset = GlobalDataset(federated_trainset)
        global_valset = GlobalDataset(federated_valset)
        global_testset =  GlobalDataset(federated_testset)
        
        global_trainloader = torch.utils.data.DataLoader(global_trainset,batch_size=args.batch_size,shuffle=True,num_workers=2)
        global_valloader = torch.utils.data.DataLoader(global_valset,batch_size=args.test_batch,shuffle=False,num_workers=2)
        global_testloader = torch.utils.data.DataLoader(global_testset,batch_size=args.test_batch,shuffle=False,num_workers=2)
        
        
    ## get unlabeled dataset
    if unlabeled_data:
        unlabeled_dataset = UnlabeledDataset()
        for i in range(all_worker_num):
            if i not in worker_id_list:
                unlabeled_dataset.data = unlabeled_dataset.data + all_federated_trainset[i].data
        unlabeled_dataset,_ = torch.utils.data.random_split(unlabeled_dataset, [args.unlabeleddata_size, len(unlabeled_dataset)-args.unlabeleddata_size])

    
    if Centralized and unlabeled_data:
        return federated_trainset,federated_valset,federated_testset,global_trainloader,global_valloader,global_testloader,unlabeled_dataset
    elif Centralized:
        return federated_trainset,federated_valset,federated_testset,global_trainloader,global_valloader,global_testloader
    elif unlabeled_data:
        return federated_trainset,federated_valset,federated_testset,unlabeled_dataset
    else:
        return federated_trainset,federated_valset,federated_testset