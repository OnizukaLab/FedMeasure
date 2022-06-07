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
import json
import re



class LocalDataset(torch.utils.data.Dataset):
  def __init__(self,dataset,worker_id):
    self.dataset = dataset
    self.id = worker_id

  def __getitem__(self, index):
    data = self.dataset['x'][index]
    target = self.dataset['y'][index]
    return data,target

  def __len__(self):
    return len(self.dataset['x'])



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


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_dataset(args, Centralized=False,unlabeled_data=False):
    train_path = '../data/sent140/data/train'
#    train_path = './data/train'
    test_path = '../data/sent140/data/test'
#    test_path = './data/test'
    clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
    
    for c in dataset_train.keys():
        dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
        dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))
        
    all_federated_trainset = []
    all_federated_testset = []
    for i,dataset in enumerate(dataset_train.values()):
        all_federated_trainset.append(LocalDataset(dataset,i))
    for i,dataset in enumerate(dataset_test.values()):
        all_federated_testset.append(LocalDataset(dataset,i))
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
                unlabeled_dataset.data = unlabeled_dataset.data + all_federated_trainset[i].dataset['x']
        unlabeled_dataset,_ = torch.utils.data.random_split(unlabeled_dataset, [args.unlabeleddata_size, len(unlabeled_dataset)-args.unlabeleddata_size])

    
    if Centralized and unlabeled_data:
        return federated_trainset,federated_valset,federated_testset,global_trainloader,global_valloader,global_testloader,unlabeled_dataset
    elif Centralized:
        return federated_trainset,federated_valset,federated_testset,global_trainloader,global_valloader,global_testloader
    elif unlabeled_data:
        return federated_trainset,federated_valset,federated_testset,unlabeled_dataset
    else:
        return federated_trainset,federated_valset,federated_testset