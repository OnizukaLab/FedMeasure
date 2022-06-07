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
import re
import json




def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
class Early_Stopping():
  def __init__(self,partience):
    self.step = 0
    self.loss = float('inf')
    self.partience = partience

  def validate(self,loss):
    if self.loss<loss:
      self.step += 1
      if self.step>self.partience:
        return True
    else:
      self.step = 0
      self.loss = loss

    return False


class KMeans(object):
    """KMeans 法でクラスタリングするクラス"""

    def __init__(self, n_clusters=2, max_iter=300):
        """コンストラクタ

        Args:
            n_clusters (int): クラスタ数
            max_iter (int): 最大イテレーション数
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.cluster_centers_ = None

    def fit_predict(self, features):
        """クラスタリングを実施する

        Args:
            features (numpy.ndarray): ラベル付けするデータ

        Returns:
            numpy.ndarray: ラベルデータ
        """
            
        # 要素の中からセントロイド (重心) の初期値となる候補をクラスタ数だけ選び出す
        feature_indexes = np.arange(len(features))
        np.random.shuffle(feature_indexes)
        initial_centroid_indexes = feature_indexes[:self.n_clusters]
        self.cluster_centers_ = features[initial_centroid_indexes]

        # ラベル付けした結果となる配列はゼロで初期化しておく
        pred = np.zeros(features.shape)
        

        # クラスタリングをアップデートする
        for _ in range(self.max_iter):

            # 各特徴ベクトルから最短距離となるセントロイドを基準に新しいラベルをつける
            new_pred = np.array([
                np.array([
                    self.Euclidean_distance(p, centroid)
                    for centroid in self.cluster_centers_
                ]).argmin()
                for p in features
            ])

            if np.all(new_pred == pred):
                # 更新前と内容を比較して、もし同じなら終了
                break

            pred = new_pred
            
            # 各クラスタごとにセントロイド (重心) を再計算する
            self.cluster_centers_ = np.array([features[pred == i].mean(axis=0)
                                              for i in range(self.n_clusters)])

        return pred

    def KLD(self, p0, p1):
        P = torch.from_numpy(p0.astype(np.float32)).clone()
        Q = torch.from_numpy(p1.astype(np.float32)).clone()
        P = F.softmax(Variable(P), dim=1)
        Q = F.softmax(Variable(Q), dim=1)
        kld = ((P/(P+Q))*(P * (P / ((P/(P+Q))*P + (Q/(P+Q))*Q)).log())).sum() + ((Q/(P+Q))*(Q * (Q / ((P/(P+Q))*P + (Q/(P+Q))*Q)).log())).sum()
        return kld
    
    def Euclidean_distance(self, p0, p1):
        return np.sum((p0 - p1) ** 2)