# FedBench

FedBench is a Jupyter notebook-based tool, which supports performing easily experimental studies with various methods, experimental setups, and datasets.

## Resources
Paper: An Empirical Study of Personalized Federated Learning

## Setup
FedBench needs the following packages to be installed.

* PyTorch
* Torchvision
* Numpy
* Scikit-learn
* Pandas
* Matplotlib
* Jupyter notebook

Please install packages with `pip install -r requirements.txt`. 


## Data
FedBench can use five datasets: FEMNIST, Shakespeare, Sent140, MNIST, and CIFAR-10.

| Dataset     | Overview                                 | Task                      | 
| ----------- | ---------------------------------------- | ------------------------- | 
| FEMNIST     | Images dataset of handwritten character  |  Image Classification     | 
| Shakespeare | Text dataset of Shakespeare dialogues    | Next-Character Prediction | 
| Sent140     | Text dataset of tweets                   | Sentiment Analysis        | 
| MNIST       | Images dataset of handwritten characters |  Image Classification     | 
| CIFAR-10    | Image dataset of photo                   |  Image Classification     | 


These datasets should be prepared in advance by followings: 

  FEMNIST is downloaded from [Tensolflow-federated](https://github.com/tensorflow/federated), and Shakespeare is downloaded from [FedProx](https://github.com/litian96/FedProx).
  We have processed these datasets to be used as a Dataloader.
  Please download FEMNIST and Shakespeare dataset [here](https://drive.google.com/file/d/1NfmKUFeDogD6DlXkbyhbXI197F3ZfZ02/view?usp=sharing), unzip it and put the `federated_trainset_femnist.pickle`, `federated_testset_femnist.pickle`, `federated_trainset_shakespeare.pickle`, and `federated_testset_shakespeare.pickle` under `./data`.

  Sent140 are provided by the LEAF repository, which should be downloaded from [LEAF](https://github.com/TalwalkarLab/leaf/) by using the following commands:
  <br>
  `./preprocess.sh -s niid --sf 1.0 -k 50 -tf 0.8 -t sample`
  <br>
  After downloading the dataset, put folder leaf-master/data/sent140 into `./data` in this repository.

  If you use MNIST and CIFAR-10, you set any values for `alpha_label` and `alpha_size` in `class Argments()` and run the `get_dataset` function included in each code
  Then, you can download the dataset under `./data`.
  

## Model
For FEMNIST and MNIST you can use CNN, and for Shakespeare you can use LSTM.
For CIFAR-10, you can use VGG.
For Sent140, you can use a pre-trained 300-dimensional GloVe embedding and train RNN with an LSTM module.

In order to use a pre-trained 300-dimensional GloVe embedding, please download `glove.6B.300d.txt` from [here](https://nlp.stanford.edu/projects/glove/).
Next, from [the LEAF repository](https://github.com/TalwalkarLab/leaf/tree/master/models), conduct  `sent140/get_embs.py -f fp`, where fp is the file path to the `glove.6B.300d.txt`, to generate `embs.json`.
Then, put `embs.json` in `/models/` of this repository.



## Code
The jupyter notebook files for each method are available in `./code`.
We currently implemented the following methods:

* FedAvg (B. McMahan et al., Communication-efficient learning of deep networks from decentralized data, AISTATS 2017)
* FedProx (T. Li et al., Federated optimization in heterogeneous networks, MLSys 2020)
* HypCluster (Y. Mansour et al., Three approaches for personalization with applications to federated learning, arXiv 2020)
* FML (T. Shen et al., Federated mutual learning, arXiv 2020)
* FedMe (K. Matsuda et al., Fedme: Federated learning via model exchange, SDM 2022)
* LG-FedAvg (P. P. Liang et al., Think locally, act globally: Federated learning with local and global representations, arXiv 2020)
* FedPer (M. G. Arivazhagan et al., Federated learning with personalization layers, arXiv 2019)
* FedRep (L. Collins et al., Exploiting shared representations for personalized federated learning, ICML 2021)
* Ditto (T. Li et al., Ditto: Fair and robust federated learning through personalization, ICML 2021)
* pFedMe (C. T. Dinh et al., Personalized federated learning with moreau envelopes, NIPS 2020)


## Usage
You can conduct experiments by running the cells in order from the top. 
The experimental setups can be modified by changing the hyperparameters in the `class Argments()`.
Each variable is described following.

* `batch_size`: Batch size at training.   \[Default is `20`\]
* `test_batch`: Batch size at validation and testing.   \[Default is `1000`\]
* `global_epochs`: The number of global communication rounds.   \[Default is `300`\]
* `local_epochs`: The number of local training.   \[Default is `2`\]
* `lr`: Learning rate.  \[Default is `10**(-3)`\]
* `momentum`: Momentum.   \[Default is `0.9`\]
* `weight_deca`y: Weight decay.   \[Default is `10**-4.0`\]
* `clip`: Clipping gradients.   \[Default is `20.0`\]
* `partience`: The number of epochs from when the loss stops decreasing to when training stops.   \[Default is `300`\]
* `worker_num`: The number of clients.  \[Default is `20`\]
* `participation_rate`: The rate of clients who participate per global communication rounds.  \[Default is `1`\]
* `sample_num`: The number of clients who participate per global communication rounds. (automatically determined)
* `total_data_rate`: The rate of data samples to use. (only using for MNIST and CIFAR-10)   \[Default is `1`\]
* `unlabeleddata_size`: The number of unlabeled data samples.   \[Default is `1000`\]
* `device`: Machine informatio.   \[Default is `torch.device('cuda:0'if torch.cuda.is_available() else'cpu')`\]
* `criterion`: Loss function.   \[Default is `nn.CrossEntropyLoss()`\]
* `alpha_label`: The degree of label heterogeneity. (only using for MNIST and CIFAR-10)   \[Default is `0.5`\]
* `alpha_size`: The degree of data size heterogeneity. (only using for MNIST and CIFAR-10)  \[Default is `10`\]
* `dataset_name`: Name of the dataset to be used.   \[Default is `FEMNIST`\]


When the last cell is executed, the result of the experiment is stored in `./result/`.
