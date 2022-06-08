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

  MNIST and CIFAR-10 do not require preparation. 
  If you use the CIFAR-10 dataset, you run the `get_dataset` function included in each code and can download the dataset under `./data`.

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

* batch_size: Batch size at training.
* test_batch: Batch size at validation and testing.
* global_epochs: The number of global communication rounds.
* local_epochs: The number of local training.
* lr: Learning rate.
* momentum: Momentum.
* weight_decay: Weight decay.
* clip: Clipping gradients.
* partience: The number of times to stop training if the loss does not decrease.
* worker_num: The number of clients.
* participation_rate: The rate of clients who participate per global communication rounds.
* sample_num: The number of clients who participate per global communication rounds. (automatically determined)
* total_data_rate: The rate of data samples to use. (only using for MNIST and CIFAR-10)
* device: Machine information 
* criterion: Loss function
* alpha_label: The degree of label heterogeneity (only using for MNIST and CIFAR-10)
* alpha_size: The degree of data size heterogeneity (only using for MNIST and CIFAR-10)

The list `dataset_names` contains the names of the available datasets.
Each dataset can be used by setting `dataset_name = dataset_names[i]` where `i` is the following number.

* 0: FEMNIST
* 1: Shakespeare
* 2: Sent140
* 3: MNIST
* 4: CIFAR-10


When the last cell is executed, the result of the experiment is stored in `./result/`.
