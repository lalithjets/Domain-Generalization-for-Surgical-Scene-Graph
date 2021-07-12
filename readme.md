# Learning Domain Generaliazation with Graph Neural Network for Surgical Scene Understanding.

<!---------------------------------------------------------------------------------------------------------------->
Code adopted and modified from :
a) Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion
- Official Pytorch implementation for [Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion](https://arxiv.org/abs/2001.02302).
b) End-to-End Incremental Learning
- Official Pytorch implementation for :

<!---------------------------------------------------------------------------------------------------------------->
## VS-GATs
To be added
<!---------------------------------------------------------------------------------------------------------------->
## Graph
### Preliminary
To be added

<!---------------------------------------------------------------------------------------------------------------->
## Code Overview
In this project, we implement our method using the Pytorch and DGL library and there are three main folders: 

- `datasets/`: 
- `model/`: 
- `result/`: 

In the following, we briefly introduce some main scripts.

#### datasets/
To be added

#### model/
- `config.py`: To be added
- `model.py`: To be added
- `grnn.py`: To be added
- `utils.py`: To be added

#### result/
- To be added
<!---------------------------------------------------------------------------------------------------------------->
## Library Prerequisities.

### DGL
<a href='https://docs.dgl.ai/en/latest/install/index.html'>DGL</a> is a Python package dedicated to deep learning on graphs, built atop existing tensor DL frameworks (e.g. Pytorch, MXNet) and simplifying the implementation of graph-based neural networks.

### Prerequisites
- Python 3.6
- Pytorch 1.1.0
- DGL 0.3
- CUDA 10.0
- Ubuntu 16.04

### Installation
1. Clone this repository.   

    ```
    git clone https://github.com/BIGJUN777/VS-GATs.git
    ```
  
2. Install Python dependencies:   

    ```
    pip install -r requirements.txt
    ```

### Prepare Data
#### Download original data
1. gdrive_link for features: 
3. Download the pretrain word2vec model on [GoogleNews](https://code.google.com/archive/p/word2vec/) and put it into `datasets/word2vec` 

#### Process Data
- feature_extractor/feature_extraction.ipynb.

### Training
- train.py

- Checkpoints will be saved in `checkpoints/` folder.
### Testing
- Validation.py

### Results
- Please check the paper for the quantitative results and several qualitative detection results are as follow:
    ![detection_results](./assets/detection.png)

### Acknowledgement
In this project, some codes which process the data and eval the model are built upon [ECCV2018-Learning Human-Object Interactions by Graph Parsing Neural Networks](https://github.com/SiyuanQi/gpnn) and [ICCV2019-No-Frills Human-Object Interaction Detection: Factorization, Layout Encodings, and Training Techniques](https://github.com/BigRedT/no_frills_hoi_det). Thanks them for their great works.
