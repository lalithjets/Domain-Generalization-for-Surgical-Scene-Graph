# Learning Domain Generalization with Graph Neural Network for Surgical Scene Understanding in Robotic Surgeries}

<!---------------------------------------------------------------------------------------------------------------->
- A domain generalized approach on surgical scene graphs to predict instrument-tissue interaction during robot-assisted surgery. We incorporate incremental learning to the feature extraction network and knowledge distillation-based student-teacher learning to the graph network, to accommodate new instruments and domain shifts in the new domain. 
- We design an enhanced curriculum by smoothing (E-CBS) based on Laplacian of Gaussian kernel and Gaussian kernel, and integrate with feature extraction network and visual-semantic graph attention network to improve the model performance. 
- Furthermore, we normalize the feature extraction and graph network’s logits by T-Norm and study its effect in calibrating the model. 
- The proposed SSU is trained on nephrectomy procedures video frames and then domain generalized to transoral robotics surgery video frames.

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

- `Feature_extractor/`: Used to extract features from dataset images to train the graph network.
- `datasets/`: Contains the dataset needed to train the network.
- `model/`: Contains network models.
- `utils/`: Contains utility tools used for training and evaluation.
- `checkpoints/`: Conatins trained weights

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


### Dataset
#### Download feature extracted data for training and evalutation
1. gdrive_link for features [To be added]()
2. Download the pretrain word2vec model on [GoogleNews](https://code.google.com/archive/p/word2vec/) and put it into `datasets/word2vec` 

### Training
- model_train.py
- Checkpoints will be saved in `checkpoints/` folder.

### Testing
- model_evaluation.py


### Acknowledgement
Code adopted and modified from :
1. Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion
    - Paper [Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion](https://arxiv.org/abs/2001.02302).
    - Official Pytorch implementation [code](https://github.com/birlrobotics/vs-gats).
2. End-to-End Incremental Learning
    - Paper [End-to-End Incremental Learning](ttps://arxiv.org/pdf/1807.09536.pdf).
    - Pytorch implementation [code](https://github.com/fmcp/EndToEndIncrementalLearning).
3. Curriculum by smoothing
    - Paper [Curriculum by smoothing](https://arxiv.org/pdf/2003.01367.pdf).
    - Pytorch implementation [code](https://github.com/pairlab/CBS).