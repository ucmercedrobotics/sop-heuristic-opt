# sop-heuristic-opt

Stochastic orienteering problem self-play pipeline leveraging GNNs with MCTS

TODO List:

- [ ] MCTS Implementation
- [ ] GNN Implementation in PyTorch
- [ ] Dockerfile supporting CUDA, CPU, and MPS, also install PyG and Pytorch

## Installation

1. Create a new environment (ex. w/ conda)

```sh
conda create -n <ENV_NAME> python=3.10
conda activate <ENV_NAME>
```

2. Install packages in requirements.txt

```sh
pip install -r requirements.txt
```

3. Install [Pytorch](https://pytorch.org/get-started/locally/) and [PyG w/ torch_scatter](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) from links
