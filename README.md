# sop-heuristic-opt

Stochastic orienteering problem self-play pipeline leveraging GNNs with MCTS

TODO List:

- [ ] Add support for torch dataloader
- [ ] MILP script, maybe add multithreading
- [ ] Proper experiment pipeline
- [ ] MCTS ACO Implementation

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

3. Install [Pytorch](https://pytorch.org/get-started/locally/) and [PyG w/ torch_scatter & torch_sparse](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) from links

## Using Docker
### Install
Using Linux (Ubuntu) install `nvidia-toolkit`
```bash
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-XXX
sudo reboot now
```
Where `XXX` is based on recommended version listed in second command.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit nvidia-container-toolkit
sudo reboot now
```

Add the following to your rc file of your shell of choice:
```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.<version>/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Where `<version>` is the version you find in `/usr/local/`.

After you've added this, make sure to `source` your rc file.

### Running Container
```bash
make build
make bash
```

## Execution
This will run `improve_heuristic.py`.
```bash
make run
```

To generate a dataset
```bash
make dataset
```
You can find the outputs in `data/`.

## Source Control
To use precommit checks, make sure that you install `pre-commit` and initialize your repo.
If you're using Linux you can `make repo-init` and it will install all hooks into your local `.git` folder.
If not, install `pre-commit` and then `pre-commit install` to initialize the repo.
For example on Mac, `brew install pre-commit` will do the trick.
