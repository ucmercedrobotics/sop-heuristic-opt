FROM python:3.11 AS builder

# install requirements through pip
COPY requirements.txt /requirements.txt
COPY requirements_torch_dep.txt /requirements_torch_dep.txt
RUN python -m pip install -r /requirements.txt
RUN python -m pip install -r /requirements_torch_dep.txt

FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base

RUN apt-get -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common build-essential wget

# PYTHON 3.11
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet \
    python3.11 \
    python3.11-dev \
    pip

RUN apt install -y software-properties-common
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
apt-get update && \
apt-get install -y libcudnn8=8.9.6.50-1+cuda12.2 && \
apt-get install -y libcudnn8-dev=8.9.6.50-1+cuda12.2 && \
apt-get install -y kmod && \
apt-get install -y nvidia-cuda-toolkit

# TODO: keep this just in case it's required. It doesn't seem that it is since apt-get works. 
# check list of availavle cudnn packages
# apt list -a libcudnn8-dev
# download cudnn for Linux from official Nvidia site
# refer https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html?ncid=em-prod-337416
# COPY cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb /var
# # install cudnn
# RUN dpkg -i /var/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb && \
# cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/*.gpg /usr/share/keyrings/ && \
# apt-get update && \
# apt-get install -y libcudnn8=8.9.7.29-1+cuda11.8 --allow-downgrades && \
# apt-get install -y libcudnn8-dev=8.9.7.29-1+cuda11.8 --allow-downgrades && \
# apt-get install -y libcudnn8-samples=8.9.7.29-1+cuda11.8 --allow-downgrades && \

RUN export LIBRARY_PATH=/usr/lib/cuda/nvvm/libdevice:$LIBRARY_PATH && \
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda && \
cd /root

# Setting LIBRARY_PATH to include the path to the libdevice library is necessary for compiling CUDA code that uses the NVVM compiler
ENV LIBRARY_PATH="/usr/lib/cuda/nvvm/libdevice:$LIBRARY_PATH"
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

# image for running with a GPU: LINUX ONLY
FROM base AS local

# copy over all python files from builder stage and add location to path
COPY --from=builder /usr/local /usr/local

WORKDIR /sop-opt