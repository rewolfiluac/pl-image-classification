FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ARG UID
ARG GID

ENV DEBIAN_FRONTEND=noninteractive
ENV VERSION 7.2.1-1+cuda11.1
ENV TRT_VERSION 7.2.1.6

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    sudo tzdata wget curl git vim \
    # Build Tool
    cmake extra-cmake-modules build-essential libssl-dev libffi-dev pkg-config \
    ccache ecm mesa-utils \
    # Python3.7用
    python3.7 python3.7-dev python3.7-tk python3.7-distutils \
    # OpenCV-Python 用
    libopencv-dev libgl1-mesa-dev \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | sudo python3.7 && \
    python3.7 -m pip install numpy \
    opencv_python opencv_python_headless \
    torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html \
    pytorch-lightning mlflow gorilla \
    scikit-image==0.17.2 scikit-learn==0.23.2 \
    onnx==1.8.0 \
    timm==0.3.2 torch_optimizer==0.0.1a17 addict==2.4.0 \
    flake8==3.8.4 autopep8==1.5.4 \
    omegaconf \
    -U git+https://github.com/albumentations-team/albumentations

# sudo権限を持つ一般ユーザーを作成
ENV USER dev
ENV GROUP dev
ENV HOME /home/${USER}
ENV SHELL /bin/bash

RUN groupadd -g ${GID} ${GROUP}
RUN useradd -u ${UID} -g ${GROUP} -m ${USER}
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:dev" | chpasswd
RUN sed -i.bak "s#${HOME}:#${HOME}:${SHELL}#" /etc/passwd

USER ${USER}
WORKDIR ${HOME}
