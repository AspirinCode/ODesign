FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set default RUN shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


# Install basic packages for compiling and building
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-12 \
    git \
    curl \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda & use Python 3.10
ARG python=3.10
ENV PYTHON_VERSION=${python}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda install -y python=${PYTHON_VERSION}


# # Set up TUNA mirror (optional)
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


# Install packages from `requirements.txt` with pip
COPY . /app/ODesign
WORKDIR /app/ODesign
RUN pip install --no-cache-dir -r requirements.txt -f https://data.pyg.org/whl/torch-2.3.1+cu121.html


# Set up kernels
# DS4Sci_EvoformerAttention
RUN git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git /kernels/cutlass
ENV CUTLASS_PATH=/kernels/cutlass

# fast_layernorm
ENV LAYERNORM_TYPE=fast_layernorm