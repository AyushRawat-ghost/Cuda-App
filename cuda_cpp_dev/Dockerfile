# Use the NVIDIA CUDA DEVEL image for C++ compilation (includes nvcc and GCC)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set non-root user (good practice)
RUN useradd -m learner
USER learner
WORKDIR /home/learner

# Install necessary development tools
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        gdb \
        cmake \
        git \
        vim \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]# Use the NVIDIA CUDA DEVEL image for C++ compilation (includes nvcc and GCC)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04


USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        gdb \
        cmake \
        git \
        vim \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and ensure it uses the correct CUDA version
# Note: Use python3 -m pip to ensure the correct pip is used
RUN python3 -m pip install --no-cache-dir \
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Set non-root user (good practice for security)
RUN useradd -m learner
USER learner
WORKDIR /home/learner

# Set the entry point to a simple Bash shell for interactive learning and compilation
CMD ["/bin/bash"]