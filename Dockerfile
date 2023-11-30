FROM registry.iaticetc.cn:5000/lbh_nerf:latest

RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list; \
    sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && apt update && \
    apt -y install build-essential gcc g++ gdb binutils pciutils net-tools iputils-ping iproute2 git vim wget libgl1 curl make openssh-server openssh-client tmux tree man unzip unrar && \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple jupyterlab

RUN configargparse plyfile joblib