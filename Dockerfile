# Use the specified CUDA base image with Rocky Linux 9
FROM nvidia/cuda:12.8.1-cudnn-devel-rockylinux9

# Install dependencies
WORKDIR /app
RUN dnf update --assumeyes && \
    dnf install --assumeyes git python3.12-devel libcusparselt-devel && \
    dnf clean all && \
    python3.12 -m ensurepip && \
    pip3.12 install --upgrade pip

# Install the package
COPY . /app
RUN pip3.12 install .
ENV XDG_DATA_HOME=/tmp/data XDG_CONFIG_HOME=/tmp/config XDG_CACHE_HOME=/tmp/cache XDG_STATE_HOME=/tmp/state

# Set the entrypoint
ENTRYPOINT ["qmb"]
CMD []
