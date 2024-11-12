# Use the specified CUDA base image with Rocky Linux 9
FROM nvidia/cuda:12.6.2-devel-rockylinux9

# Install dependencies
WORKDIR /app
RUN dnf update --assumeyes && \
    dnf install --assumeyes git wget openssl-devel bzip2-devel libffi-devel zlib-devel && \
    dnf clean all

# Install Python 3.12.7
ENV PYTHON_VERSION=3.12.7
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xvf Python-${PYTHON_VERSION}.tgz && \
    rm Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION}*

# Install the package
COPY . /app
RUN pip3.12 install .
ENV XDG_DATA_HOME=/tmp/data XDG_CONFIG_HOME=/tmp/config XDG_CACHE_HOME=/tmp/cache XDG_STATE_HOME=/tmp/state

# Set the entrypoint
ENTRYPOINT ["qmb"]
CMD []
