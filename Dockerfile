FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

RUN apt-get update
RUN apt-get install --yes git python3-dev python3-pip ninja-build

WORKDIR /app
COPY . /app
RUN pip install --break-system-packages --user .

ENTRYPOINT ["/root/.local/bin/qmb"]
CMD []
