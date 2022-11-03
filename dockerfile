# syntax=docker/dockerfile:1

FROM mrtrix3/mrtrix3:latest
SHELL ["/bin/bash", "-c"]
WORKDIR ~/
RUN apt-get update && apt-get install git git-lfs -y && git lfs install && pip install numpy && cd ~/ && git clone --recursive https://github.com/LeeReid1/tractography-reliability.git
