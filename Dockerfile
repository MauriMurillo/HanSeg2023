# Base image to build
# FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.12-py3
# FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive

#create user and add to group
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

#create app input and output directory
RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

#create user
USER user

#change working directory
WORKDIR /opt/app

#Set needed environment variable
ENV PATH="/home/user/.local/bin:${PATH}"
ENV nnUNet_raw="/opt/app/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/app/nnUNet_preprocessed"
ENV nnUNet_results="/opt/app/nnUNet_results"
#upgrade pip
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools && python -m pip install --upgrade pip

COPY --chown=user:user nnUNet /opt/app/nnUNet
COPY --chown=user:user nnUNet_raw /opt/app/nnUNet_raw
COPY --chown=user:user nnUNet_preprocessed /opt/app/nnUNet_preprocessed
COPY --chown=user:user nnUNet_results /opt/app/nnUNet_results
COPY --chown=user:user requirements.txt /opt/app/
#install all needed packages
RUN  pip install -r requirements.txt
RUN cd nnUNet && pip install -e .

COPY --chown=user:user custom_algorithm.py /opt/app/
COPY --chown=user:user testing.py /opt/app/
COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user cuda.py /opt/app/


#run process
ENTRYPOINT [ "python", "-m", "process" ]
# ENTRYPOINT [ "python", "cuda.py" ]
