# FROM python:3.7.7
# WORKDIR /Server
# ADD . /Server
# RUN pip install -r requirements.txt
# CMD ["python","server.py"]


# FROM continuumio/miniconda3
# RUN conda create -n env python=3.6
# RUN echo "source activate env" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH

FROM continuumio/miniconda3

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

RUN echo "conda activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" >> ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

ENV CONDA_DEFAULT_ENV $(head -1 /tmp/environment.yml | cut -d' ' -f2)