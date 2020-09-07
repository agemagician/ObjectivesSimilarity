FROM continuumio/miniconda3

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml


WORKDIR /app

COPY . .

EXPOSE 5000

ENV BASH_ENV ~/.bashrc
SHELL ["/bin/bash", "-c"]

RUN conda activate semserver && conda info --envs
RUN echo "conda activate semserver" >> ~/.bashrc

CMD ["conda", "run", "-n", "semserver", "python", "server.py"]