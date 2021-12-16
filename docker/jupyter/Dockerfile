FROM jupyter/scipy-notebook:latest
USER root

RUN conda install -c intel mkl_fft
RUN conda install -c conda-forge gpaw
RUN /opt/conda/bin/pip install batchspawner dask-labextension

RUN mkdir /src
WORKDIR /src
COPY ../.. /src

RUN /opt/conda/bin/pip install --upgrade setuptools
RUN /opt/conda/bin/pip install -r requirements.txt
RUN /opt/conda/bin/pip install .

RUN jupyter labextension install dask-labextension

USER ${NB_UID}

WORKDIR "${HOME}"