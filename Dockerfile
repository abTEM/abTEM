FROM jupyter/scipy-notebook:latest

USER root

RUN conda install -c conda-forge cupy
RUN conda install -c conda-forge pyfftw gpaw
RUN /opt/conda/bin/pip install batchspawner abtem

USER ${NB_UID}
