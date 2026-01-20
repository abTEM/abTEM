# Use ROCm build container on Ubuntu 24.04 as base
FROM rocm/dev-ubuntu-24.04:6.4.4-complete

# Copy uv into image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get full-upgrade -y && apt-get -y install git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ubuntu
WORKDIR /home/ubuntu
ENV VIRTUAL_ENV="/home/ubuntu/venv"


RUN uv venv $VIRTUAL_ENV --python 3.12
RUN uv pip install --upgrade pip setuptools wheel

# Install CuPy
ENV ROCM_HOME="/opt/rocm/"
ENV CUPY_INSTALL_USE_HIP=1
ENV HCC_AMDGPU_TARGET=gfx1151
RUN uv pip install cupy

RUN uv cache clean
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# This is a supposed fix to help with gfx1151
# See: https://github.com/pytorch/pytorch/issues/164346
#ENV HSA_OVERRIDE_GFX_VERSION="11.0.0"
