FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# KFP executor uses python3 — ensure it exists
RUN ln -sf $(which python) /usr/local/bin/python3

RUN pip install --no-cache-dir \
    google-cloud-storage \
    kfp==2.12.1

WORKDIR /app
