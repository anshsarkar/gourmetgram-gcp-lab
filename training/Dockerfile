FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# KFP executor uses python3 — the conda image only has 'python'
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python3 && \
    python3 --version

RUN pip install --no-cache-dir \
    google-cloud-storage \
    kfp==2.12.1

WORKDIR /app
