ARG BASE_IMAGE=vllm-node-tf5:latest
FROM ${BASE_IMAGE}

ARG MODELOPT_VERSION=0.43.0

RUN python3 -m pip install --no-cache-dir \
      "nvidia-modelopt==${MODELOPT_VERSION}" \
      "accelerate>=1.0.0" \
      "datasets>=3.0.0" \
      "deepspeed>=0.9.6" \
      "diffusers>=0.32.2" \
      "nltk" \
      "omegaconf>=2.3.0" \
      "peft>=0.17.0" \
      "pulp" \
      "scipy" \
      "wonderwords" && \
    git clone -b "${MODELOPT_VERSION}" --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /opt/TensorRT-Model-Optimizer && \
    python3 -m pip install --no-cache-dir -e /opt/TensorRT-Model-Optimizer
