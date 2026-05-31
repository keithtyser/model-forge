ARG BASE_IMAGE=vllm-node-tf5:latest
FROM ${BASE_IMAGE}

ARG MODELOPT_VERSION=0.43.0

RUN python3 -m pip install --no-cache-dir \
      "accelerate>=1.10.0" \
      "bitsandbytes>=0.46.0" \
      "datasets>=3.0.0" \
      "deepspeed>=0.17.0" \
      "nvidia-modelopt==${MODELOPT_VERSION}" \
      "peft>=0.17.0" \
      "trl>=0.24.0" \
      "scipy" \
      "sentencepiece" && \
    git clone -b "${MODELOPT_VERSION}" --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /opt/TensorRT-Model-Optimizer && \
    python3 -m pip install --no-cache-dir -e /opt/TensorRT-Model-Optimizer
