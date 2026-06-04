ARG BASE_IMAGE=model-forge-posttrain-tf5:latest
FROM ${BASE_IMAGE}

RUN python3 -m pip install --no-cache-dir \
      "abliterix" \
    && (python3 -m pip uninstall -y kernels kernels-data || true)

ENV USE_HUB_KERNELS=NO
