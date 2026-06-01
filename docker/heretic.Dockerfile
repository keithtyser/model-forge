ARG BASE_IMAGE=model-forge-posttrain-tf5:latest
FROM ${BASE_IMAGE}

RUN python3 -m pip install --no-cache-dir \
      "heretic-llm[research]" \
      "questionary>=2.0.0"
