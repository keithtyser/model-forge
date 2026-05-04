#!/usr/bin/env bash
set -euo pipefail
cd /home/ktyser/projects/model-forge
MODEL_FORGE_TRIALS=3 ./forge eval gemma4_26b_a4b local_ft --internal
./forge eval gemma4_26b_a4b local_ft --artifact
./forge eval gemma4_26b_a4b local_ft --external
./forge compare gemma4_26b_a4b
