#!/bin/bash

set -a  # Enable auto-export for all variables
source .env
set +a  # Disable auto-export to avoid unexpected behavior later
python hifi_gan_bwe/scripts/train.py \
    bwe-on-adbenh-less-noise \
    --train_dataset_path /data/projects/audio-enhancement/datasets/adbenh/ \
    --log_path /data/projects/audio-enhancement/hifi-gan-bwe/checkpoints/ \
    --db_name audio \
    --collection_name adbenh \
    --silence_prob 0.1 \
    --use_vad_intervals