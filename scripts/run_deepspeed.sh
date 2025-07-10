#!/bin/bash

# Simple DeepSpeed training script

CONFIG_FILE="configs/config.yaml"
OUTPUT_DIR="./outputs"
NUM_GPUS=8

if [[ $NUM_GPUS -eq 1 ]]; then
    python training/deepspeed_trainer.py \
        --config_file $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --deepspeed_config configs/deepspeed_config.json \
        --local_rank 0
else
    deepspeed --num_gpus=$NUM_GPUS \
        training/deepspeed_trainer.py \
        --config_file $CONFIG_FILE \
        --output_dir $OUTPUT_DIR \
        --deepspeed_config configs/deepspeed_config.json
fi 