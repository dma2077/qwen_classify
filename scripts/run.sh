#!/bin/bash

# 精简训练启动脚本
deepspeed --num_gpus 8 --master_port 29500 \
    training/train.py \
    --config configs/config_3e_6_ls.yaml \
    --deepspeed_config configs/ds_s2.json 