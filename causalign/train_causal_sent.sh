#!/bin/bash

python train_causal_sent.py \
    --limit_data 500 \
    --max_seq_length 150 \
    --lr 5e-5 \
    --treatment_phrase love \
    --lambda_bce 1.0 \
    --lambda_reg 0.001 \
    --lambda_riesz 0.01 \
    --dataset imdb \
    --log_every 5
    # Uncomment below for additional args
    # --running_ate \
    # --estimate_targets_for_ate \