#!/bin/bash

# test script mostly
# example usage: limited training data for speed here.

python train_causal_sent.py \
    --pretrained_model_name sentence-transformers/msmarco-distilbert-base-v4 \
    --unfreeze_backbone iterative \
    --riesz_head_type fcn \
    --sentiment_head_type fcn \
    --epochs 10 \
    --limit_data 500 \
    --max_seq_length 150 \
    --lr 5e-5 \
    --treatment_phrase love \
    --lambda_bce 1.0 \
    --lambda_reg 0.01 \
    --lambda_riesz 0.01 \
    --dataset imdb \
    --log_every 5 \
    --running_ate
