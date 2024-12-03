#!/bin/bash

# test script mostly
# example usage: limited training data for speed here.

# currently training riesz only 

# best lr 1e-4
# best lambdas:
# bce: 0.3
# reg: 0.1
# riez: 0.1
# or 
# 1.0, 0.1, 0.1

python train_causal_sent.py \
    --project_name causal_sentiment_test_best \
    --pretrained_model_name sentence-transformers/msmarco-distilbert-base-v4 \
    --unfreeze_backbone top3 \
    --riesz_head_type fcn \
    --sentiment_head_type fcn \
    --epochs 10 \
    --limit_data 4000 \
    --max_seq_length 150 \
    --lr 1e-5 \
    --treatment_phrase best \
    --lambda_bce 0 \
    --lambda_reg 0 \
    --lambda_riesz 1.0\
    --dataset imdb \
    --log_every 5 \
    --running_ate \
    --doubly_robust \
    --adjust_ate \
    --adjust_change 0.5 \
    --adjust_treat_pop 0.5 \
    --treated_only
