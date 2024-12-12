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

#--treated_only  # to compute and use ATT #! dont use if running synthetic validations with  --adjust_ate

python train_causal_sent.py \
    --project_name debug_doubleback_causal_sentiment_synthetic_artichoke_pos050 \
    --pretrained_model_name sentence-transformers/msmarco-distilbert-base-v4 \
    --unfreeze_backbone_riesz top0 \
    --unfreeze_backbone_sentiment top0 \
    --riesz_head_type linear \
    --sentiment_head_type linear \
    --epochs 50 \
    --limit_data 0 \
    --max_seq_length 50 \
    --lr 1e-4 \
    --treatment_phrase artichoke \
    --lambda_bce 0 \
    --lambda_reg 0 \
    --lambda_riesz 0.1 \
    --lambda_l1 1e-5 \
    --dataset imdb \
    --log_every 50 \
    --running_ate \
    --adjust_ate \
    --synthetic_ate 0.50 \
    --synthetic_ate_treat_fraction 0.5 \
    --doubly_robust 