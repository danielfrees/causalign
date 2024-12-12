#!/bin/bash

LEARNING_RATES=(5e-5 1e-5 5e-6 1e-6 5e-4 1e-4)
LAMBDAS_BCE=(  1   1   1 0.3 0.3 0.3 1 )
LAMBDAS_REG=("0.1" "1"   "0.1" "0.1" "0.3" "0.1" "1")
LAMBDAS_RIESZ=("0.1" "0.1" "1"   "0.1" "0.1" "0.3" "1")

# Loop over all parameter combinations
for LR in "${LEARNING_RATES[@]}"; do
    for i in "${!LAMBDAS_BCE[@]}"; do
        echo "Running with Learning Rate: $LR, Lambda: ${LAMBDAS_BCE[i]}, Lambda: ${LAMBDAS_REG[i]}, Lambda: ${LAMBDAS_RIESZ[i]}"
        
        # Execute the training script with the current combination
        python train_causal_sent.py \
            --pretrained_model_name "sentence-transformers/msmarco-distilbert-base-v4" \
            --unfreeze_backbone "iterative" \
            --riesz_head_type "conv" \
            --sentiment_head_type "conv" \
            --epochs 30 \
            --limit_data 500 \
            --max_seq_length 150 \
            --lr $LR \
            --treatment_phrase love \
            --lambda_bce ${LAMBDAS_BCE[i]} \
            --lambda_reg ${LAMBDAS_REG[i]} \
            --lambda_riesz ${LAMBDAS_RIESZ[i]} \
            --dataset imdb \
            --log_every 5 \
            --doubly_robust\
            --running_ate
    done
done