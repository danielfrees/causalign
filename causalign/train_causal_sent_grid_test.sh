#!/bin/bash

PRETRAINED_MODELS=("sentence-transformers/msmarco-distilbert-base-v4") #"meta-llama/Llama-3.1-8B" 
UNFREEZE_BACKBONES=("top0" "top3" "all" "iterative")
RIESZ_HEAD_TYPES=("fcn" "linear" "conv")
SENTIMENT_HEAD_TYPES=("fcn" "linear" "conv")
RUNNING_ATE_OPTIONS=("" "--running_ate")  # Empty string for no argument

# Loop over all parameter combinations
for MODEL in "${PRETRAINED_MODELS[@]}"; do
    for BACKBONE in "${UNFREEZE_BACKBONES[@]}"; do
        for RIESZ_HEAD in "${RIESZ_HEAD_TYPES[@]}"; do
            for SENTIMENT_HEAD in "${SENTIMENT_HEAD_TYPES[@]}"; do
                for RUNNING_ATE in "${RUNNING_ATE_OPTIONS[@]}"; do
                    echo "Running with Model: $MODEL, Unfreeze Backbone: $BACKBONE, Riesz Head: $RIESZ_HEAD, Sentiment Head: $SENTIMENT_HEAD, Running ATE: $RUNNING_ATE"
                    
                    # Execute the training script with the current combination
                    python train_causal_sent.py \
                        --pretrained_model_name "$MODEL" \
                        --unfreeze_backbone "$BACKBONE" \
                        --riesz_head_type "$RIESZ_HEAD" \
                        --sentiment_head_type "$SENTIMENT_HEAD" \
                        --epochs 1 \
                        --limit_data 500 \
                        --max_seq_length 150 \
                        --lr 5e-5 \
                        --treatment_phrase love \
                        --lambda_bce 1.0 \
                        --lambda_reg 0.01 \
                        --lambda_riesz 0.01 \
                        --dataset imdb \
                        --log_every 5 \
                        $RUNNING_ATE
                done
            done
        done
    done
done