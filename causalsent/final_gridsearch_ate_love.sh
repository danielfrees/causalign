#!/bin/bash

PRETRAINED_MODELS=("sentence-transformers/msmarco-distilbert-base-v4")  # "meta-llama/Llama-3.1-8B"
UNFREEZE_BACKBONES=("top0")
RIESZ_HEAD_TYPES=("linear")
LAMBDAS_BCE=(1.0)
LAMBDAS_REG=(0 0.01 0.1)
LAMBDAS_RIESZ=(1.0)
SENTIMENT_HEAD_TYPES=("linear")

# Loop over all parameter combinations
for MODEL in "${PRETRAINED_MODELS[@]}"; do
    for BACKBONE in "${UNFREEZE_BACKBONES[@]}"; do
        for RIESZ_HEAD in "${RIESZ_HEAD_TYPES[@]}"; do
            for SENTIMENT_HEAD in "${SENTIMENT_HEAD_TYPES[@]}"; do
                for LAMBDA_BCE in "${LAMBDAS_BCE[@]}"; do
                    for LAMBDA_REG in "${LAMBDAS_REG[@]}"; do
                        for LAMBDA_RIESZ in "${LAMBDAS_RIESZ[@]}"; do
                            PROJECT_NAME="ate_inter_real_love"

                            echo "Running with Model: $MODEL, Unfreeze Backbone: $BACKBONE, Riesz Head: $RIESZ_HEAD, Sentiment Head: $SENTIMENT_HEAD, Lambda BCE: $LAMBDA_BCE, Lambda REG: $LAMBDA_REG, Lambda RIESZ: $LAMBDA_RIESZ, Project: $PROJECT_NAME"

                            # Execute the training script with the current combination
                            python train_causal_sent.py \
                                --project_name "$PROJECT_NAME" \
                                --pretrained_model_name "$MODEL" \
                                --unfreeze_backbone "$BACKBONE" \
                                --riesz_head_type "$RIESZ_HEAD" \
                                --sentiment_head_type "$SENTIMENT_HEAD" \
                                --epochs 30 \
                                --limit_data 0 \
                                --max_seq_length 400 \
                                --lr 5e-5 \
                                --treatment_phrase love \
                                --lambda_bce "$LAMBDA_BCE" \
                                --lambda_reg "$LAMBDA_REG" \
                                --lambda_riesz "$LAMBDA_RIESZ" \
                                --dataset imdb \
                                --log_every 50 \
                                --doubly_robust \
                                --interleave_training \
                                --running_ate
                        done
                    done
                done
            done
        done
    done
done