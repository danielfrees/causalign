#!/bin/bash

PRETRAINED_MODELS=("sentence-transformers/msmarco-distilbert-base-v4")  # "meta-llama/Llama-3.1-8B"
UNFREEZE_BACKBONES=("top1")
RIESZ_HEAD_TYPES=("linear")
LAMBDAS_BCE=(1.0)
LAMBDAS_REG=(0 0.01)
LAMBDAS_RIESZ=(1.0)
SENTIMENT_HEAD_TYPES=("linear")
SYNTHETIC_ATES=(-0.50 -0.25 0.25 0.50)

# Loop over all parameter combinations
for MODEL in "${PRETRAINED_MODELS[@]}"; do
    for BACKBONE in "${UNFREEZE_BACKBONES[@]}"; do
        for RIESZ_HEAD in "${RIESZ_HEAD_TYPES[@]}"; do
            for SENTIMENT_HEAD in "${SENTIMENT_HEAD_TYPES[@]}"; do
                for LAMBDA_BCE in "${LAMBDAS_BCE[@]}"; do
                    for LAMBDA_REG in "${LAMBDAS_REG[@]}"; do
                        for LAMBDA_RIESZ in "${LAMBDAS_RIESZ[@]}"; do
                            for SYNTHETIC_ATE in "${SYNTHETIC_ATES[@]}"; do
                                # Generate project name based on synthetic_ate
                                if (( $(echo "$SYNTHETIC_ATE < 0" | bc -l) )); then
                                    ATE_SIGN="neg"
                                else
                                    ATE_SIGN="pos"
                                fi
                                ATE_VALUE=$(echo "$SYNTHETIC_ATE" | awk '{printf "%.2f", $1}' | sed 's/-//;s/\.//')
                                PROJECT_NAME="causal_sentiment_synthetic_iceberg_${ATE_SIGN}${ATE_VALUE}"

                                echo "Running with Model: $MODEL, Unfreeze Backbone: $BACKBONE, Riesz Head: $RIESZ_HEAD, Sentiment Head: $SENTIMENT_HEAD, Lambda BCE: $LAMBDA_BCE, Lambda REG: $LAMBDA_REG, Lambda RIESZ: $LAMBDA_RIESZ, Synthetic ATE: $SYNTHETIC_ATE, Project: $PROJECT_NAME"

                                # Execute the training script with the current combination
                                python train_causal_sent.py \
                                    --project_name "$PROJECT_NAME" \
                                    --pretrained_model_name "$MODEL" \
                                    --unfreeze_backbone "$BACKBONE" \
                                    --riesz_head_type "$RIESZ_HEAD" \
                                    --sentiment_head_type "$SENTIMENT_HEAD" \
                                    --epochs 10 \
                                    --limit_data 2000 \
                                    --max_seq_length 150 \
                                    --lr 5e-5 \
                                    --treatment_phrase iceberg \
                                    --lambda_bce "$LAMBDA_BCE" \
                                    --lambda_reg "$LAMBDA_REG" \
                                    --lambda_riesz "$LAMBDA_RIESZ" \
                                    --dataset imdb \
                                    --log_every 10 \
                                    --adjust_ate \
                                    --synthetic_ate "$SYNTHETIC_ATE" \
                                    --synthetic_ate_treat_fraction 0.5 \
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
    done
done