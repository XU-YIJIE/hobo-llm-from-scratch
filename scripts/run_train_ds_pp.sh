#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

MODEL_PATH="lm_models/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR="outputs/qwen_sft"
PP_SIZE=2
DP_SIZE=$((NUM_GPUS/PP_SIZE))

mkdir -p $OUTPUT_DIR

deepspeed \
    --num_gpus=$NUM_GPUS \
    sft_ds_pipe.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --pp_size $PP_SIZE \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --max_seq_len 1024 \
    --fp16 \
    --checkpoint_activations \
    --checkpoint_num_layers 1 \
    --deepspeed \
    --deepspeed_config ds_config.json 