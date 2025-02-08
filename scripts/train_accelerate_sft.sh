#!/bin/bash

LAUNCHER=\
"accelerate launch \
--config_file configs/accelerate_deepspeed_config.yaml" 

CMD="sft_accelerator.py \
--from_scratch True \
--tokenizer_name_or_path "lm_models/Qwen2.5-0.5B-Instruct" \
--dataset_dir "dataset/sharegpt_gpt4" \
--dataset_name "sharegpt_gpt4" \
--batch_size 6 \
--gradient_accumulation_steps 8 \
--learning_rate 1e-6 \
--num_epochs 1 \
--max_grad_norm 1.0 \
--use_fp8 False \
--device "cuda" \
--seed 1024"

$LAUNCHER $CMD

# accelerate launch --config_file configs/accelerate_deepspeed_config.yaml sft_accelerator.py
