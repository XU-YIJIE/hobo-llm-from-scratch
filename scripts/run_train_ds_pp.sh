#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 根据实际GPU数量修改

# 训练参数
MODEL_PATH="lm_models/Qwen2.5-0.5B-Instruct"  # Qwen2模型路径
OUTPUT_DIR="outputs/qwen_sft"           # 输出目录
NUM_GPUS=2                   # 总GPU数量
PP_SIZE=2                    # Pipeline并行度
DP_SIZE=$((NUM_GPUS/PP_SIZE))  # 数据并行度

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 启动分布式训练
deepspeed \
    --num_gpus=$NUM_GPUS \
    train_ds_pipe.py \
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