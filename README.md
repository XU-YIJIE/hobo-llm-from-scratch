# Hobo-LLM: From sft to grpo

![License](https://img.shields.io/badge/License-Apache%202.0-green)

从0到1实现LLM训练代码（pt/sft/grpo）。致力于兼顾教学性和实用性，适配中小型训练场景，支持多机多卡训练。尽量少地依赖第三方库，提升可读性的同时确保通用性

## 亮点

### 1. 从0到1实现的类Llama2架构模型
模型结构详见modeling_hobo.py

实现了必要功能，并简化代码逻辑，提升可读性
- 支持 FlashAttention-2 加速
- 实现 Grouped Query Attention (GQA)
- 集成 DeepSpeed 分布式训练
- 支持 8bit/4bit 量化训练
- 支持 lora/qlora

模型结构参数:
```python
lm_config = MyConfig(
    vocab_size=151936,
    hidden_size=768,
    num_attention_heads=12,
    num_key_value_heads=4,
    num_hidden_layers=12,
    max_position_embeddings=1024,
    attention_dropout=0.0,
    flash_attn=False,
    rope_theta=10000,
)
# model_size == 0.45b
```
采用qwen2.5的tokenizer和vocab.json

### 2. 基于deepspeed支持多机多卡训练
    
accelerate config中添加配置
    deepspeed_hostfile: ./configs/hostfile

并在hostfile中配置局域网ip节点即可实现基于deepspeed的多机多卡训练

基于deepspeed.pipe.PipelineModule 实现2d并行 (dp + pp)，待开发。。

### 3. grpo 实现

grpo主程序grpo.py

## 项目结构

```
Hobo-LLM/
├── configs/                # accelerate deepspeed configs
├── data/                  # 数据预处理(基于LlamaFactory)
├── scripts/              # shell脚本
├── modeling_hobo.py      # HoboGPT模型架构定义
├── pt.py                # 预训练主程序
├── sft_accelerator.py   # 完整SFT实现(DeepSpeed/8-bit/AMP)
├── sft_amp.py          # 添加了混合精度训练的SFT(AMP/8-bit)
├── sft_vanilla.py      # 最扁平的SFT实现
├── grpo.py            # GRPO主程序（开发中）
├── grpo_trainer.py    # GRPO训练器实现（开发中）
└── reward_funcs.py   # GRPO奖励函数库（开发中）

pt.py              
    完整的pretrain流程
sft_accelerator.py 
    功能最全，集成 DeepSpeed 分布式训练，8-bit 量化训练，amp混合精度训练，wandb实时记录指标和生成效果
sft_amp.py 
    在vanilla基础上重构，集成amp混合进度训练，8-bit 量化训练，amp混合精度训练
sft_vanilla.py
    扁平的sft流程，且具备分布式训练功能。移除了非必要的功能，可读性最好

grpo.py 
    grpo主程序
grpo_trainer.py
    从0到1实现GRPO，参考了trl范式（forward/backward已跑通，开发中）
reward_funcs.py
    GRPO奖励函数库（开发中）
```

## 支持的数据集
| 数据集名称     | 介绍               | 训练流程               |
| ---------------- | -------------------- | -------------------- |
|[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)| ShareGPT中挑选出的GPT4多轮问答数据，多语言问答。|sft               |
|[deepctrl/deepctrl-sft-data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/summary)|匠数大模型SFT数据集是一个由匠数科技精心搜集整理的高质量数据集,包含10M条数据的中文数据集和包含2M条数据的英文数据集|sft               |
|[open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)|Open synthetic reasoning dataset|grpo               |
|[swulling/gsm8k_chinese](https://huggingface.co/datasets/swulling/gsm8k_chinese)|gsm8k chinese|grpo               |
|[trl-lib/tldr](https://huggingface.co/datasets/trl-lib/tldr)|gsm8k chinese|grpo               |
## 支持的模型
| 模型名称     | 介绍               |
| ---------------- | -------------------- |
|[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)|Qwen2.5-0.5B-Instruct是一个基于Qwen2.5架构的模型，具有768个隐藏维度、12个注意力头和12个隐藏层。|

## 环境配置
```bash
git clone https://github.com/XU-YIJIE/hobo-llm.git

# use nvidia ngc image as development environment
docker pull nvcr.io/nvidia/pytorch:24.01-py3

or

conda create -n hobo-llm python=3.10
pip install -r requirements.txt
```

### sft

```bash
# sft
python sft_accelerator.py \
    --from_scratch True \  # use custom model and training from scratch
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
    --seed 1024

# accelerate deepspeed training
bash scripts/train_accelerate_sft.sh
```

### grpo
```bash
# grpo
python grpo.py
```