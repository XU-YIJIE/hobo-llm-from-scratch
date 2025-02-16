# Hobo-LLM: from sft to grpo

![License](https://img.shields.io/badge/License-Apache%202.0-green)

Implementation of LLM training code (pt/sft/grpo) from scratch. Focused on balancing educational value and practicality, suitable for small to medium-scale training scenarios, supporting multi-node multi-GPU training. Minimizes third-party library dependencies to enhance readability while ensuring versatility.

## Highlights

### 1. Llama2-like Architecture Model Implemented from Scratch
Model structure details in modeling_hobo.py

Implements essential features with simplified code logic for better readability
- Supports FlashAttention-2 acceleration
- Implements Grouped Query Attention (GQA)
- Integrates DeepSpeed distributed training
- Supports 8bit/4bit quantization training
- Supports lora/qlora

Model structure parameters:
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
Uses qwen2.5's tokenizer and vocab.json

### 2. Multi-node Multi-GPU Training Support with DeepSpeed
    
- Configure LAN IP nodes in the hostfile to enable multi-node multi-GPU training based on deepspeed

- 2D parallelism (dp + pp) implementation based on deepspeed.pipe. PipelineModule is under development...

### 3. GRPO Implementation

Controllable text generation training based on GRPO (Grouped Reward Policy Optimization):

- Summary length control: Using TLDR dataset to train models to generate summaries of specified length
- Zero-shot transfer training: [如何0样本基于grpo训练一个夸夸机器人，单卡24GB显存耗时15分钟](https://github.com/XU-YIJIE/grpo-flat)

## Project Structure

```
Hobo-LLM/
├── configs/                # accelerate deepspeed configs
├── data/                  # Data preprocessing (based on LlamaFactory)
├── scripts/              # shell scripts
├── modeling_hobo.py      # HoboGPT model architecture definition
├── pt.py                # Pre-training main program
├── sft_accelerator.py   # Complete SFT implementation (DeepSpeed/8-bit/AMP)
├── sft_amp.py          # SFT with mixed precision training (AMP/8-bit)
├── sft_vanilla.py      # Simplified SFT implementation
├── grpo.py            # GRPO main program (in development)
├── grpo_trainer.py    # GRPO trainer implementation (in development)
└── reward_funcs.py   # GRPO reward function library (in development)
```

| File Name | Description |
|--------|----------|
| pt.py | Complete pre-training workflow |
| sft_accelerator.py | Most comprehensive implementation, integrating DeepSpeed distributed training, 8-bit quantization training, amp mixed precision training, and wandb real-time metrics and generation effect tracking |
| sft_amp.py | Refactored from vanilla, integrating amp mixed precision training and 8-bit quantization training |
| sft_vanilla.py | Concise SFT workflow with distributed training capability. Non-essential features removed to improve readability |
| grpo.py | GRPO main program |
| grpo_trainer.py | GRPO implementation from scratch |
| reward_funcs.py | GRPO reward function library |

## Supported Datasets
| Dataset Name     | Description               | Training Process               |
| ---------------- | -------------------- | -------------------- |
|[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)| Multi-round Q&A data selected from ShareGPT's GPT4 interactions, multilingual.|sft               |
|[deepctrl/deepctrl-sft-data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/summary)|DeepCtrl SFT dataset is a high-quality dataset carefully curated by DeepCtrl Technology, including 10M Chinese data entries and 2M English data entries|sft               |
|[open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)|Open synthetic reasoning dataset|grpo               |
|[swulling/gsm8k_chinese](https://huggingface.co/datasets/swulling/gsm8k_chinese)|gsm8k chinese|grpo               |
|[trl-lib/tldr](https://huggingface.co/datasets/trl-lib/tldr)|The TL;DR dataset is a processed version of Reddit posts, specifically curated to train models using the TRL library for summarization tasks.|grpo               |

## Supported Models
| Model Name     | Description               |
| ---------------- | -------------------- |
|[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)|Qwen2.5-0.5B-Instruct is a model based on the Qwen2.5 architecture with 768 hidden dimensions, 12 attention heads, and 12 hidden layers.|

## Environment Setup
```bash
git clone https://github.com/XU-YIJIE/hobo-llm.git

# use nvidia ngc image as development environment
docker pull nvcr.io/nvidia/pytorch:24.01-py3

or

conda create -n hobo-llm python=3.10
pip install -r requirements.txt
```

### SFT

```bash
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

### GRPO

```bash
accelerate launch grpo.py
```


## TODO
- [ ] Implement MOE/MLA
- [ ] Implement training with deepspeed pipemodule paradigm to achieve 2D parallelism (pp + dp)
- [ ] Train a 0.5B chat model from scratch
- [ ] Add model evaluation code to generate batch performance reports


## Acknowledgements

- [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)
- [jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [huggingface/trl](https://github.com/huggingface/trl)

