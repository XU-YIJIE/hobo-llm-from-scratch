#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Optional
from pathlib import Path
from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    trainers,
    normalizers,
    pre_tokenizers,
    processors,
    models,
)
from transformers import PreTrainedTokenizerFast

QWEN_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]

CHAT_TEMPLATE = """{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nYou are Hobo, a helpful AI assistant.<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == "user" or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""


def format_chat_message(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>"


def load_training_data(
    dataset_dir: str,
    input_jsonl: List[str],
) -> List[str]:
    dataset = load_dataset(path=dataset_dir, data_files=input_jsonl, split="train")
    texts = []
    for item in dataset:
        if item["history"]:
            for user_msg, assistant_msg in item["history"]:
                texts.append(format_chat_message("user", user_msg))
                texts.append(format_chat_message("assistant", assistant_msg))
        if item["instruction"]:
            texts.append(format_chat_message("system", item["instruction"]))
        if item["input"]:
            texts.append(format_chat_message("user", item["input"]))
        if item["output"]:
            texts.append(format_chat_message("assistant", item["output"]))
    
    return texts


def train_and_save_tokenizer(
    texts: List[str],
    vocab_size: int = 8000,  # with smaller vocab size
    min_frequency: int = 2,
    save_dir: str = "tokenizer",
    special_tokens: Optional[List[str]] = None
) -> Tokenizer:
    """类Qwen的BPE tokenizer"""
    if special_tokens is None:
        special_tokens = QWEN_SPECIAL_TOKENS
    
    tokenizer = Tokenizer(models.BPE(
        dropout=None,  # 不使用dropout，保持分词的确定性
        fuse_unk=False,  # 不融合未知token，保持每个未知token独立
    ))
    
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(r"\s+", " "),  # 合并多个空格
        normalizers.Strip(),  # 去除首尾空格
        normalizers.NFKC(),  # Unicode标准化
        normalizers.Replace(r"[^\x00-\x7F\u4E00-\u9FFF]", ""),  # 只保留ASCII和中文字符
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),  # 按utf8字节分割
        pre_tokenizers.Digits(individual_digits=True),  # 数字单独分词
    ])
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # ASCII + UTF-8
    )
    
    # train tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A",
        pair="$A $B",  # 句对输入的处理
        special_tokens=[
            ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
            ("<|im_start|>", tokenizer.token_to_id("<|im_start|>")),
            ("<|im_end|>", tokenizer.token_to_id("<|im_end|>")),
        ],
    )
    
    # save tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # convert to hf format
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=None,  # not use bos_token
        eos_token="<|im_end|>",
        unk_token=None,  # not use unk_token
        pad_token="<|endoftext|>",
        model_max_length=8192,
        clean_up_tokenization_spaces=False,
    )
    
    # set chat template
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(save_dir)
    
    return tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="", help="dataset directory")
    parser.add_argument("--input_jsonl", type=str, default="", help="input jsonl file")
    parser.add_argument("--vocab_size", type=int, default=8000, help="vocab size")
    parser.add_argument("--min_frequency", type=int, default=2, help="min frequency")
    parser.add_argument("--save_dir", type=str, default="tokenizer", help="save directory")
    parser.add_argument("--num_proc", type=int, default=4, help="number of processes for loading data")
    args = parser.parse_args()
    return args

def main(args):
    # load training data
    print(f"loading training data: {args.dataset_dir}")
    texts = load_training_data(
        dataset_dir=args.dataset_dir,
        input_jsonl=args.input_jsonl,
    )
    
    # train tokenizer
    print(f"Start training tokenizer, vocab size: {args.vocab_size}")
    train_and_save_tokenizer(
        texts=texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_dir=args.save_dir
    )
    
    print(f"Tokenizer training completed, saved to: {args.save_dir}")
    
    # test tokenizer
    test_messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"},
        {"role": "assistant", "content": "你好！我是一个AI助手，很高兴为你服务。"},
        {"role": "user", "content": "你能做什么？"},
        {"role": "assistant", "content": "我可以帮助你回答问题、编写代码、进行文本分析等任务。"}
    ]
    
    print("\nTest results:")
    # use hf tokenizer for testing, because it supports chat_template
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.save_dir)
    formatted_chat = hf_tokenizer.apply_chat_template(test_messages, tokenize=False)
    print(f"\nFormatted chat text:\n{formatted_chat}")
    
    # test tokenization results
    tokens = hf_tokenizer.tokenize(formatted_chat)
    print(f"\nTokenization results:\n{tokens}")

if __name__ == "__main__":
    args = parse_args()
    args.dataset_dir = "dataset/deepctrl-sft-data"
    args.input_jsonl = ["sft_data_en.jsonl", "sft_data_zh.jsonl"]
    main(args)