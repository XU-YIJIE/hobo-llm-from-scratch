from datasets import load_dataset
from loguru import logger
from transformers import (PreTrainedTokenizer, 
                          AutoTokenizer, 
                          DataCollatorForSeq2Seq, 
                          AutoModelForCausalLM, 
                          set_seed,
                          get_linear_schedule_with_warmup)
from typing import List, Dict, Any
from torch.utils.data import DataLoader, RandomSampler
import argparse
from accelerate import Accelerator
from functools import partial
import sys
import torch
from tqdm import tqdm
import os

from data.template import Template, get_template_and_fix_tokenizer
from data.preprocess import preprocess_supervised_dataset
from data.aligner import convert_sharegpt, align_dataset
from data.data_args import DataArguments
from data.parser import DatasetAttr
from constants import IGNORE_INDEX
from configuration_model import MyConfig
from modeling_hobo import HoboGPTModelForCausalLM

logger = logger.bind(name="sft")
# logger.remove()
# logger.add(
#     sys.stderr,
#     format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
#     level="INFO",
#     filter=lambda record: record["level"].name in ["INFO", "ERROR"]
# )


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model")
    # data process

    parser.add_argument("--input_jsonl", type=str, default="dataset/sharegpt_gpt4/sharegpt_zh_38K_format.jsonl", help="input_jsonl")
    parser.add_argument("--dataset_name", type=str, default="sharegpt_gpt4", help="dataset_name")
    parser.add_argument("--template", type=str, default="qwen", help="template")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="cutoff_len")
    
    # model
    parser.add_argument("--model_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", help="model_name_or_path")
    parser.add_argument("--model_tag", type=str, default=None, help="model_tag")
    parser.add_argument("--model_out_dir", type=str, default="model_ckpts", help="model_out_dir")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning_rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="batch_size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument("--seed", type=int, default=1024, help="transformer_random_seed")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--from_scratch", type=bool, default=True, help="if to train from scratch")
    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainer():
    args = parse_args()
    model_name_or_path = args.model_name_or_path
    model_tag = args.model_tag
    model_out_dir = args.model_out_dir
    input_jsonl = args.input_jsonl
    dataset_name = args.dataset_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size  # per device
    lr = args.learning_rate
    gradient_accumulation_steps = args.gradient_accumulation_steps
    seed = args.seed
    device = args.device
    from_scratch = args.from_scratch
    template = args.template
    cutoff_len = args.cutoff_len
    
    data_args = DataArguments(template=template, cutoff_len=cutoff_len, train_on_prompt=False, mask_history=False, preprocessing_num_workers=8)

    set_seed(seed)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    if accelerator.is_main_process:
        logger.info("Training Arguments:")
        for arg, value in vars(args).items():
            logger.info(f"  {arg}: {value}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    full_dataset = load_dataset("json", data_files=input_jsonl, split="train")
    # full_dataset = full_dataset.select(range(1000))
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    full_dataset = align_dataset(full_dataset, dataset_attr=DatasetAttr(load_from="file", dataset_name=dataset_name), data_args=data_args)
    column_names_to_remove = list(next(iter(full_dataset)).keys())
    preprocess_func = partial(
        preprocess_supervised_dataset,
        template=template,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    with accelerator.main_process_first():  # 保证打印
        full_dataset = full_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names_to_remove,
            desc="Preprocessing dataset"
        )
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info(f"Preprocessed dataset length: {len(full_dataset)}")
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX,
    )

    train_dataloader = DataLoader(
        full_dataset,
        shuffle=False,
        # sampler=RandomSampler(full_dataset, sample_size=100),
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=data_args.preprocessing_num_workers
    )
    counter_dataloader = DataLoader(
        full_dataset,
        shuffle=False,
        # sampler=RandomSampler(full_dataset, sample_size=100),
        collate_fn=data_collator,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=data_args.preprocessing_num_workers
    )
    train_token_counter = 0
    for counter_batch in counter_dataloader:
        train_token_counter += int(counter_batch["attention_mask"].sum())
    train_token_counter *= num_epochs
    
    if accelerator.is_main_process:
        logger.info(f"total training token: {train_token_counter} ({train_token_counter/1e9:.2f}B)")

    if from_scratch:
        config = MyConfig(
            vocab_size=151936,
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=4,
            num_hidden_layers=12,
            max_position_embeddings=2048,
            attention_dropout=0.0,
            flash_attn=False,
            rope_theta=10000,
        )
        model = HoboGPTModelForCausalLM(config=config).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(device)

    if accelerator.is_main_process:
        logger.info(f"total trainable parameters: {count_parameters(model)/1e9:.2f}B")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,  # warmup可调小
        num_training_steps=len(train_dataloader) * num_epochs,
    )
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # 先使用accelerator准备模型和优化器
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )
    
    if accelerator.is_main_process:
        logger.info("Model Architecture:")
        logger.info("=" * 50)
        logger.info(f"{model}")
        logger.info("=" * 50)
    
        
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs[0]
                del outputs
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.detach().float()
                progress_bar.set_postfix({
                    'step': f"{step + 1}",
                    'loss': f"{total_loss / (step + 1):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
        torch.cuda.empty_cache()

        avg_train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(avg_train_epoch_loss)
        logger.info(f"{epoch=}: {train_ppl=} {avg_train_epoch_loss=}")

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        model = accelerator.unwarp_model(model)
        model_id = f"{model_tag}_{model_name_or_path}".replace("/", "_")
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)
        model.save_pretrained(
            os.path.join(model_out_dir, model_id), safe_serialization=False
        )  # 避免safe_tensor存储
        tokenizer.save_pretrained(os.path.join(model_out_dir, model_id))


if __name__ == "__main__":
    trainer()