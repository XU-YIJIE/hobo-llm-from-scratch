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
from functools import partial
import sys
import torch
from tqdm import tqdm
import os
from accelerate.utils.transformer_engine import convert_model
from transformer_engine.pytorch.fp8 import fp8_autocast
import math
from typing import Optional
from torch.cuda.amp import GradScaler, autocast
import bitsandbytes as bnb

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
# # 只显示INFO和ERROR级别
# logger.add(
#     sys.stderr,
#     format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
#     level="INFO",
#     filter=lambda record: record["level"].name in ["INFO", "ERROR"]
# )

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model")
    # data process
    parser.add_argument("--input_jsonl", type=str, default="dataset/sharegpt_gpt4/sharegpt_zh_38K_format.jsonl", help="input_jsonl")
    parser.add_argument("--dataset_name", type=str, default="sharegpt_gpt4", help="dataset_name")
    parser.add_argument("--template", type=str, default="qwen", help="template")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="cutoff_len")

    # model
    parser.add_argument("--use_fp8", type=bool, default=False, help="use_fp8")
    parser.add_argument("--model_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", help="model_name_or_path")
    parser.add_argument("--model_tag", type=str, default=None, help="model_tag")
    parser.add_argument("--model_out_dir", type=str, default="model_ckpts", help="model_out_dir")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay")
    parser.add_argument("--num_epochs", type=int, default=3, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="batch_size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument("--seed", type=int, default=1024, help="transformer_random_seed")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--from_scratch", type=bool, default=True, help="if to train from scratch")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="clip grad norm")
    args = parser.parse_args()
    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_batch_for_fp8(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """准备批次数据以适应FP8要求"""
    processed_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = pad_to_multiple(v, 8, pad_dims=[0, 1])
        processed_batch[k] = v
    return processed_batch


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 8, pad_dims: Optional[List[int]] = None) -> torch.Tensor:
    """将张量填充到指定倍数
    Args:
        tensor: 需要填充的张量
        multiple: 填充的倍数
        pad_dims: 需要填充的维度列表，如果为None则填充所有维度
    """
    size = list(tensor.size())
    padded_size = size.copy()
    
    # 如果指定了需要填充的维度，则只填充这些维度
    if pad_dims is not None:
        for dim in pad_dims:
            if dim < len(size):
                padded_size[dim] = math.ceil(size[dim] / multiple) * multiple
    else:
        padded_size = [(math.ceil(s / multiple) * multiple) for s in size]
    
    if size == padded_size:
        return tensor
    
    padding = []
    for orig, padded in zip(reversed(size), reversed(padded_size)):
        padding.extend([0, padded - orig])
        
    padded_tensor = torch.nn.functional.pad(tensor, padding)
    # logger.info(f"Tensor padded from {size} to {list(padded_tensor.size())}")
    
    return padded_tensor


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.from_scratch = args.from_scratch
        self.model_name_or_path = args.model_name_or_path
        self.input_jsonl = args.input_jsonl
        self.dataset_name = args.dataset_name
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_fp8 = args.use_fp8
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_grad_norm = args.max_grad_norm
        seed = args.seed
        set_seed(seed)
        
        self.data_args = DataArguments(template=args.template, 
                                       cutoff_len=args.cutoff_len, 
                                       train_on_prompt=False, 
                                       mask_history=False, 
                                       preprocessing_num_workers=8)
        
        self.scaler = GradScaler()
        self.initializer()

    def initializer(self):
        if self.from_scratch:
            config = MyConfig(
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
            self.model = HoboGPTModelForCausalLM(config=config).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            ).to(self.device)
        logger.info(f"total trainable parameters: {count_parameters(self.model)/1e9:.2f}B")
        
        self.model.gradient_checkpointing_enable()  # 时间换空间, 減少显存消耗 # model.enable_input_require_grads()
        self.model.config.use_cache = False  # use cache用于在內容生成时缓存历史kv在显存中, 空间换时间, 減少无谓的计算
        
        if self.use_fp8:
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                convert_model(self.model)  # to transformer_engine
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        
        self.create_dataloader()
        self.create_optimizer(use_8bit_adam=self.use_fp8)

    def create_optimizer(self, use_8bit_adam: bool = False) -> torch.optim.Optimizer:
        # 将参数分为需要和不需要权重衰减的两组
        decay_parameters = []
        no_decay_parameters = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name.lower() for nd in ['bias', 'layernorm', 'ln_f']):
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
                    
        adam_kwargs = {
                "lr": self.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
        }
        if use_8bit_adam:
            optimizer_cls = bnb.optim.AdamW8bit
            optimizer_grouped_parameters = [
                {
                    "params": decay_parameters,
                    "weight_decay": self.weight_decay,
                    "use_8bit": True,  # 大参数使用8-bit
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                    "use_8bit": False,  # 小参数保持FP32
                }
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": decay_parameters,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                }
            ]
            optimizer_cls = torch.optim.AdamW
        
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            **adam_kwargs
        )
        
        num_update_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        max_train_steps = self.num_epochs * num_update_steps_per_epoch
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
        
    def create_dataloader(self):
        full_dataset = load_dataset("json", data_files=self.input_jsonl, split="train")
        # full_dataset = full_dataset.select(range(1000))
        template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        full_dataset = align_dataset(full_dataset, dataset_attr=DatasetAttr(load_from="file", dataset_name=self.dataset_name), data_args=self.data_args)
        column_names_to_remove = list(next(iter(full_dataset)).keys())
        preprocess_func = partial(
            preprocess_supervised_dataset,
            template=template,
            tokenizer=self.tokenizer,
            data_args=self.data_args,
        )
        full_dataset = full_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names_to_remove,
            desc="Preprocessing dataset"
        )
        logger.info(f"Preprocessed dataset length: {len(full_dataset)}")
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            label_pad_token_id=IGNORE_INDEX,
        )
        self.train_dataloader = DataLoader(
            full_dataset,
            shuffle=False,
            # sampler=RandomSampler(full_dataset, sample_size=100),
            collate_fn=data_collator,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=self.data_args.preprocessing_num_workers
        )
        
        # 计算总训练token数
        counter_dataloader = DataLoader(
            full_dataset,
            shuffle=False,
            # sampler=RandomSampler(full_dataset, sample_size=100),
            collate_fn=data_collator,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=self.data_args.preprocessing_num_workers
        )
        train_token_counter = 0
        for counter_batch in counter_dataloader:
            train_token_counter += int(counter_batch["attention_mask"].sum())
        train_token_counter *= self.num_epochs
        logger.info(f"total training token: {train_token_counter} ({train_token_counter/1e9:.2f}B)")

    def forward_step(self, batch: Dict[str, torch.Tensor]) -> float:
        # 将数据移到GPU
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 如果使用FP8，确保张量维度符合要求
        if self.use_fp8:
            batch = prepare_batch_for_fp8(batch)
        
        # 根据是否使用FP8选择不同的精度上下文
        if self.use_fp8:
            with fp8_autocast(enabled=True):
                outputs = self.model(**batch)
        else:
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(**batch)
        loss = outputs[0]
        del outputs
        
        # 使用scaler来处理反向传播。计算梯度并累积
        self.scaler.scale(loss).backward()
        
        return loss.detach().float()
    
    def train(self):
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                total_loss += self.forward_step(batch)
                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)  # 梯度更新
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'step': f"{step + 1}",
                        'loss': f"{total_loss / (step + 1):.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })
            
            torch.cuda.empty_cache()

            avg_train_epoch_loss = total_loss / len(self.train_dataloader)
            train_ppl = torch.exp(avg_train_epoch_loss)
            logger.info(f"{epoch=}: {train_ppl=} {avg_train_epoch_loss=}")

    
if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()