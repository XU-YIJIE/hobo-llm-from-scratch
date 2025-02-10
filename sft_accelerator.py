from datasets import load_dataset
from loguru import logger
from transformers import (PreTrainedTokenizer, 
                          AutoTokenizer, 
                          DataCollatorForSeq2Seq, 
                          AutoModelForCausalLM, 
                          set_seed,
                          get_linear_schedule_with_warmup,
                          get_scheduler)
from typing import List, Dict, Any
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
import argparse
from accelerate import Accelerator
from accelerate.utils.transformer_engine import convert_model
from transformer_engine.pytorch import fp8_autocast
from functools import partial
import sys
import torch
from tqdm import tqdm
import os
import math
from typing import Optional
import bitsandbytes as bnb
import gc
import wandb
import shutil
import datetime
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

from data.template import Template, get_template_and_fix_tokenizer
from data.preprocess import preprocess_supervised_dataset
from data.aligner import convert_sharegpt, align_dataset, convert_deepctrl
from data.data_args import DataArguments
from data.parser import DatasetAttr
from constants import IGNORE_INDEX
from configuration_model import MyConfig
from modeling_hobo import HoboGPTModelForCausalLM


logger = logger.bind(name="sft")
logger.remove()
# 只显示INFO和ERROR级别
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    filter=lambda record: record["level"].name in ["INFO", "ERROR"]
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model")
    # data process

    parser.add_argument("--input_jsonl", type=list, default=None, help="input_jsonl")
    parser.add_argument("--dataset_dir", type=str, default="dataset/sharegpt_gpt4", help="dataset_dir")
    parser.add_argument("--dataset_name", type=str, default="sharegpt_gpt4", help="dataset_name")
    parser.add_argument("--template", type=str, default="qwen", help="template")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="cutoff_len")
    
    # model
    parser.add_argument("--model_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", help="model_name_or_path")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", help="tokenizer_name_or_path")
    parser.add_argument("--use_fp8", type=bool, default=False, help="use_fp8")
    parser.add_argument("--model_tag", type=str, default=None, help="model_tag")
    parser.add_argument("--model_out_dir", type=str, default="model_ckpts", help="model_out_dir")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning_rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="batch_size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--seed", type=int, default=1024, help="transformer_random_seed")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--from_scratch", type=bool, default=True, help="if to train from scratch")
    parser.add_argument("--use_peft", type=bool, default=False, help="if to use peft")
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora dropout")
    parser.add_argument("--modules_to_save", type=str, default=None, help="List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. ")
    parser.add_argument("--target_modules", type=str, default="all", help="The names of the modules to apply Lora to.")
    parser.add_argument("--max_save", type=int, default=3, help="max save checkpoints")
    
    # logging
    parser.add_argument("--wandb_project", type=str, default="sft_training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb_dir", type=str, default=None, help="wandb local dir, default is ./wandb/")
    parser.add_argument("--log_steps", type=int, default=10, help="log every n steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="save every n steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="save every n steps")
    args = parser.parse_args()
    return args


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_batch_for_fp8(batch: Dict[str, torch.Tensor], tokenizer: PreTrainedTokenizer) -> Dict[str, torch.Tensor]:
    processed_batch = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            processed_batch[k] = v
            continue
        
        # 提前检查维度，避免不必要的内存分配
        if v.dim() >= 2:
            current_shape = list(v.size())
            batch_pad = (8 - current_shape[0] % 8) % 8
            seq_pad = (8 - current_shape[1] % 8) % 8
            needs_padding = batch_pad > 0 or seq_pad > 0
        else:
            needs_padding = False
        
        if k == "attention_mask":
            v = v.bool() if not v.dtype == torch.bool else v
        elif k in ["input_ids", "labels"]:
            v = v.long() if not v.dtype == torch.long else v
        else:
            v = v.float() if not v.dtype == torch.float else v
        
        if v.dim() >= 2 and needs_padding:
            new_shape = current_shape.copy()
            new_shape[0] += batch_pad
            new_shape[1] += seq_pad
            
            if k == "attention_mask":
                fill_value = False
            elif k == "labels":
                fill_value = IGNORE_INDEX
            elif k == "input_ids":
                fill_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            else:
                fill_value = 0
            
            padded_tensor = torch.full(new_shape, fill_value, 
                                    dtype=v.dtype, 
                                    device=v.device)
            
            # 复制原始数据（就地操作）
            padded_tensor[:current_shape[0], :current_shape[1], ...].copy_(v)
            v = padded_tensor
            
            del padded_tensor
            torch.cuda.empty_cache()
        
        if v.dim() >= 2:
            assert v.size(0) % 8 == 0, f"Batch dimension {v.size(0)} is not a multiple of 8 for tensor {k}"
            assert v.size(1) % 8 == 0, f"Sequence dimension {v.size(1)} is not a multiple of 8 for tensor {k}"
        
        if not v.is_contiguous():
            v = v.contiguous()
        
        processed_batch[k] = v
        
    return processed_batch


class Trainer:
    def __init__(self, args: argparse.Namespace):
        # data
        self.input_jsonl = args.input_jsonl
        self.dataset_name = args.dataset_name
        self.dataset_dir = args.dataset_dir
        
        # model
        self.model_name_or_path = args.model_name_or_path
        self.tokenizer_name_or_path = args.tokenizer_name_or_path
        self.use_fp8 = args.use_fp8
        self.from_scratch = args.from_scratch
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_grad_norm = args.max_grad_norm
        self.log_steps = args.log_steps
        self.save_steps = args.save_steps
        self.eval_steps = args.eval_steps
        self.max_save = args.max_save
        
        # peft
        self.use_peft = args.use_peft
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.modules_to_save = args.modules_to_save
        self.target_modules = args.target_modules
        
        seed = args.seed
        set_seed(seed)
        
        self.data_args = DataArguments(template=args.template, 
                                       cutoff_len=args.cutoff_len, 
                                       train_on_prompt=False, 
                                       mask_history=False, 
                                       preprocessing_num_workers=8)
        
        self.accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        self.initializer()
        
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name or f"{args.wandb_project.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.wandb_dir = args.wandb_dir or f"./wandb/{self.wandb_run_name}"
        
        # wandb init
        if self.accelerator.is_main_process:
            if not os.path.exists(self.wandb_dir):
                os.makedirs(self.wandb_dir)
            
            # # 配置wandb日志
            # wandb_log_path = os.path.join(self.wandb_dir, "wandb.log")
            
            # 配置wandb
            # os.environ["WANDB_SILENT"] = "true"  # 静默模式
            # os.environ["WANDB_CONSOLE"] = "off"  # 关闭控制台输出
            
            # 初始化wandb
            self.run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                dir=self.wandb_dir,
                config={
                    "model_name": "MyGPTModel" if self.from_scratch else self.model_name_or_path,
                    "dataset": self.dataset_name,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "max_grad_norm": self.max_grad_norm,
                    "use_fp8": self.use_fp8,
                    "seed": seed,
                    "wandb_dir": self.wandb_dir,
                }
            )
            
            # 打印wandb本地路径信息
            logger.info(f"Wandb local dir: {self.wandb_dir}")
            # logger.info(f"Wandb log file: {wandb_log_path}")

    def initializer(self):
        torch.cuda.empty_cache()  # 初始化前清理显存
        
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
            self.model = HoboGPTModelForCausalLM(config=config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                # torch_dtype=torch.float32,  # load FP32
                low_cpu_mem_usage=True
            )
        
        if self.use_peft:
            target_modules = self.target_modules.split(',') if self.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(self.model, int4=self.use_fp8, int8=self.use_fp8)
            modules_to_save = self.modules_to_save.split(',') if self.modules_to_save is not None else ["lm_head"]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"] if not target_modules else target_modules,
                inference_mode=False,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                modules_to_save=modules_to_save
            )
            self.model = get_peft_model(self.model, peft_config)
            for param in filter(lambda p: p.requires_grad, self.model.parameters()):
                param.data = param.data.to(torch.float32)
            self.model.print_trainable_parameters()

        if self.accelerator.is_main_process:
            logger.info(f"total trainable parameters: {count_parameters(self.model)/1e9:.2f}B")
        
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        if self.use_fp8:
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                convert_model(self.model)
                torch.cuda.empty_cache()  # clear cache after conversion
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=True,
            model_max_length=self.data_args.cutoff_len * 2,  # 如果比model_max_length更长，preprocess环节也会截断
            padding_side="right"
        )
        
        self.create_dataloader()
        self.create_optimizer()

        # Prepare everything with accelerator
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.use_fp8:
            self.optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=self.learning_rate)
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        num_update_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        max_train_steps = self.num_epochs * num_update_steps_per_epoch
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )

    def create_dataloader(self):
        full_dataset = load_dataset(path=self.dataset_dir, data_files=self.input_jsonl, split="train")  # 可选sharegpt或deepctrl作为训练数据
        
        # full_dataset = full_dataset.select(range(100))
        full_dataset = align_dataset(full_dataset, dataset_attr=DatasetAttr(load_from="file", dataset_name=self.dataset_name), data_args=self.data_args)
        
        column_names_to_remove = list(next(iter(full_dataset)).keys())
        template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        preprocess_func = partial(
            preprocess_supervised_dataset,
            template=template,
            tokenizer=self.tokenizer,
            data_args=self.data_args,
        )
        
        with self.accelerator.main_process_first():
            full_dataset = full_dataset.map(
                preprocess_func,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names_to_remove,
                desc="Preprocessing dataset"
            )
            self.accelerator.wait_for_everyone()
            
        if self.accelerator.is_main_process:
            logger.info(f"Preprocessed dataset length: {len(full_dataset)}")
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            label_pad_token_id=IGNORE_INDEX,
        )
        self.train_dataloader = DataLoader(
            full_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            pin_memory=False,  # set pin_memory=False to reduce GPU memory usage
            num_workers=8
        )
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
        if self.accelerator.is_main_process:
            logger.info(f"total training token: {train_token_counter} ({train_token_counter/1e9:.2f}B)")
        
        # clear counter dataloader
        del counter_dataloader
        gc.collect()


    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        with self.accelerator.accumulate(self.model):
            if self.use_fp8:
                batch = prepare_batch_for_fp8(batch, self.tokenizer)
                with fp8_autocast(enabled=True):
                    outputs = self.model(**batch)
                    loss = outputs[0]
            else:
                outputs = self.model(**batch)
                loss = outputs[0]
            
            del outputs
            
            self.accelerator.backward(loss)
            
            if self.max_grad_norm is not None:
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True to clean gradients more thoroughly
            
            return loss.detach().float().cpu()
    
    def train(self):
        total_steps = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                              disable=not self.accelerator.is_local_main_process)
            
            self.model.train()
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                total_loss += loss
                total_steps += 1
                
                # calculate current metrics
                current_loss = total_loss / (step + 1)
                current_ppl = torch.exp(current_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # update progress bar
                progress_bar.set_postfix({
                    'step': f"{step + 1}",
                    'loss': f"{current_loss:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
                # wandb logging
                if self.accelerator.is_main_process and total_steps % self.log_steps == 0:
                    lr_scale = current_lr / self.learning_rate
                    
                    metrics = {
                        "train/loss": current_loss,
                        "train/perplexity": current_ppl,
                        "train/learning_rate": current_lr,
                        "train/lr_scale": lr_scale,
                        "train/epoch": epoch + (step + 1) / len(self.train_dataloader),
                        "train/global_step": total_steps,
                    }
                    
                    # if gradient clipping is used, record gradient norm
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        metrics["train/grad_norm"] = grad_norm
                    
                    # # record GPU memory usage
                    # if torch.cuda.is_available():
                    #     metrics["system/gpu_memory_used"] = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    #     metrics["system/gpu_memory_reserved"] = torch.cuda.max_memory_reserved() / 1024**2  # MB
                    
                    # save best model to wandb
                    if current_loss < best_loss and total_steps % self.save_steps == 0:
                        best_loss = current_loss
                        metrics["best_loss"] = best_loss
                        self.save_manager(epoch + 1, total_steps, current_loss, max_save=self.max_save, prefix="step")
                    
                    wandb.log(metrics, step=total_steps)
                    
                # evaluate
                if self.accelerator.is_main_process and total_steps % self.eval_steps == 0:
                    self.evaluate(checkpoint_dir=f"checkpoints/{self.wandb_run_name}", step_num=total_steps, data_args=self.data_args, from_scratch=self.from_scratch)
                    
                # clear memory every 100 steps
                if step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            gc.collect()

            # wandb logging and save checkpoint after each epoch
            if self.accelerator.is_main_process:
                avg_train_epoch_loss = total_loss / len(self.train_dataloader)
                train_ppl = torch.exp(avg_train_epoch_loss)

                epoch_metrics = {
                    "train/epoch_loss": avg_train_epoch_loss,
                    "train/epoch_perplexity": train_ppl,
                    "train/epoch": epoch + 1,
                }
                
                self.save_manager(epoch + 1, total_steps, avg_train_epoch_loss, max_save=self.max_save, prefix="epoch")
                    
                # # record model size
                # model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                # epoch_metrics["model_size_mb"] = model_size_mb
                
                wandb.log(epoch_metrics, step=total_steps)
                
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}:")
                logger.info(f"  Average Loss: {avg_train_epoch_loss:.4f}")
                logger.info(f"  Perplexity: {train_ppl:.2f}")
                logger.info(f"  Learning Rate: {current_lr:.2e}")
            
        self.accelerator.wait_for_everyone()
        
        # close wandb after training
        if self.accelerator.is_main_process:
            # save final model to wandb
            self.save_manager(self.num_epochs, total_steps, avg_train_epoch_loss, max_save=self.max_save, prefix="final")
            wandb.finish()

    def evaluate(self, checkpoint_dir, step_num, data_args: DataArguments, from_scratch=True):
        if not os.path.exists(checkpoint_dir):
            logger.error(f"checkpoints dir not found: {checkpoint_dir}")
            return
            
        step_dirs = [d for d in os.listdir(checkpoint_dir) 
                    if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("step_")]
        if not step_dirs:
            logger.error("no step checkpoints found")
            return
            
        step_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        best_model_dir = os.path.join(checkpoint_dir, step_dirs[-1])
        logger.info(f"load best model: {best_model_dir}")
        
        try:
            if from_scratch:
                eval_model = HoboGPTModelForCausalLM.from_pretrained(
                    best_model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to("cpu")
            else:
                eval_model = AutoModelForCausalLM.from_pretrained(
                    best_model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to("cpu")
            
            eval_model.eval()
            eval_tokenizer = AutoTokenizer.from_pretrained(
                best_model_dir,
                trust_remote_code=True,
                model_max_length=data_args.cutoff_len * 2,
                padding_side="left"
            )
            
            test_queries = [
                "请介绍一下你自己。",
                "解释一下量子计算的基本原理。", 
                "写一首关于春天的诗。",
                "如何实现快速排序算法？请用Python代码示例。",
                "总结一下中国近代史上的重要事件。",
                "你能扮演一个心理咨询师吗？我最近工作压力很大。",
                "请用简单的语言解释相对论。",
                "帮我写一个购物清单模板，包含常见的生活用品。",
                "如何制作一道经典的红烧肉？",
                "给我讲一个有趣的历史故事。",
                "如何培养良好的学习习惯？",
                "解释一下区块链技术的工作原理。",
                "帮我写一封商务邮件，主题是产品推广。",
                "如何在家种植盆栽植物？",
                "分析一下当前的全球经济形势。",
                "教我几个实用的英语口语表达。",
                "如何写一篇有吸引力的小说开头？",
                "解释下什么是人工智能的深度学习？",
                "推荐几本值得阅读的经典文学作品。",
                "如何准备一次成功的工作面试？"
            ]
            
            gen_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": eval_tokenizer.pad_token_id,
                "eos_token_id": eval_tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1
            }
            
            logger.info("\n" + "="*50 + " start evaluation " + "="*50)
            
            input_strs = []
            for query in test_queries: 
                MESSAGES = [
                    {"role": "user", "content": query}
                ]
                input_str = eval_tokenizer.apply_chat_template(MESSAGES, template=eval_tokenizer.chat_template, tokenize=False, add_generation_prompt=True)
                input_strs.append(input_str) 
            inputs = eval_tokenizer(input_strs, return_tensors="pt", add_special_tokens=True, padding=True)
            outputs = eval_model.generate(
                            **inputs,
                            **gen_config
                        )
            responses = eval_tokenizer.batch_decode(outputs, skip_special_tokens=True) 
            responses = [response[len(eval_tokenizer.decode(inputs["input_ids"][index], skip_special_tokens=True)):] for index, response in enumerate(responses)]
            del outputs
            gc.collect()
            
            data = [[step_num, input_str, response] for input_str, response in zip(input_strs, responses)]
            
            if not hasattr(self, 'eval_history'):
                self.eval_history = []
            
            self.eval_history.extend(data)
            
            eval_table = wandb.Table(
                columns=["Step", "InputStr", "Response"],
                data=self.eval_history
            )
            
            wandb.log({"eval_results": eval_table}, commit=True)

            logger.info("\n" + "="*50 + " evaluation completed " + "="*50)
            
        except Exception as e:
            logger.error(e)
        
        finally:
            if 'eval_model' in locals():
                del eval_model
            if 'eval_tokenizer' in locals():
                del eval_tokenizer
            gc.collect()

    def save_manager(self, current_epoch, current_steps, current_loss, max_save=None, prefix=None):
        checkpoints_dir = f"checkpoints/{self.wandb_run_name}"
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
        if max_save is not None:
            step_dirs = [d for d in os.listdir(checkpoints_dir) 
                        if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith(prefix)]
            step_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
            
            while len(step_dirs) >= max_save:
                oldest_dir = os.path.join(checkpoints_dir, step_dirs[0])
                shutil.rmtree(oldest_dir)
                step_dirs.pop(0)
        
        save_dir = os.path.join(checkpoints_dir, 
                                f"{prefix}_{current_steps}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        training_state = {
            'step': current_steps,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': current_loss,
            'epoch': current_epoch,
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        # self.run.save(os.path.join(save_dir, "*"), base_path=checkpoints_dir)


if __name__ == "__main__":
    args = parse_args()
    args.use_peft = True
    args.from_scratch = False
    data_args = DataArguments(template=args.template, 
                              cutoff_len=args.cutoff_len, 
                              train_on_prompt=False, 
                              mask_history=False, 
                              preprocessing_num_workers=8)
    trainer = Trainer(args)
    trainer.train()
    # trainer.evaluate(checkpoint_dir=f"checkpoints/sft_training_20250203_163642", step_num=100, data_args=data_args, from_scratch=True)
