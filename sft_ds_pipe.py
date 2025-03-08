import datetime
from functools import partial
import itertools
import math
import os
import argparse
import shutil
import deepspeed
from deepspeed.pipe import PipelineModule
import deepspeed.comm as dist
from dataclasses import dataclass
from typing import Any, Optional, Union
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from loguru import logger
import wandb

from constants import IGNORE_INDEX
from data.aligner import align_dataset
from data.data_args import DataArguments
from data.parser import DatasetAttr
from data.preprocess import preprocess_supervised_dataset
from data.template import get_template_and_fix_tokenizer
# from pipemodule_qwen2 import get_deepspeed_pipemodule
from pipemodule_qwen2 import get_deepspeed_pipemodule


@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if labels is not None:
            if not self.padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_label_length = self.max_length if self.max_length is not None else max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label[:max_label_length] + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label[:max_label_length]
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label[:max_label_length],
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label[:max_label_length],
                            ]
                        )
                        for label in labels
                    ]

        if batch.get("labels", None) is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        else:
            batch["labels"] = None

        input_ids = batch["input_ids"]
        device = input_ids.device
        attention_mask = batch["attention_mask"].to(device)
        
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0).expand_as(input_ids)
        
        if batch["labels"] is not None:
            labels = batch["labels"].to(device)
        else:
            labels = None

        return (input_ids, attention_mask, position_ids, labels), None


def parse_args():
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", required=True)
    parser.add_argument("--model_out_dir", type=str, default="output")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    
    # dataset
    parser.add_argument("--dataset_dir", type=str, default="dataset/sharegpt_gpt4")
    parser.add_argument("--input_jsonl", type=str, default="sharegpt_gpt4.jsonl")
    parser.add_argument("--dataset_name", type=str, default="sharegpt_gpt4")
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    
    # training
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10)
    
    # parallel
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--checkpoint_activations", action="store_true")
    parser.add_argument("--checkpoint_num_layers", type=int, default=1)
    parser.add_argument("--pipe_chunk_size", type=int, default=1)
    parser.add_argument("--steps_per_print", type=int, default=1000)
    
    # mixed precision
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    
    # logging
    parser.add_argument("--wandb_project", type=str, default="sft_ds_pipe_training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb_dir", type=str, default=None, help="wandb local dir, default is ./wandb/")
    parser.add_argument("--log_steps", type=int, default=10, help="log every n steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="save every n steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="save every n steps")
    
    # DeepSpeed config
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, args):
        deepspeed.init_distributed()
        self.args = args
        
        # model
        self.model_name_or_path = args.model_name_or_path
        self.model_out_dir = args.model_out_dir
        self.num_epochs = args.num_epochs
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.max_seq_len = args.max_seq_len
        
        # dataset
        self.dataset_dir = args.dataset_dir
        self.input_jsonl = args.input_jsonl
        self.dataset_name = args.dataset_name
        self.cutoff_len = args.cutoff_len
        self.template = args.template
        self.preprocessing_num_workers = args.preprocessing_num_workers
        
        # training
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.warmup_steps = args.warmup_steps
        
        # parallel
        self.pp_size = args.pp_size
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers
        self.pipe_chunk_size = args.pipe_chunk_size
        self.steps_per_print = args.steps_per_print
        
        self.local_rank = args.local_rank
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # mixed precision
        self.fp16 = args.fp16
        self.bf16 = args.bf16
        
        # logging
        self.log_steps = args.log_steps
        self.save_steps = args.save_steps
        self.eval_steps = args.eval_steps
        
        # wandb
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name or f"{args.wandb_project.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.wandb_dir = args.wandb_dir or f"./wandb/{self.wandb_run_name}"
        
        if self.global_rank == 0:
            if not os.path.exists(self.wandb_dir):
                os.makedirs(self.wandb_dir)
            self.run = wandb.init(
                project=self.wandb_project, name=self.wandb_run_name,
                dir=self.wandb_dir, config=vars(self.args)
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = get_deepspeed_pipemodule(args)
        self.ds_config = self.create_ds_config()
        
        self.train_dataloader, self.total_training_steps = self.initialize_dataset_and_dataloader()
        self.model_engine, _, _, _ = deepspeed.initialize(
            model=self.model,
            config=self.ds_config,
            model_parameters=self.model.parameters()
        )
        
        if self.gradient_accumulation_steps != self.model_engine.gradient_accumulation_steps():
            if self.global_rank == 0:
                logger.warning(f"Gradient accumulation steps ({self.gradient_accumulation_steps}) "
                               f"does not match DeepSpeed engine's gradient accumulation steps "
                               f"({self.model_engine.gradient_accumulation_steps()})")
                
                logger.warning(f"Using DeepSpeed engine's gradient accumulation steps: "
                               f"{self.model_engine.gradient_accumulation_steps()}")
        
        actual_gas = self.model_engine.gradient_accumulation_steps()
        logger.info(f"Actual gradient accumulation steps: {actual_gas}")
        logger.info(f"Pipeline micro batch count: {actual_gas}")
        
    
    def create_ds_config(self):
        ds_config = {
            "train_micro_batch_size_per_gpu": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_print": self.steps_per_print,
            
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                }
            },
            
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": self.warmup_steps
                }
            },
            
            "fp16": {
                "enabled": self.fp16,
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            
            "bf16": {
                "enabled": self.bf16
            },
            
            "gradient_clipping": 1.0,
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
            
            "zero_optimization": {
                "stage": 0,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": False,  # 流水线并行中设为False
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
                # "offload_optimizer": {"device": "cpu", "pin_memory": True},
            },
            
            "activation_checkpointing": {
                "partition_activations": False,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": self.checkpoint_num_layers if self.checkpoint_activations else 0,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            
            "pipeline": {
                "enabled": True,
                "num_stages": self.pp_size,
                "pipe_chunk_size": self.pipe_chunk_size,
                "num_micro_batches": self.per_device_train_batch_size,
                "activation_checkpoint_interval": self.checkpoint_num_layers if self.checkpoint_activations else 0,
                "pipe_schedule": "forward-backward",
                # "pipe_schedule": "interleaved",
                "communication_data_type": "fp16" if self.fp16 else "bf16" if self.bf16 else "fp32",
                "timeout": 3600.0,
                "barrier_token_comm": True
            },
        }
        
        return ds_config


    def initialize_dataset_and_dataloader(self):
        dataset = load_dataset(path=self.dataset_dir, data_files=self.input_jsonl, split="train")
        dataset = dataset.select(range(100))
        data_args = DataArguments(template=self.template, 
                                  cutoff_len=self.cutoff_len, train_on_prompt=False, 
                                  mask_history=False, preprocessing_num_workers=self.preprocessing_num_workers)
        template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        
        dataset = align_dataset(
            dataset,
            dataset_attr=DatasetAttr(load_from="file", dataset_name=self.dataset_name),
            data_args=data_args
        )
        
        column_names_to_remove = list(next(iter(dataset)).keys())
        
        preprocess_func = partial(preprocess_supervised_dataset, template=template, 
                                  tokenizer=self.tokenizer, data_args=data_args)
        dataset = dataset.map(preprocess_func, 
                              batched=True, num_proc=self.preprocessing_num_workers, remove_columns=column_names_to_remove, desc="preprocessing dataset")
        
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            drop_last=True,
        )
        
        # TODO 实现sequence_packing
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            label_pad_token_id=IGNORE_INDEX,
            max_length=self.max_seq_len,  # 使用args中的max_seq_len
            padding="max_length"  # 使用max_length填充策略，固定长度适配pp
        )
        
        train_dataloader = DataLoader(
            dataset,
            sampler=train_sampler,
            collate_fn=data_collator,
            batch_size=self.per_device_train_batch_size,
            pin_memory=False,
            num_workers=self.preprocessing_num_workers,
            drop_last=True,
        )
        
        total_training_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        
        return train_dataloader, total_training_steps
    
    def train(self):
        steps_per_epoch = len(self.train_dataloader) // self.model_engine.gradient_accumulation_steps()  # 优化器步数
        if self.global_rank == 0:
            logger.info(f"Total training steps = {steps_per_epoch}, Total micro batch count = {len(self.train_dataloader)}")
            
        total_steps = 0
        for epoch in range(1, self.num_epochs + 1):
            self.model_engine.train()
            self.train_dataloader.sampler.set_epoch(epoch)
            data_iterator = iter(self.train_dataloader)
            
            for step in range(1, steps_per_epoch + 1):
                loss = self.model_engine.train_batch(data_iter=data_iterator)
                dist.barrier()
                if self.global_rank == 0 and step % 10 == 0:
                    logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")
                total_steps += 1
                    
            self.save_manager(current_epoch=epoch, current_steps=total_steps, current_loss=loss.item(), max_save=2, prefix=f"epoch")
            dist.barrier()

    def save_manager(self, current_epoch, current_steps, current_loss, max_save=None, prefix=None):
        checkpoints_dir = f"{self.model_out_dir}/{self.wandb_run_name}"
        if self.global_rank == 0:
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            if max_save is not None:
                step_dirs = [d for d in os.listdir(checkpoints_dir) 
                            if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith(prefix)]
                step_dirs.sort(key=lambda x: os.path.getctime(os.path.join(checkpoints_dir, x)))
                
                while len(step_dirs) >= max_save:
                    oldest_dir = os.path.join(checkpoints_dir, step_dirs[0])
                    shutil.rmtree(oldest_dir)
                    step_dirs.pop(0)
            
        save_dir = os.path.join(checkpoints_dir, f"{prefix}_{current_steps}")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving checkpoint for epoch {current_epoch} to {save_dir}")
            
        self.model_engine.save_checkpoint(save_dir=save_dir, tag=None, save_latest=False)
        
        if self.global_rank == 0:
            self.tokenizer.save_pretrained(save_dir)
            training_state = {
                'step': current_steps,
                'optimizer_state_dict': self.model_engine.optimizer.state_dict(),
                'loss': current_loss,
                'epoch': current_epoch,
            }
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
            

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
