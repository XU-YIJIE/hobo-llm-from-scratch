from functools import partial
import itertools
import math
import os
import argparse
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

        # 手动处理标签填充
        if labels is not None:
            if not self.padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_label_length = max(len(l) for l in labels)
                if self.max_length is not None:
                    max_label_length = min(max_label_length, self.max_length)
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

        # 确保input_ids不超过词表大小
        vocab_size = len(self.tokenizer)
        input_ids = batch["input_ids"].clamp_(0, vocab_size - 1)
        if batch["labels"] is not None:
            batch["labels"].clamp_(min=-100, max=vocab_size - 1)

        # 生成position_ids
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).expand_as(input_ids)
        
        return (input_ids, batch["attention_mask"], position_ids, batch["labels"]), None


def parse_args():
    parser = argparse.ArgumentParser()
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="lm_models/Qwen2.5-0.5B-Instruct", required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--dataset_dir", type=str, default="dataset/sharegpt_gpt4")
    parser.add_argument("--input_jsonl", type=str, default="sharegpt_gpt4.jsonl")
    parser.add_argument("--dataset_name", type=str, default="sharegpt_gpt4")
    parser.add_argument("--template", type=str, default="qwen")
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    
    # 训练参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    # 并行参数
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--checkpoint_activations", action="store_true")
    parser.add_argument("--checkpoint_num_layers", type=int, default=1)
    
    # 混合精度
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    
    # DeepSpeed配置
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    return args

def create_ds_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps
            }
        },
        
        "fp16": {
            "enabled": args.fp16,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        "bf16": {
            "enabled": args.bf16
        },
        
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        
        "pipeline": {
            "enabled": True,
            "num_stages": args.pp_size,
            "pipe_chunk_size": 2,
            "num_micro_batches": 4,
            "activation_checkpoint_interval": args.checkpoint_num_layers if args.checkpoint_activations else 0,
            "pipe_schedule": "interleaved"
        },
        
        "data_efficiency": {
            "dataloader_type": "single",
            "num_workers": 2,
            "pin_memory": False
        }
    }
    
    return ds_config


def initialize_dataset_and_dataloader(args, tokenizer):
    dataset = load_dataset(path=args.dataset_dir, data_files=args.input_jsonl, split="train") 
    data_args = DataArguments(template=args.template, cutoff_len=args.cutoff_len, train_on_prompt=False, mask_history=False, preprocessing_num_workers=args.preprocessing_num_workers)
    
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    data_args = DataArguments(
        template=args.template,
        cutoff_len=args.cutoff_len,
        train_on_prompt=False,
        mask_history=False,
        preprocessing_num_workers=min(args.preprocessing_num_workers, 4)  # 限制工作进程数
    )
    
    dataset = align_dataset(
        dataset,
        dataset_attr=DatasetAttr(load_from="file", dataset_name=args.dataset_name),
        data_args=data_args
    )
    
    sample_dataset = list(itertools.islice(dataset, 1))
    
    column_names_to_remove = list(next(iter(dataset)).keys())
    
    preprocess_func = partial(preprocess_supervised_dataset, template=template, tokenizer=tokenizer, data_args=data_args)
    dataset = dataset.map(preprocess_func, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=column_names_to_remove, desc="preprocessing dataset")
    
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=args.pp_size,
        rank=args.local_rank,
        shuffle=True,
        drop_last=True
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    
    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        pin_memory=False,
        num_workers=2,
        prefetch_factor=2
    )
    
    total_training_steps = math.ceil(len(train_dataloader) * args.num_epochs)

    return train_dataloader, total_training_steps
    # return train_dataloader


def train(args):
    deepspeed.init_distributed()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    model = get_deepspeed_pipemodule(args, args.pp_size)
    
    ds_config = create_ds_config(args)
    
    train_dataloader, total_training_steps = initialize_dataset_and_dataloader(args, tokenizer)
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
        # training_data=train_dataloader
    )
    
    for epoch in range(args.num_epochs):
        model_engine.train()
        
        # for step, batch in enumerate(train_dataloader)
        train_dataloader = deepspeed.utils.RepeatingLoader(train_dataloader)
        train_iterator = iter(train_dataloader)
        
        for iteration in range(total_training_steps):
            loss = model_engine.train_batch(data_iter=train_iterator)
            
            if dist.get_rank() == 0 and iteration % 100 == 0:
                print(f"Epoch: {epoch}, Step: {iteration}, Loss: {loss.item():.4f}")
        
        if dist.get_rank() == 0:
            model_engine.save_checkpoint(
                save_dir=os.path.join(args.output_dir, f"epoch_{epoch}")
            )


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main() 
