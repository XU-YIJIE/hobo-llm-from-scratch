from datasets import load_dataset
from loguru import logger
from transformers import (PreTrainedTokenizer, 
                          AutoTokenizer, 
                          DataCollatorForLanguageModeling,
                          AutoModelForCausalLM, 
                          set_seed,
                          get_linear_schedule_with_warmup,
                          get_scheduler,
                          BitsAndBytesConfig)
from transformers.integrations import is_deepspeed_zero3_enabled
from typing import List, Dict, Any
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
import argparse
from accelerate import Accelerator
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

from data.preprocess import preprocess_pretrain_dataset
from data.aligner import align_dataset
from data.data_args import DataArguments
from data.parser import DatasetAttr
from constants import IGNORE_INDEX
from configuration_model import MyConfig
from modeling_hobo import HoboGPTModelForCausalLM
from arguments import parse_args

logger = logger.bind(name="pretrain")
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    filter=lambda record: record["level"].name in ["INFO", "ERROR"]
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"  # Using RTX 4000 series doesn't support faster communication broadband via P2P or IB

args = parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(self, args: argparse.Namespace):
        # data
        self.input_jsonl = args.input_jsonl
        self.dataset_name = args.dataset_name
        self.dataset_dir = args.dataset_dir
        
        # model
        self.model_name_or_path = args.model_name_or_path
        self.tokenizer_name_or_path = args.tokenizer_name_or_path
        self.use_8bit = args.use_8bit
        self.use_4bit = args.use_4bit
        self.torch_dtype = args.torch_dtype
        self.from_scratch = args.from_scratch
        self.device = args.device
        
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_grad_norm = args.max_grad_norm
        
        self.weight_decay = args.weight_decay
        self.warmup_steps = args.warmup_steps
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.adam_epsilon = args.adam_epsilon
        
        self.log_steps = args.log_steps
        self.save_steps = args.save_steps
        self.eval_steps = args.eval_steps
        self.max_save = args.max_save
        seed = args.seed
        set_seed(seed)
        
        self.data_args = DataArguments(cutoff_len=args.cutoff_len, 
                                       preprocessing_num_workers=8, 
                                       packing=True)
        
        self.accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        self.initializer()
        
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name or f"{args.wandb_project.split('/')[-1]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.wandb_dir = args.wandb_dir or f"./wandb/{self.wandb_run_name}"
        
        if self.accelerator.is_main_process:
            if not os.path.exists(self.wandb_dir):
                os.makedirs(self.wandb_dir)
            
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
                    "use_8bit": self.use_8bit,
                    "use_4bit": self.use_4bit,
                    "torch_dtype": self.torch_dtype,
                    "weight_decay": self.weight_decay,
                    "warmup_steps": self.warmup_steps,
                    "seed": seed,
                    "wandb_dir": self.wandb_dir,
                }
            )
            
            logger.info(f"Wandb local dir: {self.wandb_dir}")

    def initializer(self):
        torch.cuda.empty_cache()
        
        # bnb
        quantization_config = None
        load_in_4bit = self.use_4bit
        load_in_8bit = self.use_8bit
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if self.qlora:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",  # qlora使用nf4配合双精度
                        bnb_4bit_compute_dtype=self.torch_dtype,
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.torch_dtype,
                    )
            
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
                torch_dtype=self.torch_dtype,
            )
            if quantization_config:
                config.quantization_config = quantization_config
            self.model = HoboGPTModelForCausalLM(config=config)
        else:
            kwargs = {
                "pretrained_model_name_or_path": self.model_name_or_path,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
            if quantization_config:
                kwargs["quantization_config"] = quantization_config
            self.model = AutoModelForCausalLM.from_pretrained(**kwargs)

        if self.accelerator.is_main_process:
            logger.info(f"total trainable parameters: {count_parameters(self.model)/1e9:.5f}B")
        
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=True,
            model_max_length=self.data_args.cutoff_len * 2,
            padding_side="left"
        )
        
        self.create_dataloader()
        self.create_optimizer()

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.use_8bit or self.use_4bit:
            if self.use_4bit:
                from torchao.prototype.low_bit_optim import AdamW4bit
                self.optimizer = AdamW4bit(
                    optimizer_grouped_parameters,
                    lr=self.learning_rate,
                    betas=(self.adam_beta1, self.adam_beta2),
                    eps=self.adam_epsilon
                )
                logger.info("using 4-bit AdamW optimizer")
            else:
                from bitsandbytes.optim import AdamW8bit
                self.optimizer = AdamW8bit(
                    optimizer_grouped_parameters,
                    lr=self.learning_rate,
                    betas=(self.adam_beta1, self.adam_beta2),
                    eps=self.adam_epsilon
                )
                logger.info("using 4-bit AdamW optimizer")
        else:
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon
            )
            logger.info("using standard AdamW optimizer")
        
        num_update_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        max_train_steps = self.num_epochs * num_update_steps_per_epoch
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=max_train_steps,
        )


    def create_dataloader(self):
        full_dataset = load_dataset(path=self.dataset_dir, data_files=self.input_jsonl, split="train")
        # full_dataset = full_dataset.select(range(100))
        full_dataset = align_dataset(
            full_dataset, 
            dataset_attr=DatasetAttr(load_from="file", dataset_name=self.dataset_name), 
            data_args=self.data_args
        )
        
        column_names_to_remove = list(next(iter(full_dataset)).keys())
        
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=self.tokenizer,
            data_args=self.data_args,
        )
        
        with self.accelerator.main_process_first():
            full_dataset = full_dataset.map(
                preprocess_func,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names_to_remove,
                desc="preprocess dataset"
            )
            self.accelerator.wait_for_everyone()
            
        if self.accelerator.is_main_process:
            logger.info(f"Preprocessed dataset length: {len(full_dataset)}")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.train_dataloader = DataLoader(
            full_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.batch_size,
            pin_memory=False,
            num_workers=8,
            drop_last=True
        )
        
        counter_dataloader = DataLoader(
            full_dataset,
            shuffle=False,
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
            logger.info(f"total training tokens: {train_token_counter} ({train_token_counter/1e9:.2f}B)")
        
        del counter_dataloader
        gc.collect()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        with self.accelerator.accumulate(self.model):
            outputs = self.model(**batch)
            loss = outputs[0]
            
            del outputs
            
            self.accelerator.backward(loss)
            
            if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            return loss.detach().float().cpu()

    def train(self):
        total_steps = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            # 创建进度条
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                disable=not self.accelerator.is_local_main_process
            )
            
            self.model.train()
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                total_loss += loss
                total_steps += 1
                
                current_loss = total_loss / (step + 1)
                current_ppl = torch.exp(current_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                progress_bar.set_postfix({
                    'step': f"{step + 1}",
                    'loss': f"{current_loss:.4f}",
                    'ppl': f"{current_ppl:.2f}",
                    'lr': f"{current_lr:.2e}"
                })
                
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
                    
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        metrics["train/grad_norm"] = grad_norm
                    
                    if current_loss < best_loss and total_steps % self.save_steps == 0:
                        best_loss = current_loss
                        metrics["best_loss"] = best_loss
                        self.save_manager(epoch + 1, total_steps, current_loss, max_save=self.max_save, prefix="step")
                    
                    wandb.log(metrics, step=total_steps)

                if self.accelerator.is_main_process and total_steps % self.eval_steps == 0:
                    self.evaluate(
                        checkpoint_dir=f"checkpoints/{self.wandb_run_name}",
                        step_num=total_steps,
                        data_args=self.data_args,
                        from_scratch=self.from_scratch
                    )
                
                if step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            self.accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            gc.collect()

            if self.accelerator.is_main_process:
                avg_train_epoch_loss = total_loss / len(self.train_dataloader)
                train_ppl = torch.exp(avg_train_epoch_loss)

                epoch_metrics = {
                    "train/epoch_loss": avg_train_epoch_loss,
                    "train/epoch_perplexity": train_ppl,
                    "train/epoch": epoch + 1,
                }
                
                self.save_manager(epoch + 1, total_steps, avg_train_epoch_loss, max_save=self.max_save, prefix="epoch")
                    
                wandb.log(epoch_metrics, step=total_steps)
                
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}:")
                logger.info(f"  Average Loss: {avg_train_epoch_loss:.4f}")
                logger.info(f"  Perplexity: {train_ppl:.2f}")
                logger.info(f"  Learning Rate: {current_lr:.2e}")
            
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
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
            logger.info("\n" + "="*50 + " start evaluation " + "="*50)
            logger.info(f"loading best model from {best_model_dir}")
            if from_scratch:
                eval_model = HoboGPTModelForCausalLM.from_pretrained(
                    best_model_dir,
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
            
            test_prompts = [
                "今天天气真好,",
                "人工智能是",
                "春天来了,",
                "我最喜欢的季节是",
                "学习编程的第一步是",
                "深度学习主要研究",
                "地球是一颗",
                "音乐能够",
                "读书使我",
                "科技发展",
                "环境保护需要",
                "互联网给我们带来了",
                "运动对健康的好处是",
                "中国传统文化的精髓在于",
                "未来的教育应该",
                "太空探索的意义是",
                "大数据时代的特点是",
                "创新思维能够",
                "城市发展应该注重",
                "良好的生活习惯包括"
            ]
            
            # model_max_length = eval_model.config.max_position_embeddings
            # input_ids = eval_tokenizer(test_prompts, return_tensors="pt", padding=True)["input_ids"]
            # max_prompt_length = input_ids.shape[1]
            # max_new_tokens = min(512, model_max_length - max_prompt_length)  # 确保总长度不超过模型限制
            
            gen_config = {
                "max_new_tokens": 1024,
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
            
            input_ids = eval_tokenizer(test_prompts, return_tensors="pt", padding=True)["input_ids"]
            outputs = eval_model.generate(
                input_ids,
                **gen_config
            )
            responses = eval_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            data = [[step_num, prompt, response] for prompt, response in zip(test_prompts, responses)]
            
            if not hasattr(self, 'eval_history'):
                self.eval_history = []
            
            self.eval_history.extend(data)
            
            eval_table = wandb.Table(
                columns=["Step", "Prompt", "Response"],
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

if __name__ == "__main__":
    args = parse_args()
    args.from_scratch = True
    # args.eval_steps = 100
    # args.save_steps = 100
    # args.batch_size = 10
    
    # args.use_4bit = True
    # args.use_8bit = True
    args.wandb_project = "pt_training"
    trainer = Trainer(args)
    trainer.train()