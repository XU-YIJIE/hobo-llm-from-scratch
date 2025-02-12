import torch
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)
from grpo_trainer import GRPOTrainer, GRPOConfig
from reward_funcs import reward_json_format, reward_punish_too_long
import os
from loguru import logger
from accelerate import Accelerator
from transformers.data.data_collator import DataCollatorForSeq2Seq
import numpy as np
from torch.utils.data import DataLoader
from functools import partial

from datasets import load_dataset
from data.aligner import convert_tldr
from data.preprocess import preprocess_rl_dataset_v1
from constants import IGNORE_INDEX

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def main():
    learning_rate = 1e-5
    group_num = 8
    mini_batch_size = 1
    gradient_accumulation_steps = 8
    max_grad_norm = 1
    seed = 1024
    batch_size = 4
    num_epochs = 100
    model_name = "lm_models/Qwen2.5-0.5B-Instruct"  # 使用Qwen2.5-0.5B-Instruct作为基础模型
    dataset_dir = "dataset/tldr"
    
    # config
    config = GRPOConfig(
        learning_rate=learning_rate,
        group_num=group_num,  # 每个输入生成4个候选回复
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        seed=seed
    )
    
    # accelerator init
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    # model
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    # 注意padding区分left/right
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cpu()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # dataprocess
    full_dataset = load_dataset(path=dataset_dir, split="train")
    # full_dataset = full_dataset.select(range(100))
    full_dataset = convert_tldr(full_dataset)
    column_names = list(next(iter(full_dataset)).keys())
    preprocess_func = partial(
        preprocess_rl_dataset_v1,
        tokenizer=tokenizer,
    )
    with accelerator.main_process_first():
        full_dataset = full_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            desc="Preprocessing dataset"
        )
    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX)
    dataloader = DataLoader(full_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    num_training_steps = len(dataloader) * (batch_size // mini_batch_size) // gradient_accumulation_steps
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    model, optimizer, dataloader, lr_scheduler, ref_model = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler, ref_model)
    
    trainer = GRPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    
    # reward_funcs = [reward_json_format, reward_length]
    reward_funcs = [partial(reward_punish_too_long, punish_length=100)]
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            input_len = input_ids.shape[1]
            
            # 采样参数
            gen_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,
                "num_return_sequences": group_num,
            }
            gen_count = 0
            all_rewards = []
            while True:
                if accelerator.is_main_process:
                    logger.info(f"starting generation {gen_count} times")
                responses = trainer.generate(
                    input_ids,
                    **gen_config
                )
                torch.cuda.empty_cache()

                response_ids = responses[:, input_len:]
                response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)  # group_num * batch_size
                
                print(response_texts)
                
                for reward_func in reward_funcs:
                    rewards = np.array(reward_func(response_texts))  # length: group_num * batch_size
                    all_rewards.append(rewards)  # length: func_num
                all_rewards = np.array(all_rewards)  # func_num, group_num * batch_size
                all_rewards = all_rewards.sum(axis=0)  # group_num * batch_size
                
                if len(set(all_rewards)) > 1:
                    break
                else:
                    # scores如果输出单一值则没有训练意义
                    logger.info(f"invalid generation with count {gen_count}, continue")
                    gen_count += 1
                    all_rewards = []

            if accelerator.is_main_process:
                logger.info(f"Batch {step + 1}, avg_reward_score: {sum(all_rewards)/len(all_rewards):.3f}")

            # 扩展input_ids，对齐responses
            input_ids = input_ids.unsqueeze(1).expand(-1, config.group_num, -1).reshape(-1, input_ids.size(-1))
            rewards = torch.tensor(all_rewards, device=accelerator.device)
            
            # rl step
            state = trainer.step(
                query_ids=input_ids,
                response_ids=response_ids,
                reward_scores=rewards
            )
            
        # model.save_pretrained(f"json_model_epoch_{epoch+1}")
        

if __name__ == "__main__":
    main()