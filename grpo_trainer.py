import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from accelerate import Accelerator
import warnings
import time
import numpy as np
from tqdm import tqdm
import json


def expand_indice(input_tensor: torch.Tensor, group_num: int) -> torch.Tensor:
    result = []
    for start in input_tensor:
        sequence = torch.arange(start * group_num, start * group_num + group_num)
        result.append(sequence)
    
    return torch.cat(result)


@dataclass
class GRPOConfig:
    learning_rate: float = 1e-5
    batch_size: int = 8
    group_num: int = 4  # 每个输入的采样数量
    mini_batch_size: int = 1  # minibatch更新的batch_size
    micro_batch_size: int = 1  # logprobs计算时的batch_size，用于降低显存峰值占用
    gradient_accumulation_steps: int = 1
    
    max_grad_norm: float = 1.0
    max_grad_norm: float = 1.0
    
    # KL惩罚相关
    init_kl_coef: float = 0.2
    beta: float = 0.01
    
    epochs: int = 4
    cliprange: float = 0.2
    
    seed: int = 42

class GRPOTrainer:
    def __init__(
        self,
        config: GRPOConfig,
        model,
        ref_model,
        tokenizer: PreTrainedTokenizer,
        optimizer = None,
        data_collator = None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        if data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        else:
            self.data_collator = data_collator
            
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate
            )
        else:
            self.optimizer = optimizer
            
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare(self.ref_model)
            
        self.current_device = self.accelerator.device
        # self.json_rewards_log = []

    def get_per_token_logps(self, model, input_ids, num_logits_to_keep):
        # We add 1 to `num_logits_to_keep` because the last logits of the sequence is later excluded
        # num_logits_to_keep 仅作用在lm_head上，控制logits长度
        logits = model(input_ids=input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def get_all_logprobs(
        self,
        model,
        query_ids: torch.Tensor,  # 输入padding后的
        response_ids: torch.Tensor,  # 输入padding后的
    ):
        batch_size = query_ids.shape[0] // self.config.group_num
        mini_batch_size = self.config.mini_batch_size
        all_logprobs = []
        
        for i in range(0, batch_size, mini_batch_size):
            mini_query_ids = query_ids[i * self.config.group_num: (i + mini_batch_size) * self.config.group_num]
            mini_response_ids = response_ids[i * self.config.group_num: (i + mini_batch_size) * self.config.group_num]
            input_ids = torch.cat([mini_query_ids, mini_response_ids], dim=1)
            logprobs = self.get_per_token_logps(model, input_ids, num_logits_to_keep=response_ids.shape[1])  # (mini_batch_size * group_num, seq_len)
            all_logprobs.append(logprobs)
            
            del input_ids, logprobs
            torch.cuda.empty_cache()
            
        return torch.cat(all_logprobs)

    def grpo_advantage(self, rewards: torch.FloatTensor, epsilon: float = 1e-4):
        rewards = rewards.to(torch.float32)
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True)
        
        advantages = (rewards - mean) / (std + epsilon)
        return advantages

    def step(
        self,
        query_ids: torch.LongTensor,
        response_ids: torch.LongTensor,
        scores: List[torch.FloatTensor],
    ):
        """
        queries: group_num * batch_size, query_len
        responses: group_num * batch_size, response_len
        """
        self.model.train()
        scores = torch.tensor(scores, device=self.current_device)
        batch_size = query_ids.shape[0] // self.config.group_num
        
        responses_mask = response_ids != self.tokenizer.pad_token_id
        
        with torch.no_grad():
            old_logprobs = self.get_all_logprobs(
                self.model,
                query_ids,
                response_ids,
            )  # (group_num * batch_size, response_len)
            
            ref_logprobs = self.get_all_logprobs(
                self.ref_model,
                query_ids,
                response_ids,
            )  # (group_num * batch_size, response_len)
        
        advantages = self.grpo_advantage(scores)  # (group_num * batch_size)
        stats = []
        
        b_inds = torch.arange(batch_size)
        
        for mini_batch_start in range(0, batch_size, self.config.mini_batch_size):
            mini_batch_end = mini_batch_start + self.config.mini_batch_size
            mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
            
            mini_batch_inds = expand_indice(mini_batch_inds, self.config.group_num)
            
            mini_batch = {
                "old_logprobs": old_logprobs[mini_batch_inds],  # group_num * mini_batch_size, response_len
                "responses_mask": responses_mask[mini_batch_inds],
                "advantages": advantages[mini_batch_inds],
                "query_ids": query_ids[mini_batch_inds],
                "response_ids": response_ids[mini_batch_inds]
            }
            
            with self.accelerator.accumulate(self.model):
                new_logprobs = self.get_all_logprobs(
                    self.model,
                    mini_batch["query_ids"],
                    mini_batch["response_ids"],
                )
                
                old_log_probs = mini_batch["old_logprobs"]
                
                ratio = torch.exp(new_logprobs - old_log_probs)
                ratio_clip = torch.clamp(ratio, 1 - self.config.cliprange, 1 + self.config.cliprange)
                
                pg_loss1 = -mini_batch["advantages"].unsqueeze(dim=1) * ratio
                pg_loss2 = -mini_batch["advantages"].unsqueeze(dim=1) * ratio_clip
                pg_loss = torch.max(pg_loss1, pg_loss2)
                
                kl = torch.exp(ref_logprobs[mini_batch_inds] - new_logprobs) - \
                    (ref_logprobs[mini_batch_inds] - new_logprobs) - 1
                
                loss = pg_loss + self.config.beta * kl
                loss = ((pg_loss * mini_batch["responses_mask"]).sum(dim=1) / mini_batch["responses_mask"].sum(dim=1)).mean()
                
                # print(loss.detach().float().cpu())
                
                self.accelerator.backward(loss)
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return stats

    def generate(
        self,
        query_tensor: Union[torch.Tensor],
        **generation_kwargs
    ):
        self.model.eval()
        return self.model.generate(
            input_ids=query_tensor,
            **generation_kwargs
        )