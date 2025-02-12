import torch
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)
from datasets import Dataset
from grpo_trainer import GRPOTrainer, GRPOConfig
from reward_funcs import reward_json_format, reward_punish_too_long
import os
from loguru import logger
from accelerate import Accelerator
from transformers.data.data_collator import DataCollatorForLanguageModeling
import numpy as np

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def prepare_sample_dataset(batch_size=4):
    """
    test dataset
    """
    data = {
        "query": [
            "将以下信息转换为JSON格式：姓名=张三,年龄=25,职业=工程师",
            "生成一个JSON，包含：书名=西游记,作者=吴承恩,类型=古典小说", 
            "创建一个用户信息的JSON：用户=李四,邮箱=li4@email.com,等级=VIP",
            "转换成JSON：商品=手机,价格=3999,品牌=小米,型号=13",
            "生成一个订单的JSON：订单号=A12345,金额=299.99,状态=已付款",
            "将学生信息转为JSON：学号=2024001,姓名=王五,班级=三年二班,成绩=95",
            "转换成JSON格式：天气=晴朗,温度=26,湿度=65%,风力=3级",
            "生成一个活动信息的JSON：活动名=新年晚会,时间=2024-01-01,地点=大礼堂,人数=500"
        ],
        "response": [
            '{"name": "张三", "age": 25, "occupation": "工程师"}',
            '{"title": "西游记", "author": "吴承恩", "genre": "古典小说"}',
            '{"user": "李四", "email": "li4@email.com", "level": "VIP"}',
            '{"product": "手机", "price": 3999, "brand": "小米", "model": "13"}',
            '{"order_id": "A12345", "amount": 299.99, "status": "已付款"}',
            '{"student_id": "2024001", "name": "王五", "class": "三年二班", "score": 95}',
            '{"weather": "晴朗", "temperature": 26, "humidity": "65%", "wind_level": 3}',
            '{"event_name": "新年晚会", "time": "2024-01-01", "location": "大礼堂", "capacity": 500}'
        ]
    }
    
    dataset = Dataset.from_dict(data)
    dataset.set_format(type="torch", columns=["query", "response"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False
    )
    
    return dataloader


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
    
    config = GRPOConfig(
        learning_rate=learning_rate,
        group_num=group_num,  # 每个输入生成4个候选回复
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        seed=seed
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    
    # 注意padding区分left/right
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cpu()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = prepare_sample_dataset(batch_size=batch_size)
    num_training_steps = len(dataloader) * (batch_size // mini_batch_size) // gradient_accumulation_steps
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
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
    reward_funcs = [reward_punish_too_long]
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            batch_query = batch["query"]
            input_strs = []
            for query in batch_query:
                MESSAGES = [
                    {"role": "user", "content": query}
                ]
                input_str = tokenizer.apply_chat_template(MESSAGES, template=tokenizer.chat_template, tokenize=False, add_generation_prompt=True)
                input_strs.append(input_str)
            input_ids = tokenizer(input_strs, return_tensors="pt", add_special_tokens=True, padding=True)["input_ids"]

            input_ids = input_ids.to(accelerator.device)
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
                response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                
                # print(response_texts)
                
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
                logger.info(f"Batch {batch_idx + 1}, avg_reward_score: {sum(all_rewards)/len(all_rewards):.3f}")

            # 扩展input_ids，对齐responses
            input_ids = input_ids.unsqueeze(1).expand(-1, config.group_num, -1).reshape(-1, input_ids.size(-1))
            rewards = torch.tensor(all_rewards, device=accelerator.device)
            
            state = trainer.step(
                query_ids=input_ids,
                response_ids=response_ids,
                reward_scores=rewards
            )
            
        # model.save_pretrained(f"json_model_epoch_{epoch+1}")
        

if __name__ == "__main__":
    main()
    
    # response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    # all_rewards = []
    # for reward_func in reward_funcs:
    #     rewards = torch.tensor(reward_func(response_texts), device=self.current_device).unsqueeze(dim=0)  # length: group_num * batch_size
    #     all_rewards.append(rewards)  # length: func_num
    # all_rewards = torch.cat(all_rewards, dim=0).sum(dim=0)
    # advantages = self.grpo_advantage(all_rewards)  # (group_num * batch_size)
    # if set(advantages.unique()) == 1:
    #     if self.accelerator.is_main_process:
    #         logger.warning(f"all rewards are the same, skip this step")
    #     del all_rewards, advantages
    #     return stats, "skip"