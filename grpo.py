import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from grpo_trainer import GRPOTrainer, GRPOConfig
from reward_funcs import compute_json_format_reward
import os
from loguru import logger

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
    config = GRPOConfig(
        learning_rate=1e-5,
        group_num=4,  # 每个输入生成4个候选回复
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        seed=42
    )
    
    dataloader = prepare_sample_dataset(batch_size=4)

    model_name = "lm_models/Qwen2.5-0.5B-Instruct"  # 使用Qwen2.5-0.5B-Instruct作为基础模型
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cpu()
    model.gradient_checkpointing_enable()
    
    # 注意padding区分left/right
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cpu()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    trainer = GRPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    num_epochs = 100
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

            input_ids = input_ids.to("cuda")
            input_len = input_ids.shape[1]
            
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
                "num_return_sequences": config.group_num,
            }
            
            gen_count = 0
            while True:
                responses = trainer.generate(
                    input_ids,
                    **gen_config
                )
                torch.cuda.empty_cache()

                response_ids = responses[:, input_len:]
                response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                scores = compute_json_format_reward(response_texts)
                
                if len(set(scores)) > 1:
                    break
                
                # scores如果输出单一值则没有训练意义
                logger.info(f"generation {gen_count} times, no valid JSON, continue")
                gen_count += 1
                
            print(f"Batch {batch_idx}, avg_score: {sum(scores)/len(scores):.3f}")

            # 扩展input_ids，对其responses
            input_ids = torch.repeat_interleave(input_ids, repeats=config.group_num, dim=0)
            stats = trainer.step(
                query_ids=input_ids,
                response_ids=response_ids,
                scores=scores
            )
            
        # model.save_pretrained(f"json_model_epoch_{epoch+1}")
        

if __name__ == "__main__":
    main()