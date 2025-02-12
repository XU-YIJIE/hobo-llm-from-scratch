from functools import partial 
from itertools import chain 
from typing import Any, Dict, List, Optional, Sequence, Tuple 
from loguru import logger

logger = logger.bind(name="preprocess")


from transformers import Seq2SeqTrainingArguments 
from transformers.tokenization_utils import PreTrainedTokenizer 
from .processors.processor_utils import infer_seqlen 
from .template import Template
from .data_args import DataArguments

IGNORE_INDEX = -100 


# referenced from llamafactory
def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[Any]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    eos_token = "<|end_of_text|>" if data_args.template == "llama3" else tokenizer.eos_token
    # text_examples = [messages[0]["content"] + eos_token  for messages in examples["_prompt"]]
    
    text_examples = []
    for hists, resps in zip(examples["_prompt"], examples["_response"]):
        for hist in hists:
            text_examples.append(hist["content"] + eos_token)
        if resps:
            text_examples.append(resps[0]["content"] + eos_token)
    
    if not data_args.packing:
        if getattr(tokenizer, "add_bos_token", False):
            text_examples = [tokenizer.bos_token + example for example in text_examples]

        result = tokenizer(text_examples, add_special_tokens=False, truncation=True, max_length=data_args.cutoff_len)
    else:
        tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if getattr(tokenizer, "add_bos_token", False):
            for i in range(len(result["input_ids"])):
                result["input_ids"][i][0] = tokenizer.bos_token_id

    return result


# referenced from llamafactory
def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: Template,
    tokenizer: PreTrainedTokenizer,
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:

    # build inputs with format <bos> X Y <eos> and labels with format `<ignore> ... <ignore> Y <eos>
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.

    model_inputs = {"input_ids": [], "labels": [], "attention_mask": []}
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:  # 提示词部分的一问一答是奇数
            logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
            continue
        
        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i], 
            response=examples["_response"][i], 
            system=examples["_system"][i], 
            tools=examples["_tools"][i], 
            template=template, 
            tokenizer=tokenizer, 
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
        mask_history=data_args.mask_history)
        model_inputs["input_ids"].append(input_ids)
        model_inputs["labels"].append(labels) 
        model_inputs["attention_mask"].append([1] * len(input_ids)) 
    return model_inputs


def preprocess_rl_dataset_v1(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[List[int]]]:
    model_inputs = {"input_ids": []}
    for i in range(len(examples["_prompt"])):
        system = examples["_system"][i]
        prompt = examples["_prompt"][i]
        # response = examples["_response"][i]
        input_str = tokenizer.apply_chat_template([system] + prompt, template=tokenizer.chat_template, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_str, return_tensors="pt", add_special_tokens=True, padding=True)["input_ids"]
        model_inputs["input_ids"].append(input_ids[0])
        # model_inputs["responses"].append(response)
    return model_inputs


# referenced from llamafactory 
def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    messages = prompt + response # 将多轮列数据回拼 
    input_ids, labels = [], [] 
    '''
    messages = [
        {"role": "user", "content": "How are you"}, 
        {"role": "assistant", "content": "I am fine!"}, 
        {"role": "user", "content": "你好"}, 
        {"role": "assistant", "content": "很高兴认识你!"}, 
    ]
    '''
    
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels

