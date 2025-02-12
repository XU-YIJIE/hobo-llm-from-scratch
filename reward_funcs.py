import json
from typing import List
import re

# def reward_length(completions, **kwargs):
#     '''Reward function that gives higher scores to longer completions.'''
#     return [float(len(completion)) for completion in completions]

def reward_punish_too_long(completions, punish_length=20, **kwargs):
    '''
    Reward function that gives higher scores to completions that are close to 20 tokens.
    '''
    return [-abs(punish_length - len(completion.split(" ")))/100 for completion in completions]


def reward_hard_thinking_format(completions, **kwargs) -> list[float]:
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]


def reward_soft_thinking_format(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reward_think_mark(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [think_mark_num(response) for response in responses]


# 生成思考是否正确的奖励
def reward_unbias(completions, responses, **kwargs):
    import nltk
    # responses = [completion[0]['content'] for completion in completions]
    # extracted_thinks = [extract_think(r) for r in responses]
    # print(f"问题:\n{prompts[0][-1]['content']}", f"\n思考:\n{think[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的思考:\n{extracted_thinks[0]}")
    # print([nltk.translate.bleu_score.sentence_bleu(response_thk.split(' '), thk.split(' '), weights=(1,0,0,0)) for response_thk, thk in zip(extracted_thinks, think)])
    return [nltk.bleu(response.split(' '), completion.split(' '), weights=(1,0,0,0)) for completion, response in zip(completions, responses)]


def extract_think(text):
    if "<think>" not in text or "</think>" not in text:
        return ""
    think = text.split("<think>")[-1]
    think = think.split("</think>")[0]
    return think.strip()


def think_mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

# def reward_json_format(responses: List[str], **kwargs) -> List[float]:
#     rewards = []
#     for response in responses:
#         cleaned_text = response.strip()
        
#         if "```json" in cleaned_text:
#             parts = cleaned_text.split("```json", 1)
#             if parts[0].strip():
#                 rewards.append(-1.0)
#                 continue
                
#             if len(parts) != 2:
#                 rewards.append(-1.0)
#                 continue
                
#             json_part = parts[1].strip()
#             if not json_part.endswith("```"):
#                 rewards.append(-1.0)
#                 continue
                
#             json_content = json_part[:-3].strip()
            
#             try:
#                 parsed_json = json.loads(json_content)
#                 rewards.append(0.0)
#             except json.JSONDecodeError:
#                 rewards.append(-1.0)
#             continue
            
#         try:
#             parsed_json = json.loads(cleaned_text)
#             parsed_str = json.dumps(parsed_json)
#             if cleaned_text.strip() == parsed_str:
#                 rewards.append(1.0)
#             else:
#                 rewards.append(-1.0)
#         except json.JSONDecodeError:
#             rewards.append(-1.0)
#     return rewards
