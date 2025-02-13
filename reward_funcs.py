import json
from typing import List
import re

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

def reward_unbias(completions, responses, **kwargs):
    import nltk
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
