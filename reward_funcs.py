import json
from typing import List
import re

# def reward_length(completions, **kwargs):
#     '''Reward function that gives higher scores to longer completions.'''
#     return [float(len(completion)) for completion in completions]

def reward_punish_too_long(completions, **kwargs):
    '''
    Reward function that gives higher scores to completions that are close to 20 tokens.
    '''
    return [-abs(20 - len(completion)) for completion in completions]


def reward_json_format(responses: List[str]) -> List[float]:
    rewards = []
    for response in responses:
        cleaned_text = response.strip()
        
        if "```json" in cleaned_text:
            parts = cleaned_text.split("```json", 1)
            if parts[0].strip():
                rewards.append(-1.0)
                continue
                
            if len(parts) != 2:
                rewards.append(-1.0)
                continue
                
            json_part = parts[1].strip()
            if not json_part.endswith("```"):
                rewards.append(-1.0)
                continue
                
            json_content = json_part[:-3].strip()
            
            try:
                parsed_json = json.loads(json_content)
                rewards.append(0.0)
            except json.JSONDecodeError:
                rewards.append(-1.0)
            continue
            
        try:
            parsed_json = json.loads(cleaned_text)
            parsed_str = json.dumps(parsed_json)
            if cleaned_text.strip() == parsed_str:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except json.JSONDecodeError:
            rewards.append(-1.0)
    return rewards


# def reward_strict_thinking_format(completions, **kwargs) -> list[float]:
#     pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses] 
#     return [0.5 if match else 0.0 for match in matches]


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]