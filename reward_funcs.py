import json
from typing import List

def compute_json_format_reward(responses: List[str]) -> List[float]:
    '''
    计算基于JSON格式的奖励
    '''
    rewards = []
    for response in responses:
        cleaned_text = response.strip()
        
        if "```json" in cleaned_text:
            # 检查是否只包含```json {...} ```格式
            parts = cleaned_text.split("```json", 1)
            if parts[0].strip():  # ```json前有文本
                rewards.append(-1.0)
                continue
                
            if len(parts) != 2:  # 不完整的```json
                rewards.append(-1.0)
                continue
                
            json_part = parts[1].strip()
            if not json_part.endswith("```"):  # 没有结尾的```
                rewards.append(-1.0)
                continue
                
            json_content = json_part[:-3].strip()  # 移除结尾的```
            
            try:
                parsed_json = json.loads(json_content)
                rewards.append(0.0)
            except json.JSONDecodeError:
                rewards.append(-1.0)
            continue
            
        try:
            parsed_json = json.loads(cleaned_text)
            # 确保没有额外文本
            parsed_str = json.dumps(parsed_json)
            if cleaned_text.strip() == parsed_str:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except json.JSONDecodeError:
            rewards.append(-1.0)
            
    return rewards


def is_valid_json(text: str) -> bool:
    '''
    检查文本是否为有效的JSON格式
    
    Args:
        text (str): 需要检查的文本
        
    Returns:
        bool: 如果是有效的JSON格式返回True,否则返回False
    '''
    try:
        # 尝试解析JSON
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
    

def get_json_depth(obj, depth=1):
    '''
    计算JSON对象的最大深度
    '''
    if isinstance(obj, dict):
        if not obj:
            return depth
        return max(get_json_depth(v, depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return depth
        return max(get_json_depth(v, depth + 1) for v in obj)
    return depth

def count_json_fields(obj):
    '''
    计算JSON对象中的字段总数
    '''
    count = 0
    if isinstance(obj, dict):
        count += len(obj)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                count += count_json_fields(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                count += count_json_fields(v)
    return count


def compute_detailed_json_reward(text: str) -> float:
    '''
    基于JSON的复杂度计算更细粒度的奖励
    '''
    if not is_valid_json(text):
        return 0.0
    
    try:
        json_obj = json.loads(text)
        depth = get_json_depth(json_obj)
        num_fields = count_json_fields(json_obj)
        
        # 根据复杂度给出额外奖励
        bonus = min(0.5, (depth * 0.1 + num_fields * 0.05))
        return 1.0 + bonus
    except:
        return 0.0