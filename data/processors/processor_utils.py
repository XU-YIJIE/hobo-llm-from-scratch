from typing import TYPE_CHECKING, List, Sequence, Tuple 


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]: 
    r'''
    computes the real sequence length after truncation by. the cutoff_len
    '''
    if target_len * 2 < cutoff_len: # truncate source 
        max_target_len = cutoff_len 
    elif source_len * 2 < cutoff_len: # truncate target 
        max_target_len = cutoff_len - source_len 
    else: # truncate both 
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len))) 
    
    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len) 
    return new_source_len, new_target_len
