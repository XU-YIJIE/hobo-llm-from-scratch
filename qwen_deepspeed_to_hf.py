import re
import os
import torch
import json
import argparse
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers import AutoModelForCausalLM, AutoConfig
from collections import OrderedDict

# # hf模型
# ['model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.q_proj.bias', 'model.layers.0.self_attn.k_proj.weight', 'model.layers.0.self_attn.k_proj.bias', 'model.layers.0.self_attn.v_proj.weight', 'model.layers.0.self_attn.v_proj.bias', 'model.layers.0.self_attn.o_proj.weight', 'model.layers.0.mlp.gate_proj.weight', 'model.layers.0.mlp.up_proj.weight', 'model.layers.0.mlp.down_proj.weight', 'model.layers.0.input_layernorm.weight', 'model.layers.0.post_attention_layernorm.weight']
# ['model.norm.weight', 'lm_head.weight']

# # pp模型
# ['decoder_layer.self_attn.q_proj.weight', 'decoder_layer.self_attn.q_proj.bias', 'decoder_layer.self_attn.k_proj.weight', 'decoder_layer.self_attn.k_proj.bias', 'decoder_layer.self_attn.v_proj.weight', 'decoder_layer.self_attn.v_proj.bias', 'decoder_layer.self_attn.o_proj.weight', 'decoder_layer.mlp.gate_proj.weight', 'decoder_layer.mlp.up_proj.weight', 'decoder_layer.mlp.down_proj.weight', 'decoder_layer.input_layernorm.weight', 'decoder_layer.post_attention_layernorm.weight']
# ['norm.weight', 'lm_head.weight']

QWEN_HF_STATE_DICT_MAPPINGS = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "decoder_layer.self_attn.q_proj.weight": "model.layers.<LAYER>.self_attn.q_proj.weight",
    "decoder_layer.self_attn.q_proj.bias": "model.layers.<LAYER>.self_attn.q_proj.bias",
    "decoder_layer.self_attn.k_proj.weight": "model.layers.<LAYER>.self_attn.k_proj.weight", 
    "decoder_layer.self_attn.k_proj.bias": "model.layers.<LAYER>.self_attn.k_proj.bias",
    "decoder_layer.self_attn.v_proj.weight": "model.layers.<LAYER>.self_attn.v_proj.weight",
    "decoder_layer.self_attn.v_proj.bias": "model.layers.<LAYER>.self_attn.v_proj.bias",
    "decoder_layer.self_attn.o_proj.weight": "model.layers.<LAYER>.self_attn.o_proj.weight",
    "decoder_layer.mlp.gate_proj.weight": "model.layers.<LAYER>.mlp.gate_proj.weight",
    "decoder_layer.mlp.up_proj.weight": "model.layers.<LAYER>.mlp.up_proj.weight",
    "decoder_layer.mlp.down_proj.weight": "model.layers.<LAYER>.mlp.down_proj.weight",
    "decoder_layer.input_layernorm.weight": "model.layers.<LAYER>.input_layernorm.weight",
    "decoder_layer.post_attention_layernorm.weight": "model.layers.<LAYER>.post_attention_layernorm.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight"
}


def map_keys_from_pipe_to_hf(key, file): 
    """Convert Megatron-Deepspeed TP/PP weights mapping in transformers PP only""" 
    # Handle first and last layers 
    transformer_key = QWEN_HF_STATE_DICT_MAPPINGS[key]
    
    if transformer_key.find("<LAYER>") != -1: 
        # Handle transformer blocks 
        layer_number = int(re.match(r".*layer_(\d*).*", file)[1]) 
        layer_number -= 1  # 第一个file是embed_tokens
        transformer_key = transformer_key.replace("<LAYER>", str(layer_number)) 

    return transformer_key


def convert_qwen_checkpoint_to_pytorch(checkpoint_path, pytorch_dump_folder_path, hf_model_path):
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_config(config).to("cpu")
    file_names = os.listdir(checkpoint_path)
    file_names = sorted(filter(lambda s: s.startswith("layer") and "model_00" in s, file_names))
    
    # 创建一个空的状态字典用于合并
    merged_state_dict = OrderedDict()
    
    for _, f_name in enumerate(file_names):
        print("Processing file: {}".format(f_name), flush=True)
        pt_dict = torch.load(os.path.join(checkpoint_path, f_name), map_location="cpu")
        
        # Rename keys in the transformers names
        keys = list(pt_dict.keys())
        
        for key in keys:
            transformer_key = map_keys_from_pipe_to_hf(key, f_name)
            merged_state_dict[transformer_key] = pt_dict.pop(key)
            # print(f"### [{f_name}] set ds key {key} to transformer_key {transformer_key}, shape: {merged_state_dict[transformer_key].shape}", flush=True)
            
    hf_model.load_state_dict(merged_state_dict, strict=True)
    if not pytorch_dump_folder_path:
        pytorch_dump_folder_path = os.path.dirname(checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # Required parameters 
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the Megatron-LM checkpoint path.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--hf_model_path", default=None, type=str, help="Path to the Hugging Face model.")
    args = parser.parse_args() 
    
    args.checkpoint_path = "checkpoints/sft_ds_pipe_training_20250323_151609/step_10/global_step10"
    args.hf_model_path = "lm_models/Qwen2.5-0.5B-Instruct"
    
    convert_qwen_checkpoint_to_pytorch(
        args.checkpoint_path, 
        args.pytorch_dump_folder_path, 
        args.hf_model_path
    )