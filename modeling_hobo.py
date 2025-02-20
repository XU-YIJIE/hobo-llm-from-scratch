from torch import nn
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, Qwen2Model, Qwen2Config, GenerationMixin
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.cache_utils import DynamicCache, Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.loss.loss_utils import fixed_cross_entropy
from loguru import logger

from constants import IGNORE_INDEX
from configuration_model import MyConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.
    q: (b, n_heads, seq_len, size_per_head)
    cos: (1, seq_len, emb_dim * 2)
    """
    cos = cos.unsqueeze(1)  # 1, 1, seq_len, emb_dim * 2
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        output = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output.to(input_dtype)  # 均方根归一化，公式：x / sqrt(x^2 + eps) * weight

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_position_embeddings: int = 2048, scaling_factor: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # 1, head_dim / 2
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
    
    def _compute_scaled_inv_freq(self, seq_len, device):
        # if seq_len > self.max_position_embeddings
        scaled_base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)) ** (self.head_dim / (self.head_dim - 2))
        scaled_inv_freq = 1.0 / (scaled_base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
        return scaled_inv_freq
            
    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq = self._compute_scaled_inv_freq(seq_len=seq_len, device=device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset inv_freq
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
    
    @torch.no_grad()
    def forward(self, hidden_state, position_ids, seq_len):
        '''
        position_ids: (1, seq_len)
        '''
        self._dynamic_frequency_update(position_ids, device=hidden_state.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)  # 广播到batch维度
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # 1, seq_len, head_dim / 2
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()  # 1, seq_len, head_dim
        sin = emb.sin()
        return cos.to(dtype=hidden_state.dtype), sin.to(dtype=hidden_state.dtype)

class Attention(nn.Module):
    def __init__(self, config:MyConfig, layer_idx:int):
        super().__init__()
        self.config = config
        self.is_causal = True
        self.flash_attn = config.flash_attn
        self.attention_dropout = config.attention_dropout
        self.n_heads = config.num_attention_heads
        self.size_per_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.n_kv_heads = config.num_key_value_heads  # 用于gqa
        self.n_kv_groups = config.num_attention_heads // config.num_key_value_heads  # 计算gqa的groups。mqa的情况下，n_kv_groups = 1
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads {self.n_heads} must be divisible by n_kv_heads {self.n_kv_heads}"
        # self.max_len = config.max_position_embeddings
        self.layer_idx = layer_idx  # 用于区分不同层，用于cache
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.size_per_head)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.size_per_head)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.rms_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.size_per_head,
            base=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=config.rope_scaling
        )
        
    def forward(self, hidden_states, attention_mask, position_ids, cache_position, past_key_values: Cache=None, use_cache=False):
        # TODO 根据llama定义的结构，实现attention。还需要再检查
        
        b, seq_len, hidden_size = hidden_states.size()
        
        q = self.q_proj(hidden_states).view(b, seq_len, self.n_heads, -1).transpose(1, 2)
        # (b, seq_len, hidden_size) -> (b, n_heads, seq_len, size_per_head)
        k = self.k_proj(hidden_states).view(b, seq_len, self.n_kv_heads, -1).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, self.n_kv_heads, -1).transpose(1, 2)
        
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        k = repeat_kv(k, self.n_kv_groups)  # gqa情况下，将kv的维度与q的维度对齐
        v = repeat_kv(v, self.n_kv_groups)
        
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        
        if not self.flash_attn:
            # eager sdpa
            qk_dot = torch.matmul(q, k.transpose(-2, -1))  
            # (b, n_heads, seq_len, size_per_head) * (b, n_heads, size_per_head, seq_len) -> (b, n_heads, seq_len, seq_len)
            qk_dot = qk_dot * (self.size_per_head ** -0.5)  # scaling factor
            qk_dot = qk_dot + attention_mask
            attn = F.softmax(qk_dot, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)
            attention_output = output.transpose(1, 2).contiguous().view(b, seq_len, -1)
        else:
            attention_output, attn_weights = flash_attention_forward(
                self, q, k, v,
                attention_mask,  # (b, seq_len)
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.size_per_head ** -0.5,
            )
        
        attention_output = self.out_proj(attention_output)
        return attention_output


class MLP(nn.Module):
    def __init__(self, config:MyConfig):
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)  # 上采样
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)  # 门控
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)  # 下采样
        self.rms_norm = RMSNorm(config.hidden_size)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))  # llama2的mlp结构


class GPTBlock(nn.Module):
    def __init__(self, config:MyConfig, layer_idx:int):
        super().__init__()
        self.config = config
        self.attention = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.gradient_checkpointing = False
        
    def forward(self, hidden_states, attention_mask, position_ids, cache_position, past_key_values=None, use_cache=False):
        residual = hidden_states + self.attention(self.norm1(hidden_states), attention_mask, position_ids, cache_position, past_key_values, use_cache)  # llama2的normalization位于输入之前
        output = residual + self.mlp(self.norm2(residual))
        return output, past_key_values


class GPTModel(PreTrainedModel):
    config_class = MyConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    
    def __init__(self, config:MyConfig):
        super().__init__(config)
        self.config = config
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_layers = nn.ModuleList([GPTBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.rms_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.gradient_checkpointing = False
        # causal_mask = torch.full(
        #     (config.max_position_embeddings, config.max_position_embeddings), 
        #     fill_value=torch.finfo(torch.float32).min, dtype=torch.float32)
        # causal_mask = torch.triu(causal_mask, diagonal=1)  # 生成上三角矩阵，diagonal=1表示主对角线上移一位
        # self.register_buffer("causal_mask", causal_mask, persistent=False)  # persistent=False表示不保存到state_dict
        self.post_init()
        
    def forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None, use_cache=False, cache_position=None):
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False
            
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        # cache_position 表示当前输入token在整个序列中的绝对位置，帮助模型维护正确的注意力模式和缓存更新
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )  # 启用kv_cache的情况下，通常 input_ids.shape[1] == 1
            
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        inputs_embeds = self.embedding_layer(input_ids)
        
        # 在attention外部先计算好causal_mask
        # 适配kv cache的_update_causal_mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values)
        # flash_attn == True: 
        #   2d causal_mask = attention_mask
        # flash_attn == False:
        #   4d causal_mask
        
        hidden_states = inputs_embeds
        
        for decoder_layer in self.decoder_layers:
            # 梯度检查点功能
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    cache_position,
                    past_key_values,
                    use_cache
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    use_cache=use_cache
                )
            hidden_states, past_key_values = layer_outputs
        
        hidden_states = self.rms_norm(hidden_states)
        
        # output = BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=past_key_values if use_cache else None,
        # )
        
        return hidden_states, past_key_values if use_cache else None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _update_causal_mask(self, attention_mask, inputs_embeds, cache_position, past_key_values: DynamicCache):
        if self.config.flash_attn:
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        
        batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        # 只针对dynamic_cache实现
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + seq_len + 1
        )
        
        if attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:  # (b, sequence_length) -> (b, 1, sequence_length, sequence_length)  mask = padding_mask & causal_mask
            ''' 可读性更佳的版本
            # padding mask
            padding_mask = mask[:, None, None, :].expand(b, 1, seq_len, seq_len).to(dtype=dtype)  
            # (b, 1, 1, seq_len) -> (b, 1, seq_len, seq_len)
            # (1, seq_len) -> (seq_len, seq_len)
            # 假设输入[[[1, 1, 0]], [[1, 1, 1]]]，经过变换后变成[[[1, 1, 0], [1, 1, 0], [1, 1, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
            
            # causal mask
            causal_mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
            mask_cond = torch.arange(causal_mask.size(-1), device=device)
            causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 1)
            causal_mask = causal_mask[None, None, :, :].expand(b, 1, seq_len, seq_len).to(dtype=dtype)
            
            # 将padding mask和causal mask进行逻辑与运算，得到最终的mask
            mask = padding_mask & causal_mask  # 两个mask都是(b, 1, seq_len, seq_len)维度
            qk_dot = qk_dot.masked_fill(mask == 0, torch.finfo(dtype).min)
            '''
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (seq_len, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device
            )  # kv cache的常规情况下，target_length = past_seen_tokens + seq_len，seq_len = 1
            
            if seq_len != 1:  # 通常指训练的情况
                causal_mask = torch.triu(causal_mask, diagonal=1)
            
            # 使用cache_position更新掩码
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            
            # 扩展到batch维度
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            
            # 处理padding mask（直接在causal_mask上修改，避免额外内存分配）
            if attention_mask is not None:  
                causal_mask = causal_mask.clone()  # 复制到连续内存以进行原地编辑
                target_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :target_length] + attention_mask[:, None, None, :]  # broadcasting
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :target_length] = causal_mask[:, :, :, :target_length].masked_fill(padding_mask, min_dtype)  # 最终的mask
            
        return causal_mask


class HoboGPTModelForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = MyConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    def __init__(self, config: MyConfig):
        super().__init__(config)
        self.config = config
        self.model = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.gradient_checkpointing = False
        self.post_init()
    
    def gradient_checkpointing_enable(self):
        """启用梯度检查点功能"""
        self.gradient_checkpointing = True
        self.model.gradient_checkpointing = True
        self.model.gradient_checkpointing_enable()
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTBlock):
            module.gradient_checkpointing = value
    
    def forward(self, input_ids, attention_mask, labels=None, inputs_embeds=None, position_ids=None, num_logits_to_keep=0, past_key_values=None, use_cache=False, cache_position=None, return_dict=False, **kwargs):
        outputs = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache, cache_position)
        
        hidden_states, past_key_values = outputs
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            labels = labels.to(logits.device)
            
            if self.training:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = fixed_cross_entropy(shift_logits, shift_labels)
        
        output = (logits, past_key_values)
        if return_dict:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
            )
        else:
            return (loss,) + output if loss is not None else output

if __name__ == "__main__":
    
    config = MyConfig()
    model = GPTModel(config)
    model.half().to("cuda")
    
    pass