import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from transformers import Qwen2ForCausalLM, AutoConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import os
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from torch import einsum, nn
import importlib.util
from functools import partial

torch.utils.checkpoint.checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)


def get_deepspeed_pipemodule(args, pp_size):
    if not dist.is_initialized():
        deepspeed.init_distributed()
    
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    model_config.max_seq_len = args.max_seq_len
    
    model = QwenPipelineModel.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    world_size = dist.get_world_size()
    dp_size = world_size // pp_size
    topology = PipeModelDataParallelTopology(
        num_pp=pp_size,
        num_mp=1,
        num_dp=dp_size
    )
    pipe_model = PipelineModule(
        layers=model.to_pipe_layers(),
        loss_fn=loss_fn,
        topology=topology,
        activation_checkpoint_interval=args.checkpoint_num_layers if args.checkpoint_activations else 0,
        partition_method='uniform'
    )
    
    print(f"[rank_{dist.get_rank()}]: Pipeline model initialization completed.")
    return pipe_model


class QwenPipelineModel(Qwen2ForCausalLM):
    def to_pipe_layers(self):
        self.train()
        layers = []
        layers.append(LayerSpec(EmbeddingPipeLayer, self.model.embed_tokens, self.config))
        
        for decoder_layer in self.model.layers:
            layers.append(LayerSpec(TransformerPipeLayer, decoder_layer))
        
        layers.append(LayerSpec(NormPipeLayer, self.model.norm))
        layers.append(LayerSpec(LMHeadPipeLayer, self.lm_head))
        
        return layers


class EmbeddingPipeLayer(nn.Module):
    def __init__(self, embed_tokens, config):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.config = config
    
    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels = inputs
        
        device = self.embed_tokens.weight.device
        vocab_size = self.embed_tokens.weight.size(0)
        
        input_ids = input_ids.to(device=device, dtype=torch.long).clamp_(0, vocab_size - 1)
        attention_mask = attention_mask.to(device=device)
        position_ids = position_ids.to(device=device, dtype=torch.long)
        if labels is not None:
            labels = labels.to(device=device, dtype=torch.long).clamp_(min=-100, max=vocab_size - 1)
        
        hidden_states = self.embed_tokens(input_ids)
        
        batch_size, seq_length = input_ids.shape
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length=0,
            sliding_window=getattr(self.config, 'sliding_window', None)
        )
        
        # dtype = hidden_states.dtype
        attention_mask = attention_mask.to(dtype=torch.long, device=device)
        
        return hidden_states, attention_mask, position_ids, labels


class TransformerPipeLayer(nn.Module):
    
    def __init__(self, decoder_layer):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.rotary_emb = RotaryEmbedding(dim=self.decoder_layer.self_attn.head_dim, theta=10000.0)
    
    def forward(self, inputs):
        hidden_states, attention_mask, position_ids, labels = inputs
        
        # if not hidden_states.requires_grad:
        #     hidden_states.requires_grad_(True)
        
        seq_length = hidden_states.shape[1]
        position_embeddings = self.rotary_emb(seq_length)
        position_embeddings = [x.to(dtype=hidden_states.dtype, device=hidden_states.device) for x in position_embeddings]
        
        if self.training:
            def custom_forward(hidden_states, attention_mask, position_ids, position_embeddings):
                outputs = self.decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False
                )
                output = outputs[0]
                # if not output.requires_grad:
                #     output.requires_grad_(True)
                return output
            
            hidden_states = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_ids,
                position_embeddings,
                use_reentrant=False
            )
        else:
            outputs = self.decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False
            )
            hidden_states = outputs[0]
        # if not hidden_states.requires_grad:
        #     hidden_states.requires_grad_(True)
        # print(f'RANK={dist.get_rank()} +++++++++++++++++++++++++++')
        # print("hidden_states", hidden_states, hidden_states.shape)
        # print(f'RANK={dist.get_rank()} +++++++++++++++++++++++++++')
        return hidden_states, attention_mask, position_ids, labels


class NormPipeLayer(nn.Module):
    def __init__(self, norm_layer):
        super().__init__()
        self.norm = norm_layer
    
    def forward(self, inputs):
        hidden_states, attention_mask, position_ids, labels = inputs
        hidden_states = self.norm(hidden_states)
        return hidden_states, attention_mask, position_ids, labels


class LMHeadPipeLayer(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head
        self.loss_fct = CrossEntropyLoss()
    
    def forward(self, inputs):
        hidden_states, attention_mask, position_ids, labels = inputs
        
        vocab_size = self.lm_head.weight.size(0)
        labels = labels.to(dtype=torch.long, device=hidden_states.device)
        labels = torch.clamp(labels, min=0, max=vocab_size-1)
        
        orig_dtype = hidden_states.dtype
        
        logits = self.lm_head(hidden_states)
        
        del hidden_states
        torch.cuda.empty_cache()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        del logits, labels
        torch.cuda.empty_cache()
        
        shift_logits_float = shift_logits.to(torch.float32)
        
        del shift_logits
        torch.cuda.empty_cache()
        
        # if not shift_logits_float.requires_grad:
        #     shift_logits_float.requires_grad_(True)
        
        loss = self.loss_fct(
            shift_logits_float.view(-1, shift_logits_float.size(-1)),
            shift_labels.view(-1)
        )
        
        del shift_logits_float, shift_labels
        torch.cuda.empty_cache()
        
        loss = loss.to(orig_dtype)
        # if not loss.requires_grad:
        #     loss.requires_grad_(True)
        
        return loss


def loss_fn(loss_tensor, *args, **kwargs):
    if not loss_tensor.requires_grad:
        loss_tensor.requires_grad_(True)
    return loss_tensor


# sin, cos tensors cached for all devices
cos_cached = None
sin_cached = None

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq.to(get_accelerator().current_device_name())
        self.theta = theta
        # self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        base = rearrange(emb, 'n d -> 1 n d')
        rope = [base.cos(), base.sin()]
        return rope


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    t_pass = None
    if t.shape[-1] != rot_dim:
        # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    global cos_cached, sin_cached
    if cos_cached is None or sin_cached is None or t.shape[0] != cos_cached.shape[0]:
        freqs_ = freqs[:t.shape[0]]
        cos_cached = freqs_.cos().to(t.dtype)
        sin_cached = freqs_.sin().to(t.dtype)
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_cached) + (_rotate_half(t) * sin_cached)
    if t_pass is None:
        return t
    return torch.cat((t, t_pass), dim=-1)
