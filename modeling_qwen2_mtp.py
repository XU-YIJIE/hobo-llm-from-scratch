import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2DecoderLayer, Qwen2Model
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.loss.loss_utils import fixed_cross_entropy
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class MTPSub(nn.Module):
    def __init__(self, config:Qwen2Config):
        super().__init__()
        self.config = config
        self.norm = Qwen2RMSNorm(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.transformer = Qwen2DecoderLayer(config, -1)
    
    def forward(self, hidden_states, inputs_embeds, position_embeds, **kwargs):
        concat = torch.cat([self.norm(hidden_states), self.norm(inputs_embeds)], dim=-1)
        proj = self.linear(concat)
        layer_output = self.transformer(proj, position_embeddings=position_embeds, **kwargs)
        hidden_state = layer_output[0]
        return hidden_state

class MTP(nn.Module):
    def __init__(self, config:Qwen2Config):
        super().__init__()
        self.config = config
        self.num_additional_preds = config.num_additional_preds
        self.mtps = nn.ModuleList([MTPSub(config) for _ in range(self.num_additional_preds)])

    def forward(self, hidden_states, inputs_embeds, position_embeds, lm_head:nn.Linear):
        # 0,1,2,3,4: 0,1,2,3 => 1,2,3,4
        # 0,1 => 1,2 -> 2,3 -> 3,4  # num_additional_preds = 2
        logits_list = []
        _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states[:, 1:seq_len-self.num_additional_preds, :]
        for k in range(1, self.num_additional_preds+1):
            start_index = k
            end_index = start_index + seq_len - self.num_additional_preds - 1
            cos, sin = position_embeds
            hidden_states = self.mtps[k-1](hidden_states, inputs_embeds[:, start_index:end_index, :], (cos[:, start_index:end_index, :], sin[:, start_index:end_index, :]))
            logits = lm_head(hidden_states)  # b, seq_len-k, vocab_size
            logits_list.append(logits)
        return hidden_states, logits_list

class Qwen2MTPForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = MTP(config)
        self._copy_tfblock_weights_to_mtp()
        # Initialize weights and apply final processing
        self.post_init()
        
    def _copy_tfblock_weights_to_mtp(self):
        source_layers = self.model.layers[-self.config.num_additional_preds:]
        for i, mtp_sub in enumerate(self.mtp.mtps):
            source_layer = source_layers[i]
            target_layer = mtp_sub.transformer
            with torch.no_grad():
                for target_name, target_param in target_layer.named_parameters():
                    for source_name, source_param in source_layer.named_parameters():
                        if target_name == source_name:
                            target_param.data.copy_(source_param.data)
                            break
        
    def forward(self, input_ids, labels=None, inputs_embeds=None, position_ids=None, num_logits_to_keep=0, past_key_values=None, use_cache=False, cache_position=None, return_dict=False, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        
        if not inputs_embeds:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if not position_ids:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        hidden_states, logits_list = self.mtp(hidden_states, inputs_embeds, position_embeddings, self.lm_head)
        loss = None

        if self.training and labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            labels = labels.to(logits.device)
            _, seq_len, _ = logits.shape
            mtp_loss = 0
            for k in range(1, self.config.num_additional_preds+1):
                # 0,1,2,3,4: 0,1,2,3 => 1,2,3,4
                # 0,1 => 1,2 -> 2,3 -> 3,4  # num_additional_preds = 2
                start_index = k + 1
                end_index = start_index + seq_len - self.config.num_additional_preds - 1
                mtp_logits = logits_list[k-1].contiguous()
                mtp_labels = labels[..., start_index:end_index].contiguous()

                # Flatten the tokens
                mtp_logits = mtp_logits.view(-1, self.config.vocab_size)
                mtp_labels = mtp_labels.view(-1)
                mtp_labels = mtp_labels.to(mtp_logits.device)
                mtp_loss += fixed_cross_entropy(mtp_logits, mtp_labels)
                
            mtp_loss = self.config.mtp_lambda_weight * mtp_loss / self.config.num_additional_preds
            
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = fixed_cross_entropy(shift_logits, shift_labels) + mtp_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )