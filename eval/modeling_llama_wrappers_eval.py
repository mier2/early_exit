from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention, LlamaModel, LLAMA_INPUTS_DOCSTRING, LlamaDecoderLayer, LlamaRMSNorm, LlamaForCausalLM, _CONFIG_FOR_DOC, LlamaPreTrainedModel, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import(
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

logger = logging.get_logger(__name__)


class CustomLlamaMLP(LlamaMLP):
    def forward(self, x, lora_hidden_states=None, **kwargs):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            model_status = kwargs["model_status"]
            if "adapter_activation" in kwargs:
                if model_status == "eval":
                    kwargs["lora_hidden_states"] = lora_hidden_states
                    down_proj, lora_down_proj= self.down_proj(self.act_fn(self.gate_proj(x, **kwargs)) * self.up_proj(x, **kwargs), **kwargs)
                    return down_proj, lora_down_proj
                elif model_status == "train":
                    down_proj = self.down_proj(self.act_fn(self.gate_proj(x, **kwargs)) * self.up_proj(x, **kwargs), **kwargs)
                    return down_proj
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                return down_proj

class CustomLlamaAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        lora_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            model_status = kwargs["model_status"]
            if "adapter_activation" in kwargs:
                if model_status == "eval":
                    kwargs["lora_hidden_states"] = lora_hidden_states
                    query_states, lora_query_states = self.q_proj(hidden_states, **kwargs)
                    key_states, lora_key_states = self.k_proj(hidden_states, **kwargs)
                    value_states, lora_value_states = self.v_proj(hidden_states, **kwargs)
                elif model_status == "train":
                    kwargs.pop('exit_layers', None)
                    # print("*********************")
                    query_states = self.q_proj(hidden_states, **kwargs)
                    key_states = self.k_proj(hidden_states, **kwargs)
                    value_states = self.v_proj(hidden_states, **kwargs)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # lora part
        if model_status == "eval":
            lora_query_states = lora_query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            lora_key_states = lora_key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            lora_value_states = lora_value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # lora part
        if model_status == "eval":
            lora_cos, lora_sin = self.rotary_emb(lora_value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # lora part
        if model_status == "eval":
            lora_query_states, lora_key_states = apply_rotary_pos_emb(lora_query_states, lora_key_states, lora_cos, lora_sin, position_ids)

        if past_key_value is not None:
            print("************************past_key_value***********************")
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        
        # lora part
        if model_status == "eval":
            lora_past_key_value = (lora_key_states, lora_value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # lora part
        if model_status == "eval":
            lora_key_states = repeat_kv(lora_key_states, self.num_key_value_groups)
            lora_value_states = repeat_kv(lora_value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # lora part
        if model_status == "eval":
            lora_attn_weights = torch.matmul(lora_query_states, lora_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            # lora 
            if model_status == "eval":
                lora_attn_weights = lora_attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # lora part 
        if model_status == "eval":
            lora_attn_weights = nn.functional.softmax(lora_attn_weights, dim=-1, dtype=torch.float32).to(lora_query_states.dtype)
            lora_attn_output = torch.matmul(lora_attn_weights, lora_value_states)


        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # lora part
        if model_status == "eval":
            lora_attn_output = lora_attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # # lora part
        if model_status == "eval":
            lora_attn_output = lora_attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            print("*********************self.config.pretraining_tp*******************")
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            if "adapter_activation" in kwargs:
                if model_status == "eval":
                    attn_output, lora_attn_output = self.o_proj(attn_output, **kwargs)
                elif model_status == "train":
                    attn_output = self.o_proj(attn_output, **kwargs)
            else:
                attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
            if model_status == "eval":
                lora_attn_output = None
        
        if model_status == "eval":
            return attn_output, attn_weights, past_key_value, lora_attn_output, lora_attn_weights, lora_past_key_value
        elif model_status == "train":
            return attn_output, attn_weights, past_key_value

class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        lora_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        model_status = kwargs["model_status"]
        if model_status == "train":
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            # kwargs['model_status'] = "train"
            hidden_states = self.mlp(hidden_states, **kwargs)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs
        elif model_status == "eval":
            residual = hidden_states
            lora_residual = lora_hidden_states

            hidden_states = self.input_layernorm(hidden_states)
            lora_hidden_states = self.input_layernorm(lora_hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value, lora_hidden_states, lora_self_attn_weights, lora_present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                lora_hidden_states = lora_hidden_states,
                **kwargs,
            )

            hidden_states = residual + hidden_states
            lora_hidden_states = lora_residual + lora_hidden_states

            # Fully Connected
            residual = hidden_states
            lora_residual = lora_hidden_states

            hidden_states = self.post_attention_layernorm(hidden_states)
            lora_hidden_states = self.post_attention_layernorm(lora_hidden_states)

            hidden_states, lora_hidden_states = self.mlp(hidden_states, lora_hidden_states = lora_hidden_states, **kwargs)

            hidden_states = residual + hidden_states
            lora_hidden_states = lora_residual + lora_hidden_states

            outputs = (hidden_states,)
            lora_outputs = (lora_hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)
                # lora part
                lora_outputs += (lora_self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)
                # lora part
                lora_outputs += (lora_present_key_value,)

            return outputs, lora_outputs

class CustomLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self,config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.adapter_activation = torch.ones(len(self.layers))

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds
        lora_hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        model_status = kwargs["model_status"]

        
        if model_status == "eval":
            exit_layers = kwargs["exit_layers"]
            
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                #***************[MODIFY]***************
                if model_status == "eval":
                    all_hidden_states += (lora_hidden_states,)
                elif model_status == "train":
                    all_hidden_states += (hidden_states,)
            
            # if the current layer is one of the early exit layers, we need to 
            # set the lora_hidden_states as the backbone hidden states
            if model_status == "eval":
                if idx - 1 in exit_layers:
                    lora_hidden_states = hidden_states

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if "adapter_activation" in kwargs:
                #adapter_activation
                    if(len(kwargs["adapter_activation"]) == len(self.layers)) and not(torch.equal(self.adapter_activation.cuda(), kwargs["adapter_activation"].cuda())):
                        self.adapter_activation = kwargs["adapter_activation"]
                        print("received adapter_activation in LlamaModel: " + str(kwargs["adapter_activation"]))
                    
                    kwargs.update(adapter_activation=self.adapter_activation[idx].unsqueeze(0))
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    **kwargs,
                )
            else:
                if "adapter_activation" in kwargs:
                    #adapter_activation
                    if(len(kwargs["adapter_activation"]) == len(self.layers)) and not(torch.equal(self.adapter_activation.cuda(), kwargs["adapter_activation"].cuda())):
                        self.adapter_activation = kwargs["adapter_activation"]
                        print("received adapter_activation in LlamaModel: " + str(kwargs["adapter_activation"]))
                    
                    kwargs.update(adapter_activation=self.adapter_activation[idx].unsqueeze(0))
                if model_status == "train":
                    kwargs.pop('exit_layers', None)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )
                if model_status == "eval":
                    layer_outputs, lora_layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            lora_hidden_states = lora_hidden_states,
                            **kwargs
                        )
            
            if model_status == "train":
                hidden_states = layer_outputs[0]
            elif model_status == "eval":
                hidden_states = layer_outputs[0]
                lora_hidden_states = lora_layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                #***************[MODIFY]***************
                if model_status == "train":
                    all_self_attns += (layer_outputs[1],)
                elif model_status == "eval":
                    all_self_attns += (lora_layer_outputs[1],)
    
        if model_status == "train":
            hidden_states = self.norm(hidden_states)
        elif model_status == "eval":
            hidden_states = self.norm(hidden_states)
            lora_hidden_states = self.norm(lora_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            if model_status == "train":
                all_hidden_states += (hidden_states,)
            elif model_status == "eval":
                all_hidden_states += (lora_hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        if model_status == "train":
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        elif model_status == "eval":
            return BaseModelOutputWithPast(
                last_hidden_state=lora_hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


class CustomLlamaForCausalLM():
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

#monkey patch it to the original class
LlamaMLP.forward = CustomLlamaMLP.forward
LlamaAttention.forward = CustomLlamaAttention.forward
LlamaModel.__init__ = CustomLlamaModel.__init__
LlamaModel.forward = CustomLlamaModel.forward
LlamaForCausalLM.forward = CustomLlamaForCausalLM.forward
LlamaDecoderLayer.forward = CustomLlamaDecoderLayer.forward


