from peft.tuners.lora import LoraLayer
from peft.tuners.lora import Linear8bitLt

import warnings
import copy

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

class CustomLinear8bitLt(Linear8bitLt):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            model_status = kwargs["model_status"]
            # the backbone of the LLM path
            hidden_states = x
            # the lora of the LLM path
            if model_status == "eval":
                lora_hidden_states = kwargs['lora_hidden_states']
            # pop out 'adapter_activation' from the kwargs used to be input to the base linear layer
            kwargs_copy = copy.deepcopy(kwargs)
            kwargs_copy.pop('adapter_activation', None)
            kwargs_copy.pop('lora_hidden_states', None)
            kwargs_copy.pop('model_status', None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                # get the backbone linear output
                result = self.base_layer(hidden_states, *args, **kwargs_copy)
                # get the lora original linear output
                if model_status == "eval":
                    lora_result = self.base_layer(lora_hidden_states, *args, **kwargs_copy)
            elif self.merged:
                result = self.base_layer(hidden_states, *args, **kwargs_copy)
                if model_status == "eval":
                    lora_result = self.base_layer(lora_hidden_states, *args, **kwargs_copy)
            else:
                result = self.base_layer(hidden_states, *args, **kwargs_copy)
                if model_status == "eval":
                    lora_result = self.base_layer(lora_hidden_states, *args, **kwargs_copy)

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.lora_A.keys():
                        continue
                    lora_A = self.lora_A[active_adapter]
                    lora_B = self.lora_B[active_adapter]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    
                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lora_A.weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)
                    # compute the lora output by inputing the lora hidden states
                    if model_status == "eval":
                        lora_output = lora_B(lora_A(dropout(lora_hidden_states)))
                    elif model_status == "train":
                        lora_output = lora_B(lora_A(dropout(hidden_states)))
                    if "adapter_activation" in kwargs:
                        lora_output = lora_output * kwargs["adapter_activation"]
                    if requires_conversion:
                        lora_output = lora_output.to(expected_dtype)
                    lora_output = lora_output * scaling
                    if model_status == "eval":
                        lora_result = lora_result + lora_output
                    elif model_status == "train":
                        lora_result = result + lora_output
            
            if model_status == "eval":
                return result, lora_result
            elif model_status == "train":
                return lora_output

Linear8bitLt.forward = CustomLinear8bitLt.forward