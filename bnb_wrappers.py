from peft.tuners.lora.bnb import Linear8bitLt, Linear4bit

import warnings
import copy

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

from peft.tuners.lora.layer import LoraLayer

class CustomLinear8bitLt(Linear8bitLt):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            #*** [MODIFIED] ***
            kwargs_copy = copy.deepcopy(kwargs)
            kwargs_copy.pop('adapter_activation', None)
            #*** [END OF MODIFICATION] ***
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs_copy)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs_copy)
            else:
                result = self.base_layer(x, *args, **kwargs_copy)
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
                    output = lora_B(lora_A(dropout(x)))
                    #*** [MODIFIED] ***
                    if "adapter_activation" in kwargs:
                        output = output * kwargs["adapter_activation"]
                    #*** [END OF MODIFICATION] ***
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    output = output * scaling
                    result += output

            return result

Linear8bitLt.forward = CustomLinear8bitLt.forward