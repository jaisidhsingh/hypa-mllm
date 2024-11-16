import torch
import torch.nn as nn
from torch import Tensor
from typing import *


class MLLM(nn.Module):
    def __init__(self, llm: nn.Module, vision_tower: nn.Module, connector: nn.Module, modules_to_freeze: Optional[List] = ["vision_tower", "llm"]) -> Tensor:
        llm.eval()
        vision_tower.eval()
        connector.eval()

        self.llm = llm
        self.vision_tower = vision_tower
        self.connector = connector
        self.modules_to_freeze = modules_to_freeze
        self.dim = self.llm.config.hidden_size
    
    def freeze_modules(self) -> None:
        for module in self.modules_to_freeze:
            if hasattr(self, module):
                for p in getattr(self, module).parameters():
                    p.requires_grad = True
            else:
                raise AttributeError("The module you want to freeze does not exist in the model.")
    
    def forward(self, text_input_ids: Tensor, images: Tensor, attention_mask: Tensor = None, labels: Tensor = None) -> Dict[str, Tensor]:
        image_features = self.vision_tower.forward_features(images)
        batch_size, num_patches, _ = image_features.shape[0]
        projected_image_features = self.connector(image_features)

        if attention_mask is not None:
            image_attention_mask = torch.ones(
                (batch_size, num_patches), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        else:
            combined_attention_mask = None
        
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
        combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)

        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
 