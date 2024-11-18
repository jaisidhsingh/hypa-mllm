import torch
import torch.nn as nn
from torch import Tensor

from typing import *

from src.model.model_utils import load_llm, load_vision_tower, load_connector


class MLLM(nn.Module):
    def __init__(
        self, 
        llm: str, 
        vision_tower: str, 
        connector_type: str, 
        connector_hidden_dims: Optional[List[int]] = [], 
        modules_to_freeze: Optional[List[str]] = ["vision_tower", "llm"],
        device="cpu"
        ) -> None:
        super().__init__()
        self.device = device
        self.llm, self.tokenizer = load_llm(llm, device=self.device)

        self.vision_tower, self.image_transform = load_vision_tower(vision_tower, self.device)

        vision_tower_dim = self.vision_tower.embed_dim
        llm_dim = self.llm.config.hidden_size

        self.connector = load_connector(connector_type, vision_tower_dim, llm_dim, connector_hidden_dims, self.device)
        self.dim = self.llm.config.hidden_size
        
        self.modules_to_freeze = modules_to_freeze
        self.freeze_modules()
        self.set_trainable_modules()

    
    def freeze_modules(self) -> None:
        for module in self.modules_to_freeze:
            if hasattr(self, module):
                getattr(self, module).eval()

                for p in getattr(self, module).parameters():
                    p.requires_grad = False
            else:
                raise AttributeError("The module you want to freeze does not exist in the model.")

    def set_trainable_modules(self) -> None:
        modules = ["vision_tower", "connector", "llm"]
        trainable_modules = list(set(modules) - set(self.modules_to_freeze))
        for module in trainable_modules:
            getattr(self, module).train()
    
    def get_image_features(self, images):
        image_features = self.vision_tower.forward_features(images)
        return image_features
    
    def connect_vision_tower_to_llm(self, image_features):
        return self.connector(image_features)

    def forward_embeddings_through_llm(self, input_embeds, attention_mask=None, labels=None):
        return self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
    
    def forward(self, text_input_ids: Tensor, images: Tensor, attention_mask: Tensor = None, labels: Tensor = None) -> Dict[str, Tensor]:
        image_features = self.vision_tower.forward_features(images)
        batch_size, num_patches, _ = image_features.shape
        projected_image_features = self.connector(image_features)

        combined_attention_mask = None
        if attention_mask is not None:
            image_attention_mask = torch.ones(
                (batch_size, num_patches), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)
        combined_embeddings = torch.cat([projected_image_features, text_embeddings], dim=1)

        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
 