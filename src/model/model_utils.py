import torch
from torch import Tensor
import torch.nn as nn
from typing import *
import timm
from transformers import AutoTokenizer, AutoModelForCausalLM


class MLP(nn.Module):
	def __init__(
        self, 
        input_dim: int, 
        intermediate_dims: List[int], 
        output_dim: int, 
        use_bias: bool = True
    ) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.intermediate_dims = intermediate_dims
		self.output_dim = output_dim
		self.num_layers = len(intermediate_dims) + 1

		self.layers = []
		current_dim = input_dim
		next_dims = intermediate_dims + [output_dim]

		for i in range(self.num_layers):
			self.layers.append(nn.Linear(current_dim, next_dims[i], bias=use_bias))
			current_dim = next_dims[i]

			if i != self.num_layers - 1:
				self.layers.append(nn.GELU())

		self.layers = nn.Sequential(*self.layers)

	def forward(self, x: Tensor) -> Tensor:
		return self.layers(x)


def load_llm(llm_name, device):
    supported_llms_map = {
        "llama-3.2-1b": "meta/Llama-3.2-1B",
        "phi-3": "microsoft/phi-3",
        "gemma-2": "google/gemma-2"
    }
    if llm_name not in supported_llms_map:
        raise KeyError("The LLM name you provided is not supported.")

    llm_id = supported_llms_map[llm_name]
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    model = AutoModelForCausalLM.from_pretrained(llm_id, device_map=device)
    return model, tokenizer


def load_vision_tower(vision_tower_name, device):
    supported_vision_towers_map = {
        "clip_vit_b16": "vit_base_patch16_224_clip.openai",
        "siglip_vit_l14": "vit_large_patch14_siglip.google",
    }
    if vision_tower_name not in supported_vision_towers_map:
        raise KeyError("The vision tower name you provided is not supported.")

    vision_tower_id = supported_vision_towers_map[vision_tower_name]
    model = timm.create_model(vision_tower_id, pretrained=True, num_classes=0)
    model = model.to(device)
    transform = None
    return model, transform


def load_connector(connector_type, vision_tower_dim, llm_dim, hidden_dims, device):
    supported_connector_types_map = {
        "linear": MLP,
        "mlp": MLP
    }
    if connector_type not in supported_connector_types_map:
        raise KeyError("The connectory type you provided is not supported.")

    if connector_type == "linear" and len(hidden_dims) > 0:
        raise AttributeError(f"Linear connectors should not have hidden dims, but you provided: {hidden_dims}.")

    connector_object = supported_connector_types_map[connector_type]
    connector = connector_object(vision_tower_dim, hidden_dims, llm_dim).to(device)
    return connector
