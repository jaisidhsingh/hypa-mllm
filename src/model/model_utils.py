import torch
import torch.nn as nn
from torch import Tensor

import timm
from typing import *
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs.model_configs import model_configs


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
    if llm_name not in model_configs.supported_llms_map:
        raise KeyError("The LLM name you provided is not supported.")

    llm_id = model_configs.supported_llms_map[llm_name]
    model = AutoModelForCausalLM.from_pretrained(llm_id, device_map=device)
    return model


def load_tokenizer(llm_name):
    if llm_name not in model_configs.supported_llms_map:
        raise KeyError("The LLM name you provided is not supported.")

    llm_id = model_configs.supported_llms_map[llm_name]
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    return tokenizer


def load_vision_tower(vision_tower_name, device):
    if vision_tower_name not in model_configs.supported_vision_towers_map:
        raise KeyError("The vision tower name you provided is not supported.")

    vision_tower_id = model_configs.supported_vision_towers_map[vision_tower_name]
    model = timm.create_model(vision_tower_id, pretrained=True, num_classes=0).to(device)
    config = timm.data.resolve_model_data_config(model)
    return model, config


def load_transform(vision_tower_name):
    if vision_tower_name not in model_configs.supported_vision_towers_map:
        raise KeyError("The vision tower name you provided is not supported.")

    vision_tower_id = model_configs.supported_vision_towers_map[vision_tower_name]
    config = timm.data.resolve_data_config({}, model=vision_tower_id)
    transform = timm.data.create_transform(**config, is_training=False)
    return transform


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
