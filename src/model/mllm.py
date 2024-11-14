"""
Start off with just getting an idea for modelling.
Keep in mind that we currently just need to train the connector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLLM(nn.Module):
    def __init__(self, llm, tokenizer, vision_tower, connector, model_dim):
        llm.eval()
        vision_tower.eval()
        connector.eval()

        self.llm = llm
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.connector = connector
        self.dim = model_dim
    
    def forward_image_features(self, image):
        features = self.vision_tower(image)
        aligned_features = self.connector(features)
        return aligned_features
    
    def combine_image_text_input(self, text_inputs, image_features):
        return None

    def get_llm_output(self, inputs):
        return None

    def forward(self, input_ids, image_inputs):
        inputs = self.tokenizer(input_ids)
        image_features = self.forward_image_features(image_inputs)
        multi_modal_input = self.combine_image_text_input(inputs, image_features)
        output = self.get_llm_output(multi_modal_input)
        return output
