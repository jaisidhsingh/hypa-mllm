import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.configs.tokenizer_configs import tokenizer_configs


class FeatureAlignmentDataset(Dataset):
    def __init__(self, image_folder, annotations_path, image_token, transform=None, tokenizer=None):
        self.image_folder = image_folder 
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        self.image_paths = [os.path.join(self.image_folder, item["image"]) for item in annotations]
        self.prompts = [item["conversations"][0]["value"] for item in annotations]
        self.answers = [item["conversations"][1]["value"] for item in annotations]
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_token = image_token

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image) 

        prompt = self.prompts[idx]
        answer = self.answers[idx]

        image_position = prompt.index(self.image_token)
        if image_position == len(prompt) - len(self.image_token):
            image_position = -1

        prompt = prompt.replace(self.image_token, "")

        return image, [prompt, answer], image_position
