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

        answer = self.answers[idx]
        prompt = self.prompts[idx]

        image_position = prompt.index(self.image_token)
        if image_position == len(prompt) - len(self.image_token):
            image_position = -1

        prompt = prompt.replace(self.image_token, "")

        return image, [prompt, answer], image_position

    def collate_fn(self, batch):
        """
        Reference: https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/main/tinyllava/data/dataset.py
        """
        images = [item[0] for item in batch]
        prompts = [item[1][0] for item in batch]
        answers = [item[1][1] for item in batch]
        image_positions = [item[2] for item in batch]

        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        prompt_ids = self.tokenizer(prompts, return_tensors="pt").input_ids

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in prompt_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = tokenizer_configs.alt_ignore_index
            
        prompt_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        answer_ids = self.tokenizer(answers, return_tensors="pt")
        answer_ids = torch.nn.utils.rnn.pad_packed_sequence(
            answer_ids,
            batch_first=True,
            padding_value=tokenizer_configs.ignore_index
        )

        prompt_ids = prompt_ids[:, :self.tokenizer.model_max_length]
        attention_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        answer_ids = answer_ids[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in prompt_ids:
                input_id[input_id == tokenizer_configs.alt_ignore_index] = self.tokenizer.eos_token_id
        
        batch = {
            "input_ids": prompt_ids,
            "images": torch.stack(images),
            "attention_mask": attention_mask,
            "labels": answer_ids,
            "image_positions": torch.stack(image_positions)
        }
        return batch
