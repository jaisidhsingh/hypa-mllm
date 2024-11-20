import os
import json
from PIL import Image
from torch.utils.data import Dataset


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
        images = [item[0] for item in batch]
        prompts = [item[1][0] for item in batch]
        answers = [item[1][1] for item in batch]
        image_positions = [item[2] for item in batch]
        return images, prompts, answers, image_positions
