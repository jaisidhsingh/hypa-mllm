import torch
from src.configs.tokenizer_configs import tokenizer_configs


class DataCollator():
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, batch):
        """
        Reference: https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/main/tinyllava/data/dataset.py
        """
        images = [item[0] for item in batch]
        prompts = [item[1][0] for item in batch]
        answers = [item[1][1] for item in batch]
        image_positions = [item[2] for item in batch]

        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        prompt_ids = self.tokenizer(prompts).input_ids
        prompt_ids = [torch.tensor(pi).long() for pi in prompt_ids]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in prompt_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = tokenizer_configs.alt_ignore_index
            
        prompt_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        answer_ids = self.tokenizer(answers).input_ids
        answer_ids = [torch.tensor(ai).long() for ai in answer_ids]

        answer_ids = torch.nn.utils.rnn.pad_sequence(
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
        
        if self.device is None:
            self.device = "cpu"
        
        batch = {
            "input_ids": prompt_ids.to(self.device),
            "images": torch.stack(images).to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": answer_ids.to(self.device),
            "image_positions": torch.tensor(image_positions).long().to(self.device)
        }
        return batch
