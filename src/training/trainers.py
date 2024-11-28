"""
Modify huggingface's Trainer() for
1. default MLLM training
2. hyper-MLLM training
"""
import torch
from transformers import Trainer


class TrainerForMLLM(Trainer):
    def create_optimizer(self):
        opt_params = self.model.get_trainable_params()
            
        self.optimizer = torch.optim.AdamW(
            opt_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        return self.optimizer