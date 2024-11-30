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


class MLLMTrainer(object):
    def __init__(
            self, 
            args, 
            model, 
            optimizer,
            train_dataset,
            eval_dataset, 
            collator,
            logging_type="wandb",
            scheduler="linear_warmup_with_cosine_decay"
        ):
        self.args = args
        self.device = args.device
        self.model = model
        self.optimizer = optimizer

