import os
import math
import wandb
import argparse
import warnings
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

from src.model import MLLM
from src.data import FeatureAlignmentDataset
from src.training.trainers import TrainerForMLLM
from src.configs.data_configs import data_configs
warnings.simplefilter("ignore")


def main(args):
    model = MLLM(
        llm="llama-3.2",
        vision_tower="vanilla_vit_b16",
        connector_type="linear",
        connector_hidden_dims=[],
        device=args.device
    )
    model.train()

    train_dataset_config = data_configs.pretraining_dataset_configs["train"]
    train_dataset_config.update({"transform": model.image_transform, "tokenizer": model.tokenizer, "device": args.device})
    train_dataset = FeatureAlignmentDataset(**train_dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

    for i in range(4):
        print(batch["input_ids"])
        
        batch = next(iter(train_loader))
        output = model(**batch)
        print(output.loss)
        print(output.logits.shape)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment_name", type=str, default="mllm_training_test_0")
    parser.add_argument("--experiment_type", type=str, default="mllm")
    parser.add_argument("--train_log_folder", type=str, default="../../logs")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()
    main(args)

