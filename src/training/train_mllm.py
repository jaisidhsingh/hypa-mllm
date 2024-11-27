from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import argparse
import wandb
import math

from src.data import FeatureAlignmentDataset
from src.model import MLLM
from src.training.trainers import TrainerForMLLM
from src.configs.data_configs import data_configs


def main(args):
    train_dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs)
    model = MLLM(
        llm="llama-3.2",
        vision_tower="vanilla_vit_b16",
        connector_type="linear",
        connector_hidden_dims=[],
        device=args.device
    )
    model.train()

    wandb.init(project=args.experiment_type, config=vars(args))

    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.num_epochs

    training_args = TrainingArguments(
        output_dir=args.train_log_folder,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        report_to="wandb",
        run_name=args.experiment_name,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=total_steps,
    )
    trainer = TrainerForMLLM(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_dataset.collate_fn
    )
    
    try:
        trainer.train()
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_log_folder", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
