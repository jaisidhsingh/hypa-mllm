import os
import math
import wandb
import argparse
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

from src.model import MLLM
from src.data import FeatureAlignmentDataset
from src.training.trainers import TrainerForMLLM
from src.configs.data_configs import data_configs


def main(args):
    train_dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs["train"])
    # val_dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs["val"])

    model = MLLM(
        llm="llama-3.2",
        vision_tower="vanilla_vit_b16",
        connector_type="linear",
        connector_hidden_dims=[],
        device=args.device
    )
    model.train()

    wandb.login(key="de80fd57553b311eb0e3b3d71e72fe38d9b3524c")
    wandb.init(project=args.experiment_type, entity="hyperalignment", name=args.experiment_name, config=vars(args))

    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.num_epochs
    train_log_folder = os.path.join(args.train_log_folder, args.experiment_name)
    os.makedirs(train_log_folder, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=train_log_folder,
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
        # eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn
    )
    
    try:
        trainer.train()
    finally:
        wandb.finish()


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
