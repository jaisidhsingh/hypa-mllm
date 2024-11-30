import os
import torch
import argparse
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model import MLLM
from src.data import load_pretraining_dataset
warnings.simplefilter("ignore")


def plot_metrics(metrics, name):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(metrics["steps"], metrics["train_loss"], label="train_loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training loss over time")
    axes[0].legend()

    axes[1].plot(metrics["steps"], metrics["train_ppl"], label="train_perplexity")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Train perplexity")
    axes[1].set_title("Training perplexity over time")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{name}.png")


def main(args):
    print("Initialising model...")
    modules_to_freeze=args.modules_to_freeze.split(",")
    model = MLLM(
        llm=args.llm_name,
        vision_tower=args.vision_tower_name,
        connector_type=args.connector_type,
        connector_hidden_dims=[],
        modules_to_freeze=modules_to_freeze,
        device=args.device
    )
    model.train()
    print("Model loaded.")
    print(" ")

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.amp.autocast(args.device)

    print("Loading datasets...")
    train_dataset, collator = load_pretraining_dataset(args, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)
    print("Datasets loaded.")
    print(" ")

    bar = tqdm(total=len(train_loader))
    logs = {"steps": [0], "train_loss": [0], "train_ppl": [0]}

    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        with autocast:
            output = model(**batch)
            loss = output.loss
        
        logs["steps"].append(idx+1)
        logs["train_loss"].append(loss.item())

        perplexity = torch.exp(torch.tensor(loss.item()))
        perplexity = round(perplexity.item(), 3)
        logs["train_ppl"].append(perplexity)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_postfix({"perplexity": perplexity})
        bar.update(1)
    
    bar.close()
    
    plot_metrics(logs, args.experiment_name)
    logs["config"] = vars(args)
    torch.save(logs, os.path.join(args.train_log_folder, f"{args.experiment_name}.pt"))
    
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, os.path.join(args.ckpt_folder, f"{args.experiment_name}.pt"))
    print("Saved model checkpoint and logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment_name", type=str, default="vanilla_vit-llama_3x2_1b-linear-cc558k")
    parser.add_argument("--experiment_type", type=str, default="mllm")
    parser.add_argument("--train_log_folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hypa-mllm/logs")
    parser.add_argument("--ckpt_folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hypa-mllm/checkpoints")
    
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    parser.add_argument("--llm_name", type=str, default="llama-3.2")
    parser.add_argument("--vision_tower_name", type=str, default="vanilla_vit_b16")
    parser.add_argument("--connector_type", type=str, default="linear")
    parser.add_argument("--modules_to_freeze", type=str, default="vision_tower,llm")

    args = parser.parse_args()
    main(args)
