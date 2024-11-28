import torch
import argparse
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import MLLM
from src.data import load_pretraining_dataset
warnings.simplefilter("ignore")


def main(args):
    model = MLLM(
        llm=args.llm_name,
        vision_tower=args.vision_tower_name,
        connector_type=args.connector_type,
        connector_hidden_dims=[],
        device=args.device
    )
    model.train()

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.amp.autocast(args.device)

    train_dataset, collator = load_pretraining_dataset(args, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)

    bar = tqdm(total=len(train_loader))
    for batch in train_loader:
        optimizer.zero_grad()

        with autocast:
            output = model(**batch)
            loss = output.loss
        
        perplexity = torch.exp(torch.tensor(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_postfix({"perplexity": perplexity.item()})
        bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment_name", type=str, default="mllm_training_test_0")
    parser.add_argument("--experiment_type", type=str, default="mllm")
    parser.add_argument("--train_log_folder", type=str, default="../../logs")
    
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    parser.add_argument("--llm_name", type=str, default="llama-3.2")
    parser.add_argument("--vision_tower_name", type=str, default="vanilla_vit_b16")
    parser.add_argument("--connector_type", type=str, default="linear")

    args = parser.parse_args()
    main(args)
