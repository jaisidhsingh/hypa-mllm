from .alignment_dataset import FeatureAlignmentDataset
from .data_collator import DataCollator
from src.configs.data_configs import data_configs
from src.model.model_utils import load_tokenizer, load_transform


def load_pretraining_dataset(args, split="train"):
    transform = load_transform(args.vision_tower_name)
    tokenizer = load_tokenizer(args.llm_name)

    dataset_config = data_configs.pretraining_dataset_configs[split]
    dataset_config.update({"transform": transform})

    dataset = FeatureAlignmentDataset(**dataset_config)
    collator = DataCollator(**{"tokenizer": tokenizer, "device": args.device})

    return dataset, collator
