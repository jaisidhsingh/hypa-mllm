from torch.utils.data import DataLoader
from src.configs.data_configs import data_configs
from src.data import FeatureAlignmentDataset
from src.model import run_model_unit_tests, MLLM
import torch
# run_model_unit_tests()


dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs)
loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
batch = next(iter(loader))

for k, v in batch.items():
    print(k, v.shape)

model = MLLM(
    llm="llama-3.2",
    vision_tower="vanilla_vit_b16",
    connector_type="linear",
    connector_hidden_dims=[],
    device="cpu"
)
model.train()

output = model(**batch)
print(output.keys())
