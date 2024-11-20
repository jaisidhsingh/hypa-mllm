from torch.utils.data import DataLoader
from src.configs.data_configs import data_configs
from src.data import FeatureAlignmentDataset
from src.model import run_model_unit_tests, MLLM
import torch
# run_model_unit_tests()


dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs)
loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
images, prompts, answers, image_positions = next(iter(loader))

print(images)
print(prompts)
print(answers)
print(image_positions)


image_size = 224

model = MLLM(
    llm="llama-3.2",
    vision_tower="vanilla_vit_b16",
    connector_type="linear",
    connector_hidden_dims=[],
    device="cpu"
)


images = torch.randn((1, 3, image_size, image_size))
input_text = "A cat sat on the"
input_text_enc = model.tokenizer(input_text, return_tensors="pt")
input_text_ids = input_text_enc.input_ids

target_text = " mat."
target_text_enc = model.tokenizer(target_text, return_tensors="pt")
target_text_ids = target_text_enc.input_ids

full_text = input_text + target_text
full_text_enc = model.tokenizer(full_text, return_tensors="pt")
full_text_ids = full_text_enc.input_ids
attention_mask = full_text_enc.attention_mask

labels = torch.full((1, model.num_patches+full_text_ids.shape[1]), fill_value=-100, dtype=torch.long)
print(labels.shape, input_text_ids.shape, target_text_ids.shape)
labels[:, model.num_patches+input_text_ids.shape[1]:] = target_text_ids

output = model(full_text_ids, images, attention_mask, labels)