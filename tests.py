import torch
from src.model.mllm import MLLM


model = MLLM(
    llm="gpt-2",
    vision_tower="vanilla_vit_b16",
    connector_type="linear",
    connector_hidden_dims=[],
    modules_to_freeze=["llm", "vision_tower"],
    device="cpu"
)

# print(model)
images = torch.randn(1, 3, 224, 224)

for n,p in model.named_parameters():
    if p.requires_grad == True:
        print(n)

text = ["A noisy image"]
model.tokenizer.pad_token = model.tokenizer.eos_token

pad_embedding = model.llm.get_input_embeddings()(torch.tensor([[model.tokenizer.pad_token_id]], dtype=torch.long))
print(pad_embedding.shape, pad_embedding.norm())

inputs = model.tokenizer(text=text, padding=True, truncation=False, return_tensors="pt")
input_ids = inputs["input_ids"]
print(input_ids.shape)

attention_mask = inputs["attention_mask"]

labels = torch.full((1, 200), fill_value=-100, dtype=torch.long)

labels[:, 197:200] = input_ids.clone()

output = model(input_ids, images, attention_mask, labels=labels)
print(output.keys())
print(output["logits"].shape)

print(model.dim)