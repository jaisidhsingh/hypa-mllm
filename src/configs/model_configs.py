from types import SimpleNamespace


model_configs = SimpleNamespace(**{})

# store vision tower configs
model_configs.supported_vision_towers_map = {
    "vanilla_vit_b16": "vit_base_patch16_224",
    "clip_vit_b16": "vit_base_patch16_clip_224.openai",
    "siglip_vit_b16": "ViT-B-16-SigLIP",
}

# store llm configs
model_configs.supported_llms_map = {
    "llama-3.2": "meta-llama/Llama-3.2-1B",
    "qwen-2.5": "Qwen/Qwen2.5-1.5b",
    "gemma-2": "google/gemma-2-2b",
    "gpt-2": "openai-community/gpt2"
}
