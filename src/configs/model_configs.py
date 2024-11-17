from types import SimpleNamespace


model_configs = SimpleNamespace(**{})

# store vision tower configs
model_configs.supported_vision_towers_map = {
    "clip_vit_b16": "vit_base_patch16_224_clip.openai",
    "siglip_vit_l14": "vit_large_patch14_siglip.google",
}

# store llm configs
model_configs.supported_llms_map = {
    "llama-3.2-1b": "meta/Llama-3.2-1B",
    "phi-3": "microsoft/phi-3",
    "gemma-2": "google/gemma-2"
}
