from types import SimpleNamespace


data_configs = SimpleNamespace(**{})

data_configs.global_data_folder = "/home/mila/s/sparsha.mishra/scratch/tinyllava-pretrain"

data_configs.pretraining_dataset_configs = {
    "image_folder": f"{data_configs.global_data_folder}/tinyllava_558k/images",
    "annotations_path": f"{data_configs.global_data_folder}/tinyllava_558k/blip_laion_cc_sbu_558k.json",
    "image_token": "<image>",
    "transform": None,
    "tokenizer": None 
}
