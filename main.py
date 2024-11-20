from src.configs.data_configs import data_configs
from src.data import FeatureAlignmentDataset
from torch.utils.data import DataLoader


dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs)
loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
images, prompts, answers, image_positions = next(loader)

print(images)
print(prompts)
print(answers)
print(image_positions)
