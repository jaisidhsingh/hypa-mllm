from torch.utils.data import DataLoader
from src.configs.data_configs import data_configs
from src.data import FeatureAlignmentDataset
from src.model import run_model_unit_tests

run_model_unit_tests()


dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs)
loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
images, prompts, answers, image_positions = next(iter(loader))

print(images)
print(prompts)
print(answers)
print(image_positions)
