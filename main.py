from src.configs.data_configs import data_configs
from src.data import FeatureAlignmentDataset


dataset = FeatureAlignmentDataset(**data_configs.pretraining_dataset_configs)

print(len(dataset))
sample = dataset[0]
for item in sample:
    print(type(item))
