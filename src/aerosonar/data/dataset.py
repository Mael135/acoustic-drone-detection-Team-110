import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from torchaudio import transforms
METADATA_PATH = 'data\processed\metadata.csv'
TENSOR_DIR = 'data\processed'
BATCH_SIZE = 16
TRAIN_PART = 0.8

class SpectrogramTensorDataset(Dataset):
    def __init__(self, metadata_file, data_dir, transform=None, train=False):
        super().__init__()
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        file_name = self.metadata.iloc[idx, 0] 
        file_path = os.path.join(self.data_dir, file_name)
        
        sample = torch.load(file_path)
        
        label = self.metadata.iloc[idx, 1]
        label = torch.tensor(label)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    


# dataset = SpectrogramTensorDataset(metadata_file=METADATA_PATH, data_dir=TENSOR_DIR)

# unique_ids = dataset.metadata['file_id'].unique()
# np.random.seed(42) # For reproducibility
# np.random.shuffle(unique_ids)

# train_count = int(TRAIN_PART * len(unique_ids))
# train_ids = unique_ids[:train_count]
# test_ids = unique_ids[train_count:]

# train_indices = dataset.metadata.index[dataset.metadata['file_id'].isin(train_ids)].tolist()
# test_indices = dataset.metadata.index[dataset.metadata['file_id'].isin(test_ids)].tolist()

# train_dataset = torch.utils.data.Subset(dataset, train_indices)
# test_dataset = torch.utils.data.Subset(dataset, test_indices)

# # train_size = int(TRAIN_PART * len(dataset))
# # test_size = int(len(dataset) - train_size)
# # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_transforms = torch.nn.Sequential(
    transforms.FrequencyMasking(freq_mask_param=10), # Max width of frequency mask
    transforms.TimeMasking(time_mask_param=8),      # Max width of time mask
)
train_base = SpectrogramTensorDataset(metadata_file=METADATA_PATH, data_dir=TENSOR_DIR, train=True, transform=train_transforms)
test_base = SpectrogramTensorDataset(metadata_file=METADATA_PATH, data_dir=TENSOR_DIR, train=False)

# 2. Split logic (remains the same)
unique_ids = train_base.metadata['file_id'].unique()
np.random.seed(42)
np.random.shuffle(unique_ids)

train_count = int(TRAIN_PART * len(unique_ids))
train_ids = unique_ids[:train_count]
test_ids = unique_ids[train_count:]

train_indices = train_base.metadata.index[train_base.metadata['file_id'].isin(train_ids)].tolist()
test_indices = test_base.metadata.index[test_base.metadata['file_id'].isin(test_ids)].tolist()

# 3. Create subsets pointing to the CORRECT base dataset
train_dataset = torch.utils.data.Subset(train_base, train_indices)
test_dataset = torch.utils.data.Subset(test_base, test_indices)

# 4. DataLoaders (remain the same)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
