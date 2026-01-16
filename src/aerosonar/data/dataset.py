import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import os

METADATA_PATH = 'data\processed\metadata.csv'
TENSOR_DIR = 'data\processed'
BATCH_SIZE = 64
TRAIN_PART = 0.8

class SpectrogramTensorDataset(Dataset):
    def __init__(self, metadata_file, data_dir, transform=None):
        super().__init__()
        self.metadata = pd.read_csv(metadata_file)
        self.data_dir = data_dir
        self.transform = transform

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
    


dataset = SpectrogramTensorDataset(metadata_file=METADATA_PATH, data_dir=TENSOR_DIR)
train_size = int(TRAIN_PART * len(dataset))
test_size = int(len(dataset) - train_size)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
