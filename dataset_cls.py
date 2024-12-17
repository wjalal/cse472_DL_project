# Custom Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
class ADNIDataset(Dataset):
    def __init__(self, features_list, sex_list, age_list):
        self.features = features_list
        self.sex = sex_list
        self.age = age_list

    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.sex[idx], dtype=torch.float32),
            torch.tensor(self.age[idx], dtype=torch.float32),
        )


class ADNIDatasetLite(Dataset):
    def __init__(self,image_list, sex_list, age_list):
        self.images = image_list
        self.sex = sex_list
        self.age = age_list

    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx], dtype=torch.float32),
            torch.tensor(self.sex[idx], dtype=torch.float32),
            torch.tensor(self.age[idx], dtype=torch.float32),
        )

class ADNIDatasetViT(Dataset):
    def __init__(self,image_list, age_list):
        self.images = image_list
        self.label = age_list

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.label[idx],
        )
