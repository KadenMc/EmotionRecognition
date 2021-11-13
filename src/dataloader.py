import numpy as np
import torch
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class to handle data conversion
    and transforms.
    """
    def __init__(self, data, targets,  transform=False):
        
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)
        self.classes = np.sort(np.unique(self.targets))
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
          x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)



def prepare_dataloaders(X, y, args, transform=False, seed=19283756):
    dataset = Dataset(X, y, transform=transform)
    
    val_n = int(len(dataset) * args.val_percent)
    test_n = int(len(dataset) * args.test_percent)
    train_n = len(dataset) - (val_n + test_n)
    
    # Split the dataset in to training, validation and testing datasets
    if seed is None:
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, \
            [train_n, val_n, test_n])
    else:
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, \
            [train_n, val_n, test_n], generator=torch.Generator().manual_seed(19283756))
    
    # Create mini-batch dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader, test_loader