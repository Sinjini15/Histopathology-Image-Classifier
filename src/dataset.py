# src/dataset.py

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HistopathologyH5Dataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_path = x_path  # path to images
        self.y_path = y_path  # path to labels
        self.transform = transform

        # Open the HDF5 files
        self.h5file_x = h5py.File(self.x_path, 'r')
        self.h5file_y = h5py.File(self.y_path, 'r')

        # Assuming the dataset inside each file is called 'x' and 'y'
        self.images = self.h5file_x['x']
        self.labels = self.h5file_y['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image to float32 tensor and normalize [0,1]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long).squeeze()
        return image, label

    def close(self):
        self.h5file_x.close()
        self.h5file_y.close()


def get_dataloaders(train_x_path, train_y_path,
                    val_x_path, val_y_path,
                    test_x_path, test_y_path,
                    batch_size=64):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
    ])

    val_test_transform = transforms.Compose([])

    # Separate datasets
    train_dataset = HistopathologyH5Dataset(train_x_path, train_y_path, transform=train_transform)
    val_dataset = HistopathologyH5Dataset(val_x_path, val_y_path, transform=val_test_transform)
    test_dataset = HistopathologyH5Dataset(test_x_path, test_y_path, transform=val_test_transform)

    # Separate DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
