import copy

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm.autonotebook import tqdm

import numpy as np

class GaussianData(Dataset):
    def __init__(
        self, 
        params: dict,
        train: bool=True,
        **kwargs,
    ):
        # params = {
        #   // Parameters of the Gaussian Process
        #   "mu": 0.0,
        #   "sigma": 1.0,
        #   // Image Extensions
        #   "img_width_px": 32,
        #   "img_height_px": 32,
        #   "img_ch_num": 3,
        # }
        self.mu = params["mu"]
        self.sigma = params["sigma"]
        
        # dimensions of the CIFAR10 dataset
        # 60 000 samples in total
        self.img_width_px = params["img_width_px"]
        self.img_height_px = params["img_height_px"]
        self.img_ch_num = params["img_ch_num"]
        
        if train:
            self.len = 50000
            
        else:
            self.len = 10000
            
        #self.data = torch.rand(
        #    (self.len, self.img_width_px, self.img_height_px, self.img_ch_num, ),
        #)
        #self.data = self.data.normal_(mean=self.mu, std=self.sigma)
        
        #self.targets = torch.zeros((self.len, 1, ))
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        img_gauss = torch.rand(
            (self.img_width_px, self.img_height_px, self.img_ch_num, ),
        )
        img_gauss = img_gauss.normal_(mean=self.mu, std=self.sigma)
        
        dummy_target = torch.Tensor([0.0])
        #return self.data[index], self.targets[index]
        return img_gauss, dummy_target


class LpUnitCIFAR10(datasets.CIFAR10):
    def __init__(self, root, transform, train=True, device=torch.device('cpu'), p=2.0, **kwargs):
        super(LpUnitCIFAR10, self).__init__(
            root=root, transform=transform, train=train, download=True, **kwargs,
        )
        self.p = p
        self.flat = nn.Flatten()
        self.device = device
        self.data = torch.tensor(self.data.astype('float32'))
        #self.data = self.data.to(self.device)
        self.data /= torch.norm(self.flat(self.data), p=self.p, dim=-1)[:, None, None, None]
        self.data = self.data.detach().cpu().numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Args:  index (int): index of the data set
        # Value: feature, target (tuple): feature-target pair 
        #                                 with respective index in the dataset
        return self.data[index], self.targets[index]


class LpUnitMNIST(datasets.MNIST):
    def __init__(self, root, train=True, device=torch.device('cpu'), p=2.0, **kwargs):
        super(LpUnitMNIST, self).__init__(
            root=root, transform=ToTensor(), train=train, download=True, **kwargs,
        )
        self.p = p
        self.flat = nn.Flatten()
        self.device = device
        self.data = self.data.type(torch.float32)
        #self.data = self.data.to(self.device)
        self.data /= torch.norm(self.flat(self.data), p=self.p, dim=-1)[:, None, None]
        self.data = self.data.detach().cpu().numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Args:  index (int): index of the data set
        # Value: feature, target (tuple): feature-target pair 
        #                                 with respective index in the dataset
        return self.data[index], self.targets[index]
    

class BaselineAccurateTestData(Dataset):

    def __init__(
            self, 
            model: nn.Module, 
            dataset: Dataset,
            ):
        super().__init__()
        
        mask = torch.zeros((len(dataset.data), ), dtype=torch.bool)
        loader = DataLoader(dataset, shuffle=False)
        with tqdm(loader, unit="batch") as tbatch:
            tbatch.set_description(f"Testing dataset [batch]")

            with torch.no_grad():
                model.pred()
                for batch_nr, batch in enumerate(tbatch):
                    data, labels = batch
                    data = data.to(model.device)
                    labels = labels.to(model.device)
                    preds, _ = model(data)
                    batch_size = len(data)
                    mask[batch_nr*batch_size:(batch_nr+1*batch_size)] = torch.eq(preds, labels)
        
        self.data = dataset.data[mask].copy()
        self.targets = list(np.array(dataset.targets)[mask]).copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



class DeviceDataLoader(DataLoader):
    # DataLoader class that operates on datasets entirely moved 
    # to a specific device
    # Should only be used with datasets that are small enough to be stored
    # on the respective device.
    # Speeds up operation by a lot, because data does not has to be moved
    # between devices during training or inference.

    def __init__(
            self,
            dataset,
            device=None,
            batch_size=100,
            num_workers=0,
            **kwargs,
            ):
        if type(device) == type(None):
            raise ValueError("device=None -- Please specify the device to compute on (CPU/GPU)")

        if device != torch.device('cpu'):
            num_workers = 0

        # initialize DataLoader with dataset and parameters
        super(DeviceDataLoader, self).__init__(dataset, batch_size=batch_size,num_workers=num_workers,**kwargs)

        # move the data to the specified device
        if type(self.dataset.data) != torch.Tensor:
            self.dataset.data = torch.tensor(self.dataset.data)
            self.dataset.targets = torch.tensor(self.dataset.targets)
        
        if self.dataset.data.device != device:
            self.dataset.data = self.dataset.data.to(device)
            self.dataset.targets = self.dataset.targets.to(device)
