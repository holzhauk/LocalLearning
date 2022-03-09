import os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from LocalLearning import LocalLearningModel
from LocalLearning import train_unsupervised

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python train_CIFAR.py <modelpath>")
        os._exit(os.EX_NOINPUT)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = Path(sys.argv[1])
    if not os.path.exists(model_path.parent):
        os.makedirs(model_path.parent)

    model_ps = {
            "in_size": 32**2*3,  # CIFAR10 dataset consists of 32x32 pixel imgs with 3 channels each
        "hidden_size": 2000,
        "p": 3.0,
        "tau_l": 1.0 / 0.04,  # 1 / learning rate
        "k": 7,
        "Delta": 0.4,  # inhibition rate
        "R": 1.0,  # asymptotic weight norm radius
    }

    model = LocalLearningModel(model_ps)
    model.to(device=device)

    training_data = datasets.CIFAR10(
        root="../data/CIFAR10", train=True, download=True, transform=ToTensor()
    )

    dataloader_train = DataLoader(
        training_data, batch_size=64, num_workers=4, shuffle=True
    )
    train_unsupervised(dataloader_train, model, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_parameters": model.param_dict(),
            "device_type": device.type,
        },
        model_path,
    )
