import os, sys
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from matplotlib import pyplot as plt

# custom modules
from .LocalLearning import LocalLearningModel
from .Statistics import cov_spectrum
from .Statistics import stringer_spectrum


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"usage: python train_MNIST.py <modelpath>")
        os._exit(os.EX_NOINPUT)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = Path(sys.argv[1])
    model_trained = torch.load(model_path)  # load trained weights

    model_ps = model_trained["model_parameters"]
    model = LocalLearningModel(model_ps)
    model.load_state_dict(model_trained["model_state_dict"])
    model.eval()

    model.to(device)

    test_data = datasets.MNIST(
        root="../data/MNIST", train=False, download=True, transform=ToTensor()
    )

    dataloader_test = DataLoader(test_data, batch_size=64, num_workers=2, shuffle=False)

    if torch.isnan(model.W).any():
        print(f"model stored in {sys.argv[1]} contains NaN weights")
        os._exit(os.EX_DATAERR)

    l_spectrum = cov_spectrum(dataloader_test, model, device)
    cpu = torch.device('cpu')
    l_spectrum = l_spectrum.to(cpu).numpy()

    fig, ax = plt.subplots()
    ax.loglog(l_spectrum, label=r"$MNIST$")
    n = np.arange(1, len(l_spectrum))
    ax.loglog(stringer_spectrum(n), label=r"$1/n$")
    ax.set_ylabel(r"$\lambda_{n}$")
    ax.set_xlabel(r"$n$")
    ax.set_title(r"MNIST")
    ax.legend()
    fig_path = model_path.parent / model_path.stem
    fig_path = Path(str(fig_path) + "_CovSpectrumPlot.pdf")
    fig.savefig(fig_path)
