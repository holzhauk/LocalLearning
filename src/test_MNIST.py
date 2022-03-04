import os, sys
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from LocalLearning import LocalLearningModel
from tqdm import tqdm

def cov_spectrum(dataloader: DataLoader, model: LocalLearningModel, device: torch.device) -> Tensor:
    '''
    Calculate the ordered spectrum of the covariance matrix
    '''
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as ttest:
            
            noE = model.pSet["hidden_size"]

            '''
            Calculate covariance matrix on shifted data to increase
            computational stability
            '''

            # initialize stat tensors
            corr = torch.zeros((noE, noE), device=device)
            mean = torch.zeros((noE, ), device=device)
            pop_size = 0

            ttest.set_description(f"Calculating covariance spectrum")
            
            # shift = estimate of the mean
            f, label = next(iter(dataloader))
            K = f.mean()
            
            for x, label in ttest:
                pop_size += len(x)
                pred = model(x.to(device))
                print(pred)
                corr += (pred.T - K) @ (pred - K)
                mean += pred.sum(axis=0) - K

            print(pop_size)
            cov = corr - mean.T @ mean / pop_size
            #cov /= pop_size - 1

            print(cov)
            #l_real = torch.linalg.eigvals(cov)
            #l_sorted, idx = torch.sort(l_real, descending=True)
            #return l_sorted



if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python train_MNIST.py <modelpath>")
        os._exit(os.EX_NOINPUT)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = Path(sys.argv[1])
    model_trained = torch.load(model_path) # load trained weights
    
    model_ps = model_trained["model_parameters"]
    model = LocalLearningModel(model_ps)
    model.load_state_dict(model_trained["model_state_dict"])
    model.eval()
    
    model.to(device)

    test_data = datasets.MNIST(
        root="../data/MNIST", train=False, download=True, transform=ToTensor()
    )

    dataloader_test = DataLoader(
            test_data, batch_size=64, num_workers=2, shuffle=False
    )

    print((model.W != model.W).sum())

    f, l = next(iter((dataloader_test)))
    print(model(f.to(device)))

    #l_spectrum = cov_spectrum(dataloader_test, model, device)
    #cpu = torch.device('cpu')
    #print(l_spectrum.to(cpu))
