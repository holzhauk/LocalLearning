import os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from LocalLearning import FKHL3
from LocalLearning import train_unsupervised
from LocalLearning import weight_convergence_criterion
from LocalLearning import weight_mean_criterion

model_ps = {
    "in_size": 28 ** 2,  # MNIST dataset consists of 28x28 pixel imgs
    "hidden_size": 2000,
    "n": 4.5,
    "p": 3.0,
    "tau_l": 1.0 / 0.04,  # 1 / learning rate
    "k": 7,
    "Delta": 0.4,  # inhibition rate
    "R": 1.0,  # asymptotic weight norm radius
}

# Unsupervised Training Hyperparameters
NO_EPOCHS = 1000
BATCH_SIZE = 100

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python train_MNIST.py <modelpath>")
        os._exit(os.EX_NOINPUT)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = Path(sys.argv[1])
    if not os.path.exists(model_path.parent):
        os.makedirs(model_path.parent)

    # explicitely stating initialization with variance 1.0
    # Glorot Initialisation leads to highly stable steady state 
    # with exclusively negative weights
    model = FKHL3(model_ps, sigma=1.0)
    model.to(device=device)

    training_data = datasets.MNIST(
        root="../data/MNIST", train=True, download=True, transform=ToTensor()
    )

    dataloader_train = DataLoader(
            training_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
    )
    
    lr = 1.0 / model_ps["tau_l"]

    def learning_rate(epoch: int) -> float:
        # Learning rate linearly decreases from 0.04
        # to 0.0 at last epoch
        return (1.0 - epoch / NO_EPOCHS) * lr

    train_unsupervised(
            dataloader_train, 
            model, 
            device, 
            model_path, 
            no_epochs=NO_EPOCHS, 
            # without "-1", lr = 0 leading to division by zero                         
            # otherwise, lr=0 corresponds to no learning at all
            checkpt_period=5,
            learning_rate=learning_rate,
            )
    
    # check convergence criteria
    # weights converge towards 1.0
    if not weight_convergence_criterion(model, 1e-2, 1e-1):
        print("Less than 10pc of weights converged close enough. Model not saved. Try running again.")
        os._exit(os.EX_OK)

    if not weight_mean_criterion(model):
        print("Weights converged to the wrong attractor. Model not saved. Try running again.")
        os._exit(os.EX_OK)  

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_parameters": model.param_dict(),
            "device_type": device.type,
        },
        model_path,
    )
