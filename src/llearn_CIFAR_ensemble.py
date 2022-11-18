import os, sys
from pathlib import Path
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from LocalLearning import FKHL3
from LocalLearning import train_unsupervised
from LocalLearning import LpUnitCIFAR10
from LocalLearning import MikkelCIFAR10
from LocalLearning import DeviceDataLoader

# Model parameters

model_ps = {
    "in_size": 3 * 32 ** 2,
    "hidden_size": 2000,
    "n": 4.5,
    "p": 3.0,
    "tau_l": 1.0 / 0.02,  # 1 / learning rate
    "k": 2,
    "Delta": 0.4,  # inhibition rate
    "R": 1.0,  # asymptotic weight norm radius
}

# Unsupervised Training Hyperparameters
NO_EPOCHS = 1000
BATCH_SIZE = 1000
ENSEMBLE_SIZE = 100

def add_AddOn(path: Path, add_on: Path) -> Path:
    filename = Path(str(path.stem) + "_" + str(add_on) + str(path.suffix))
    return path.parent / filename

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

    training_data = LpUnitCIFAR10(
            root="data/CIFAR10", train=True, transform=ToTensor(), p=model_ps["p"]
    )

    for e_idx  in range(ENSEMBLE_SIZE + 1):

        ensemble_id = Path(f"E{e_idx + 1}")
        full_path = add_AddOn(model_path, ensemble_id)

        model = FKHL3(model_ps, sigma=1.0)
        model.to(device=device) 

        dataloader_train = DeviceDataLoader(
                training_data, device=device, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
        )

        lr = 1.0 / model_ps["tau_l"] 
        def learning_rate(epoch: int) -> float:
            return (1.0 - epoch / NO_EPOCHS) * lr

        train_unsupervised(
            dataloader_train,
            model,
            device,
            full_path,
            no_epochs=NO_EPOCHS,
            checkpt_period=NO_EPOCHS,
            learning_rate=learning_rate,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_parameters": model.param_dict(),
                "device_type": device.type,
            },
            model_path,
        )
