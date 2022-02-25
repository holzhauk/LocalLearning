from locallearning import *

if __name__ == "__main__":
    model_ps = LocalLearningModel.pSet()
    model = LocaLearningModel(model_ps)

    training_data = datasets.MNIST(
        root="../data/MNIST", train=True, download=True, transform=ToTensor()
            )

    dataloader_train = DataLoader(training_data, batch_size=64)
    train_unsupervised(dataloader_train, model)
