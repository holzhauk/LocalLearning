import unittest
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import math

from context import LocalLearning

from sklearn.decomposition import PCA

# define identity model with parameters
class IdKH(LocalLearning.KHL3):
    def __init__(self, *args, **kwargs):
        super(IdKH, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.flatten(x)


class Test_cov_spectrum(unittest.TestCase):

    """
    Test whether this torch implementation of retrieving the PCA (covariance) 
    spectrum complies with the Scikit-learn implementation
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Test_cov_spectrum, self).__init__(*args, **kwargs)
        self.dev = torch.device("cpu")

        self.pSet = LocalLearning.KHL3.pSet
        self.pSet["in_size"] = 3 * 32 ** 2
        self.pSet["hidden_size"] = self.pSet["in_size"]
        self.id_kh = IdKH(self.pSet)

        self.test_data = LocalLearning.LpUnitCIFAR10(
            root="../data/CIFAR10", train=False, transform=ToTensor(), p=3.0
        )

    def test_spectrum_comparison(self) -> None:
        test_loader = DataLoader(
            self.test_data, batch_size=1000, num_workers=2, shuffle=False
        )
        l_n = LocalLearning.cov_spectrum(test_loader, self.id_kh, device=self.dev)
        foV = l_n / l_n.sum()  # fraction of variance

        pca_loader = DataLoader(
            self.test_data, batch_size=10000, num_workers=2, shuffle=False
        )
        big_data_chunk, _ = next(iter(pca_loader))
        pca = PCA()
        pca.fit(self.id_kh(big_data_chunk.to(self.dev)))

        self.assertTrue(
            torch.isclose(torch.Tensor(pca.explained_variance_ratio_), foV).all()
        )
        self.assertTrue(
            torch.isclose(
                torch.Tensor(pca.singular_values_),
                100 * torch.sqrt(l_n),
                atol=1e-4,
                rtol=1e-4,
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
