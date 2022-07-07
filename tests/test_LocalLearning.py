import unittest
import torch
import math
from context import LocalLearning
from torch import Tensor

ABS_PRECISION = 1e-6
REL_PRECISION = 1e-5

test_pSet = LocalLearning.KHL3.pSet
test_pSet["in_size"] = 3 * 3 ** 2
test_pSet["hidden_size"] = 10
test_pSet["p"] = 3
test_pSet["tau_l"] = 10.0
test_pSet["k"] = 7
test_pSet["Delta"] = 0.4
test_pSet["R"] = 1.0

BATCH_SIZE = 2


class KrotovHopfieldImplementation:
    """
    Fast AI implementation of Krotov and Hopfield as featured in
        * https://github.com/DimaKrotov/Biological_Learning/blob/master
            /Unsupervised_learning_algorithm_MNIST.ipynb
    """

    def __init__(self, param: dict, batch_size: int, device: torch.device) -> None:
        self.inp = param["in_size"]
        self.hid = param["hidden_size"]
        self.mu = 0.0
        self.sigma = 1.0
        self.delta = param["Delta"]
        self.p = param["p"]
        self.k = param["k"]
        self.eps = 1.0 / param["tau_l"]
        self.batch_size = batch_size

        self.dev = device

    def synaptic_activation(self, synapses: Tensor, v: Tensor) -> Tensor:
        return (synapses.sign() * synapses.abs() ** (self.p - 1.0)) @ v

    def learning_activation(self, indices: Tensor) -> Tensor:
        best_ind, best_k_ind = indices[0], indices[self.k - 1]
        g_i = torch.zeros(self.hid, self.batch_size).to(self.dev)
        g_i[best_ind, torch.arange(self.batch_size).to(self.dev)] = 1.0
        g_i[best_k_ind, torch.arange(self.batch_size).to(self.dev)] = self.delta
        return g_i


class TestKHL3(unittest.TestCase):

    """
    Test whether expressions were vectorised correctly
    test against index formulated problem
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TestKHL3, self).__init__(*args, **kwargs)
        self.model = LocalLearning.KHL3(test_pSet)
        self.x = torch.rand((BATCH_SIZE, 3, 3, 3), requires_grad=False)
        std = 1.0 / math.sqrt(test_pSet["in_size"] + test_pSet["hidden_size"])
        self.W_standard = torch.normal(
            mean=0.0,
            std=std,
            size=(test_pSet["in_size"], test_pSet["hidden_size"]),
            requires_grad=False,
        )

    def test_forward(self) -> None:
        model = self.model
        flatten = torch.nn.Flatten()
        x_flat = flatten(self.x)
        model.W = torch.nn.Parameter(self.W_standard.clone(), requires_grad=False)
        mini_batch_size = len(x_flat)

        # Index forward
        truth = torch.zeros(
            mini_batch_size, test_pSet["hidden_size"], requires_grad=False
        )
        for mini_batch_idx in range(mini_batch_size):
            for mu in range(test_pSet["hidden_size"]):
                for i in range(test_pSet["in_size"]):
                    truth[mini_batch_idx, mu] += (
                        model.W[i, mu]
                        * torch.pow(torch.abs(model.W[i, mu]), test_pSet["p"] - 2.0)
                        * x_flat[mini_batch_idx, i]
                    )

        val = model(self.x)
        self.assertTrue(
            torch.isclose(val, truth, atol=ABS_PRECISION, rtol=REL_PRECISION).all()
        )

    def test_train(self) -> None:
        model = self.model
        flatten = torch.nn.Flatten()
        x_flat = flatten(self.x)
        model.W = torch.nn.Parameter(self.W_standard.clone(), requires_grad=False)
        W = self.W_standard.clone()

        # index train
        mini_batch_size = len(x_flat)
        for x_mb in x_flat:
            Q = torch.zeros((test_pSet["hidden_size"],), requires_grad=False)
            for mu in range(test_pSet["hidden_size"]):
                bracket_Wx_mu = 0.0
                bracket_WW_mu = 0.0
                for i in range(test_pSet["in_size"]):
                    bracket_Wx_mu += (
                        W[i, mu]
                        * torch.pow(torch.abs(W[i, mu]), test_pSet["p"] - 2)
                        * x_mb[i]
                    )
                    bracket_WW_mu += (
                        W[i, mu]
                        * torch.pow(torch.abs(W[i, mu]), test_pSet["p"] - 2)
                        * W[i, mu]
                    )

                Q[mu] = bracket_Wx_mu / bracket_WW_mu ** (
                    (test_pSet["p"] - 1) / test_pSet["p"]
                )

            g = torch.zeros(Q.size())
            sorted_idxs = Q.argsort(descending=True)
            g[sorted_idxs[0]] = 1.0
            g[sorted_idxs[1 : test_pSet["k"]]] = -test_pSet["Delta"]

            for mu in range(test_pSet["hidden_size"]):
                bracket_Wx_mu = 0.0
                for i in range(test_pSet["in_size"]):
                    bracket_Wx_mu += (
                        W[i, mu]
                        * torch.pow(torch.abs(W[i, mu]), test_pSet["p"] - 2)
                        * x_mb[i]
                    )

                for i in range(test_pSet["in_size"]):
                    W[i, mu] += (
                        g[mu]
                        * (
                            (test_pSet["R"] ** test_pSet["p"]) * x_mb[i]
                            - bracket_Wx_mu * W[i, mu]
                        )
                        / test_pSet["tau_l"]
                    )

        model.train_step(self.x)
        self.assertTrue(
            torch.isclose(model.W, W, atol=ABS_PRECISION, rtol=REL_PRECISION).all()
        )


class TestFKHL3(unittest.TestCase):
    """
    Test the torch Fast AI implementation against the one by Krotov and Hopfield
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TestFKHL3, self).__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = LocalLearning.FKHL3(test_pSet)
        self.model.to(self.device)
        self.model.train()
        self.KHmodel = KrotovHopfieldImplementation(test_pSet, BATCH_SIZE, self.device)
        self.flat = torch.nn.Flatten()

    def test_activations(self) -> None:
        """
        Compare basic forwarding with the Krotov and Hopfield model
        """
        # define test batch
        inp = torch.rand(BATCH_SIZE, 3, 3, 3)

        # KH implementation
        v = self.flat(inp).to(self.device)
        synapses = (
            torch.Tensor(test_pSet["hidden_size"], test_pSet["in_size"])
            .normal_(self.KHmodel.mu, self.KHmodel.sigma)
            .to(self.device)
        )
        a_KH = self.KHmodel.synaptic_activation(synapses, v.T)

        # assign the same weight initialization to the FKHL3 implementation
        self.model.W = torch.nn.Parameter(synapses.T, requires_grad=False)

        self.assertTrue(torch.equal(self.model(inp.to(self.device)), a_KH.T))

    def test_weight_increments(self) -> None:
        """
        Compare weight increments used for unsupervised learning
        """

        # define test batch and weight matrix
        prec = 1e-30  # define precision value of weight update
        inp = torch.rand(BATCH_SIZE, 3, 3, 3)
        W = (
            torch.Tensor(test_pSet["hidden_size"], test_pSet["in_size"])
            .normal_(self.KHmodel.mu, self.KHmodel.sigma)
            .to(self.device)
        )

        # KH implementation
        v = self.flat(inp).to(self.device)
        synapses = W.clone()
        a_KH = self.KHmodel.synaptic_activation(synapses, v.T)
        _, indices = a_KH.topk(
            self.KHmodel.k, dim=0
        )  # find indices maximizing the synapse
        g_i = self.KHmodel.learning_activation(indices)
        # calculate weight increment
        xx = (g_i * a_KH).sum(dim=1)
        ds = g_i @ v - xx.unsqueeze(1) * synapses
        # weight update
        nc = max(ds.abs().max(), prec)
        synapses += self.KHmodel.eps * ds / nc

        # FKHL3 implementation
        self.model.W = torch.nn.Parameter(synapses.T, requires_grad=False)
        self.model.train_step_fast(inp.to(self.device), prec=prec)

        self.assertTrue(
            torch.isclose(
                self.model.W, synapses.T, atol=ABS_PRECISION, rtol=REL_PRECISION
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
