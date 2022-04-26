import unittest
import torch
from context import LocalLearning

PRECISION = 10**-6

test_pSet = LocalLearning.LocalLearningModel.pSet
test_pSet["in_size"] = 2**2
test_pSet["hidden_size"] = 5
test_pSet["p"] = 3
test_pSet["tau_l"] = 10.0
test_pSet["k"] = 7
test_pSet["Delta"] = 0.4
test_pSet["R"] = 1.0

llmodel = LocalLearning.LocalLearningModel(test_pSet)
with torch.no_grad():
    W_standard = torch.tensor(
        [
            [-0.3783, -0.1536, -0.0994, 0.0785, -0.0857],
            [-0.1967, -0.5230, -0.0562, 0.4111, 0.1823],
            [-0.0523, -0.0145, 0.0101, -0.2267, -0.8322],
            [-0.4799, 0.2755, 0.3235, -0.2454, 0.0748],
        ],
        requires_grad=False,
    )
    x = torch.tensor(
        [
            [[[0.3245, 0.6256], [0.7344, 0.7498]]],
            [[[0.1765, 0.4858], [0.8896, 0.9979]]],
        ],
        requires_grad=False,
    )


class TestLocalLearningModel(unittest.TestCase):

    """
    Test whether expressions were vectorised correctly
    test against index formulated problem
    """

    def test_forward(self):
        flatten = torch.nn.Flatten()
        x_flat = flatten(x)
        llmodel.W = torch.nn.Parameter(W_standard.clone(), requires_grad=False)
        mini_batch_size = len(x_flat)

        # Index forward
        truth = torch.zeros(
            mini_batch_size, test_pSet["hidden_size"], requires_grad=False
        )
        for mini_batch_idx in range(mini_batch_size):
            for mu in range(test_pSet["hidden_size"]):
                for i in range(test_pSet["in_size"]):
                    truth[mini_batch_idx, mu] += (
                        llmodel.W[i, mu]
                        * torch.pow(torch.abs(llmodel.W[i, mu]), test_pSet["p"] - 2.0)
                        * x_flat[mini_batch_idx, i]
                    )

        val = llmodel(x)
        self.assertTrue(torch.norm(val - truth) < PRECISION)

    def test_train(self):
        flatten = torch.nn.Flatten()
        x_flat = flatten(x)
        llmodel.W = torch.nn.Parameter(W_standard.clone(), requires_grad=False)
        W = W_standard.clone()

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
            g[sorted_idxs[1 : test_pSet["k"] + 1]] = -test_pSet["Delta"]

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

        llmodel.train_step(x)
        self.assertTrue(torch.norm(llmodel.W - W) < PRECISION)


if __name__ == "__main__":
    unittest.main()
