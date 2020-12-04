import sys
import numpy
import torch
from torch.utils.data import Dataset
import unittest

sys.path.append('../modules')
from NN_models import MLP
from optimizers import FixedPointIteration, RNA_Acceleration


###############################################################################


class Paraboloid(torch.nn.Module):
    def __init__(self, dim, condition_number=1.e3, device='cpu'):
        super().__init__()

        self.device = device

        # A is orthogonal matrix
        A, _ = torch.qr( torch.rand(dim,dim) )
        for i in range(dim):
            A[i,:] *= condition_number**(i/dim)
        self.register_buffer('A', A)

        self.weight = torch.nn.Parameter(torch.rand(dim))

    def forward(self, x):
        return torch.mv(self.A, self.weight)

    def get_model(self):
        return self

    def get_weight(self):
        return self.weight

    def get_device(self):
        return self.device


def test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='sgd', lr=1.e-4, w_decay=0.0, epochs=10000, threshold=1.e-8):
    batch_size = dim

    dataset    = torch.utils.data.TensorDataset(torch.zeros(dim), torch.zeros(dim))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = Paraboloid(dim, condition_number=condition_number)
    optimizer_classic = FixedPointIteration(dataloader, lr, w_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer(optimizer)
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)

    weights = model.get_model().get_weight()

    return weights.abs().max(), training_classic_loss_history


###############################################################################



class TestMultiscaleParaboloid(unittest.TestCase):

    def test_2d_well_conditioned_paraboloid_sgd(self):
        weight_sum, history = test_multiscale_paraboloid(dim=2, condition_number=1, optimizer='sgd', lr=1.e-3, epochs=10000)
        print("2d SGD finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_2d_well_conditioned_paraboloid_rmsprop(self):
        weight_sum, history = test_multiscale_paraboloid(dim=2, condition_number=1, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("2d RMSProp finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_2d_well_conditioned_paraboloid_adam(self):
        weight_sum, history = test_multiscale_paraboloid(dim=2, condition_number=1, optimizer='adam', lr=1.e-3, epochs=10000)
        print("2d Adam finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    ###########

    def test_100d_well_conditioned_paraboloid_sgd(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1, optimizer='sgd', lr=1.e-3, epochs=10000)
        print("100d SGD finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_well_conditioned_paraboloid_rmsprop(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("100d RMSProp finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_well_conditioned_paraboloid_adam(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1, optimizer='adam', lr=1.e-3, epochs=10000)
        print("100d Adam finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    ###########

    def test_100d_ill_conditioned_paraboloid_sgd(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='sgd', lr=1.e-3, epochs=10000)
        print("100d SGD finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_ill_conditioned_paraboloid_rmsprop(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='rmsprop', lr=1.e-3, epochs=10000)
        print("100d RMSProp finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)

    def test_100d_ill_conditioned_paraboloid_adam(self):
        weight_sum, history = test_multiscale_paraboloid(dim=100, condition_number=1.e3, optimizer='adam', lr=1.e-3, epochs=10000)
        print("100d Adam finished after "+str(len(history))+" iterations "+"\n exact weight sum: 0"+"  - "+" numerical weight sum: "+str(weight_sum))
        self.assertTrue(weight_sum<1e-3)


###############################################################################




if __name__ == "__main__":
    unittest.main()
