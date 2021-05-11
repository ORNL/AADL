import sys
import torch
import unittest

sys.path.append('../utils')
from optimizers import FixedPointIteration, DeterministicAcceleration
sys.path.append('../model_zoo')
from TestFunctions_models import Rosenbrock



###############################################################################


def test_rosenbrock(dim=2, optimizer='sgd', lr=1.e-4, epochs=10000, threshold=1.e-8, initial_guess=None):
	# dummy dataset and dataloaders
    batch_size = 1
    dataset = torch.utils.data.TensorDataset(torch.zeros(1), torch.zeros(1))
    training_dataloader   = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    w_decay = 0.0

    model = Rosenbrock(dim, initial_guess=initial_guess)
    optimizer_classic = FixedPointIteration(training_dataloader, validation_dataloader, lr, w_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('linear')
    optimizer_classic.set_optimizer(optimizer)
    _, validation_classic_loss_history, _ = optimizer_classic.train(epochs, threshold, batch_size)

    weights = model.get_model().get_weight()

    return (weights-1.0).abs().max(), validation_classic_loss_history



###############################################################################



class TestFunctions(unittest.TestCase):
    def test_rosenbrock_2d_sgd(self):
        lr = 1.e-4
        err, history = test_rosenbrock(dim=2, optimizer='sgd', lr=lr, epochs=10000, initial_guess=[-3,-4])
        # print("2d Rosenbrock using  SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_rmsprop(self):
        lr = 1.e-3
        err, history = test_rosenbrock(dim=2, optimizer='rmsprop', lr=lr, epochs=10000, initial_guess=[-3,-4])
        # print("2d Rosenbrock using RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_adam(self):
        lr = 1.e-1
        err, history = test_rosenbrock(dim=2, optimizer='adam', lr=lr, epochs=10000, initial_guess=[-3,-4])
        # print("2d Rosenbrock using Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    #########

    def test_rosenbrock_10d_sgd(self):
        lr = 1.e-4
        err, history = test_rosenbrock(dim=10, optimizer='sgd', lr=lr, epochs=10000, initial_guess=[-3,-4]*5)
        # print("2d Rosenbrock using  SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_10d_rmsprop(self):
        lr = 1.e-3
        err, history = test_rosenbrock(dim=10, optimizer='rmsprop', lr=lr, epochs=10000, initial_guess=[-3,-4]*5)
        # print("2d Rosenbrock using RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_10d_adam(self):
        lr = 1.e-1
        err, history = test_rosenbrock(dim=10, optimizer='adam', lr=lr, epochs=10000, initial_guess=[-3,-4]*5)
        # print("2d Rosenbrock using Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))


###############################################################################


if __name__ == "__main__":
    unittest.main()
