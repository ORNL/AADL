import sys
import torch
import numpy
import unittest

sys.path.append('../utils')
from optimizers import FixedPointIteration, DeterministicAcceleration
sys.path.append('../model_zoo')
from TestFunctions_models import Rosenbrock


def rosenbrock_regression_points(slope, intercept, n: int = 10):
    # create dummy data for training
    x_values = numpy.linspace(-10.0, 10.0, num=n)
    x_train = numpy.array(x_values, dtype=numpy.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [slope * i + intercept for i in x_values]
    y_train = numpy.array(y_values, dtype=numpy.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train



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


def test_rosenbrock_anderson(dim=2, optimizer='sgd', lr=1.e-4, epochs=10000, threshold=1.e-8, initial_guess=None):
	# dummy dataset and dataloaders
    batch_size = 1
    dataset = torch.utils.data.TensorDataset(torch.zeros(1), torch.zeros(1))
    training_dataloader   = torch.utils.data.DataLoader(dataset, batch_size)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    
    weight_decay = 0.0
    acceleration_type = "anderson"
    relaxation = 1.0
    wait_iterations = 1
    history_depth = 10
    frequency = 1
    reg_acc = 1e-9
    store_each_nth = frequency
    average = True   

    model = Rosenbrock(dim, initial_guess=initial_guess)
    optimizer_anderson = DeterministicAcceleration(training_dataloader,validation_dataloader,acceleration_type,lr,relaxation,weight_decay,wait_iterations,history_depth,
        frequency,reg_acc,store_each_nth, average)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('linear')
    optimizer_anderson.set_optimizer(optimizer)
    _, validation_anderson_loss_history, _ = optimizer_anderson.train(epochs, threshold, batch_size)

    weights = model.get_model().get_weight()

    return (weights-1.0).abs().max(), validation_anderson_loss_history



###############################################################################



class TestFunctions(unittest.TestCase):
    def test_rosenbrock_2d_sgd(self):
        lr = 1.e-4
        err, history = test_rosenbrock(dim=2, optimizer='sgd', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using  SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_lbfgs(self):
        lr = 1.e-4
        err, history = test_rosenbrock(dim=2, optimizer='lbfgs', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using  LBFGS (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "LBFGS (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_rmsprop(self):
        lr = 1.e-3
        err, history = test_rosenbrock(dim=2, optimizer='rmsprop', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_adam(self):
        lr = 1.e-1
        err, history = test_rosenbrock(dim=2, optimizer='adam', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_asgd(self):
        lr = 1.e-5
        err, history = test_rosenbrock(dim=2, optimizer='asgd', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using ASGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "ASGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

        
    #########
    
    def test_rosenbrock_2d_sgd_anderson(self):
        lr = 1.e-4
        err, history = test_rosenbrock(dim=2, optimizer='sgd', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using  SGD + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "SGD + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_rmsprop_anderson(self):
        lr = 1.e-3
        err, history = test_rosenbrock(dim=2, optimizer='rmsprop', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using RMSprop + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "RMSprop + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_adam_anderson(self):
        lr = 1.e-1
        err, history = test_rosenbrock(dim=2, optimizer='adam', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using Adam + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

        self.assertTrue(err<1e-3, "Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_2d_asgd_anderson(self):
        lr = 1.e-5
        err, history = test_rosenbrock(dim=2, optimizer='asgd', lr=lr, epochs=20000, initial_guess=[-3,-4])
        print("2d Rosenbrock using ASGD + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "ASGD + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))


    #########

    """
    
    def test_rosenbrock_10d_sgd(self):
        lr = 1.e-4
        err, history = test_rosenbrock(dim=10, optimizer='sgd', lr=lr, epochs=20000, initial_guess=[-3,-4]*5)
        print("10d Rosenbrock using  SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_10d_rmsprop(self):
        lr = 1.e-3
        err, history = test_rosenbrock(dim=10, optimizer='rmsprop', lr=lr, epochs=20000, initial_guess=[-3,-4]*5)
        print("10d Rosenbrock using RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_10d_adam(self):
        lr = 1.e-1
        err, history = test_rosenbrock(dim=10, optimizer='adam', lr=lr, epochs=20000, initial_guess=[-3,-4]*5)
        print("10d Rosenbrock using Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))


    #########
    
    def test_rosenbrock_10d_sgd_anderson(self):
        lr = 1.e-4
        err, history = test_rosenbrock(dim=10, optimizer='sgd', lr=lr, epochs=20000, initial_guess=[-3,-4]*5)
        print("10d Rosenbrock using  SGD + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "SGD (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_10d_rmsprop_anderson(self):
        lr = 1.e-3
        err, history = test_rosenbrock(dim=10, optimizer='rmsprop', lr=lr, epochs=20000, initial_guess=[-3,-4]*5)
        print("10d Rosenbrock using RMSprop + Andserson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "RMSprop (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))

    def test_rosenbrock_10d_adam_anderson(self):
        lr = 1.e-1
        err, history = test_rosenbrock(dim=10, optimizer='adam', lr=lr, epochs=20000, initial_guess=[-3,-4]*5)
        print("10d Rosenbrock using Adam + Anderson (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
        self.assertTrue(err<1e-3, "Adam (lr=%.1e) finished after %5d iterations with error max_i||w_i-1.0||=%.2e"%(lr,len(history),err.item()))
    
    """

###############################################################################


if __name__ == "__main__":
    unittest.main()
