import sys
import numpy
import torch
from torch.utils.data import Dataset
import unittest

sys.path.append('../modules')
from NN_models import MLP
from optimizers import FixedPointIteration, RNA_Acceleration


def linear_regression(slope, intercept, n: int = 10):
    # create dummy data for training
    x_values = numpy.linspace(-10.0, 10.0, num=n)
    x_train = numpy.array(x_values, dtype=numpy.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [slope * i + intercept for i in x_values]
    y_train = numpy.array(y_values, dtype=numpy.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


def monotonic_decreasing(x):
    dx = numpy.diff(x)
    return numpy.all(dx < 0)


class LinearData(Dataset):
    def __init__(self, slope, intercept, num_points: int = 10):
        super(LinearData, self).__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.slope = slope
        self.intercept = intercept        
        self.num_points = num_points
        
        x_sample, y_sample = linear_regression(self.slope, self.intercept, self.num_points)

        self.x_sample = x_sample
        self.y_values = y_sample
        self.y_values = numpy.reshape(self.y_values, (len(self.y_values), 1))
        

    def __len__(self):
        return self.y_values.shape[0]

    def __getitem__(self, index):
        x_sample = self.x_sample[index, :]

        y_sample = self.y_values[index]

        # Doubles must be converted to Floats before passing them to a neural network model
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.from_numpy(y_sample).float()

        return x_sample, y_sample


def linear_data(slope, intercept, num_points: int = 10):
    input_dim = 1
    output_dim = 1
    return input_dim, output_dim, LinearData(slope, intercept, num_points = num_points)


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, use_bias: bool=True, device='cpu'):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, use_bias)

        self.model = torch.nn.Sequential(self.linear)

        self.device = torch.device(device)

    def forward(self, x):
        out = self.model(x)
        return out

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device


def test_linear_regression_sgd(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-3
    weight_decay = 0.0    
    batch_size = 1
    epochs = 100000
    threshold = 1e-8

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_classic = FixedPointIteration(dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer('sgd')
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)
    
    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), training_classic_loss_history

def test_linear_regression_rmsprop(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-3
    weight_decay = 0.0    
    batch_size = 1
    epochs = 100000
    threshold = 1e-8

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_classic = FixedPointIteration(dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer('rmsprop')
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), training_classic_loss_history


def test_linear_regression_adam(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-3
    weight_decay = 0.0    
    batch_size = 1
    epochs = 100000
    threshold = 1e-8

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_classic = FixedPointIteration(dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer('adam')
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), training_classic_loss_history



def test_linear_regression_sgd_anderson(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-6
    weight_decay = 0.0    
    batch_size = 1
    epochs = 10000
    threshold = 1e-8
    wait_iterations = 1
    window_depth = epochs
    frequency = 1
    reg_acc = 0.0
    store_each = 1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_anderson = RNA_Acceleration(dataloader, learning_rate, weight_decay, wait_iterations, window_depth,
                                          frequency,
                                          reg_acc, store_each, True)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer('sgd')
    training_classic_loss_history = optimizer_anderson.train(epochs, threshold, batch_size)

    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), training_classic_loss_history



def test_linear_regression_rmsprop_anderson(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1.0
    weight_decay = 0.0    
    batch_size = 1
    epochs = 1000
    threshold = 1e-8
    wait_iterations = 1
    window_depth = epochs
    frequency = 1
    reg_acc = 0.0
    store_each = 1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_anderson = RNA_Acceleration(dataloader, learning_rate, weight_decay, wait_iterations, window_depth,
                                          frequency,
                                          reg_acc, store_each)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer('rmsprop')
    training_classic_loss_history = optimizer_anderson.train(epochs, threshold, batch_size)

    return training_classic_loss_history


def test_linear_regression_adam_anderson(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-2
    weight_decay = 0.0    
    batch_size = 1
    epochs = 1000
    threshold = 1e-8
    wait_iterations = 1
    window_depth = epochs
    frequency = 1
    reg_acc = 0.0
    store_each = 1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_anderson = RNA_Acceleration(dataloader, learning_rate, weight_decay, wait_iterations, window_depth,
                                          frequency,
                                          reg_acc, store_each)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer('adam')
    training_classic_loss_history = optimizer_anderson.train(epochs, threshold, batch_size)

    return training_classic_loss_history


def test_neural_network_linear_regression_sgd(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    num_neurons_list = [1]
    use_bias = True
    classification_problem = False
    activation = None
    weight_decay = 0.0
    learning_rate = 1e-2
    batch_size = 1
    epochs = 1000
    threshold = 1e-8

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = MLP(input_dim, output_dim, num_neurons_list, use_bias, activation, classification_problem)

    optimizer_classic = FixedPointIteration(dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer('sgd')
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)

    return training_classic_loss_history


def test_neural_network_linear_regression_adam(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    num_neurons_list = [1]
    use_bias = True
    classification_problem = False
    activation = None
    weight_decay = 0.0
    learning_rate = 1e-2
    batch_size = 1
    epochs = 1000
    threshold = 1e-8

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = MLP(input_dim, output_dim, num_neurons_list, use_bias, activation, classification_problem)

    optimizer_classic = FixedPointIteration(dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer('adam')
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)

    return training_classic_loss_history


def test_neural_network_linear_regression_sgd_anderson(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    num_neurons_list = [1]
    use_bias = True
    classification_problem = False
    activation = None
    weight_decay = 0.0
    learning_rate = 1.0
    batch_size = 1
    epochs = 10
    threshold = 1e-8
    wait_iterations = 1
    window_depth = epochs
    frequency = 1
    reg_acc = 0.0
    store_each = 1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = MLP(input_dim, output_dim, num_neurons_list, use_bias, activation, classification_problem)

    optimizer_anderson = RNA_Acceleration(dataloader, learning_rate, weight_decay, wait_iterations, window_depth,
                                          frequency,
                                          reg_acc, store_each)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer('sgd')
    training_classic_loss_history = optimizer_anderson.train(epochs, threshold, batch_size)

    return training_classic_loss_history


def test_neural_network_linear_regression_adam_anderson(slope, intercept, num_points):
    input_dim, output_dim, dataset = linear_data(slope, intercept, num_points)
    num_neurons_list = [1]
    use_bias = True
    classification_problem = False
    activation = None
    weight_decay = 0.0
    learning_rate = 1e-2
    batch_size = 1
    epochs = 1000
    threshold = 1e-8
    wait_iterations = 1
    window_depth = epochs
    frequency = 1
    reg_acc = 0.0
    store_each = 1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = MLP(input_dim, output_dim, num_neurons_list, use_bias, activation, classification_problem)

    optimizer_anderson = RNA_Acceleration(dataloader, learning_rate, weight_decay, wait_iterations, window_depth,
                                          frequency,
                                          reg_acc, store_each, True)
    optimizer_anderson.import_model(model)
    optimizer_anderson.set_loss_function('mse')
    optimizer_anderson.set_optimizer('adam')
    training_classic_loss_history = optimizer_anderson.train(epochs, threshold, batch_size)

    return training_classic_loss_history


class TestRegression(unittest.TestCase):

    def test_sgd(self):
        num_points = 2
        straight_line_parameters = torch.rand(2, 1)
        slope = straight_line_parameters[0].item()
        intercept = straight_line_parameters[1].item()
        numeric_slope, numeric_intercept, history = test_linear_regression_sgd(slope, intercept, num_points)
        print("SGD converged in "+str(len(history))+" iterations "+"\n exact slope: "+str(slope)+"  - "+" numerical slope: "+str(numeric_slope)+"\n"+" -   exact intercept: "+str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
        self.assertTrue(abs((slope-numeric_slope))<1e-3 and abs((intercept-numeric_intercept))<1e-3)
    
    def test_rmsprop(self):
        num_points = 2
        straight_line_parameters = torch.rand(2, 1)
        slope = straight_line_parameters[0].item()
        intercept = straight_line_parameters[1].item()
        numeric_slope, numeric_intercept, history = test_linear_regression_rmsprop(slope, intercept, num_points)
        print("RMSProp converged in "+str(len(history))+" iterations "+"\n exact slope: "+str(slope)+"  - "+" numerical slope: "+str(numeric_slope)+"\n"+" -   exact intercept: "+str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
        self.assertTrue(abs((slope-numeric_slope))<1e-3 and abs((intercept-numeric_intercept))<1e-3)
    
    def test_adam(self):
        num_points = 2
        straight_line_parameters = torch.rand(2, 1)
        slope = straight_line_parameters[0].item()
        intercept = straight_line_parameters[1].item()
        numeric_slope, numeric_intercept, history = test_linear_regression_adam(slope, intercept, num_points)
        print("Adam converged in "+str(len(history))+" iterations "+"\n exact slope: "+str(slope)+"  - "+" numerical slope: "+str(numeric_slope)+"\n"+" -   exact intercept: "+str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
        self.assertTrue(abs((slope-numeric_slope))<1e-3 and abs((intercept-numeric_intercept))<1e-3)
   
    """    
    def test_sgd_anderson(self):
        num_points = 2
        straight_line_parameters = torch.rand(2, 1)
        slope = straight_line_parameters[0].item()
        intercept = straight_line_parameters[1].item()
        numeric_slope, numeric_intercept, history = test_linear_regression_sgd_anderson(slope, intercept, num_points)
        self.assertTrue(abs((slope-numeric_slope))<1e-3 and abs((intercept-numeric_intercept))<1e-3)
      
    def test_rmsprop_anderson(self):
        self.assertTrue(monotonic_decreasing(test_linear_regression_rmsprop_anderson(1)))
        
    def test_adam_anderson(self):
        self.assertTrue(monotonic_decreasing(test_linear_regression_adam_anderson(2)))

    def test_nn_sgd(self):
        self.assertTrue(monotonic_decreasing(test_neural_network_linear_regression_sgd(10000)))

    def test_nn_adam(self):
        self.assertTrue(monotonic_decreasing(test_neural_network_linear_regression_adam(10000)))

    def test_nn_sgd_anderson(self):
        self.assertTrue(monotonic_decreasing(test_neural_network_linear_regression_sgd_anderson(10000)))

    def test_nn_adam_anderson(self):
        self.assertTrue(monotonic_decreasing(test_neural_network_linear_regression_adam_anderson(10000)))
    """
        

if __name__ == "__main__":
    unittest.main()
