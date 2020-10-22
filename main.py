from NN_models import *
from optimizers import *
import matplotlib.pyplot as plt
import sys

sys.path.append("./utils")
from import_UCI_ML_data import *
from import_mnist import *
from import_cifar10_data import *

plt.rcParams.update({'font.size': 16})

# Import data
#input_dim, output_dim, x_train, x_test, y_train, y_test = import_UCI_ML_data('graduate_admission')
#input_dim, output_dim, x_train, x_test, y_train, y_test = import_mnist_data()
input_dim, output_dim, x_train, x_test, y_train, y_test = import_cifar10_data()

print("Finished importing data")

# Setting for the neural network
num_neurons_list = [10, 10, 10, 10]
use_bias = True

# Parameters for optimizer
window_depth = 10
frequency = 5
max_iterations = 50000
learning_rate = 1e-4
threshold = 1e-7
batch_size = 100
weight_decay = 1e-5

# Define deep learning model
#model = MLPRegression(input_dim, output_dim, num_neurons_list, use_bias)
model = CNNClassification2D(input_dim, output_dim, num_neurons_list, use_bias)
model.set_coefficients_to_random()

# Set up optimizer
optimizer = FixedPointIteration(learning_rate, weight_decay)

# Import neural network in optimizer
optimizer.import_model(model)
optimizer.set_loss_function('ce')
optimizer.set_optimizer('adam')

training_loss_history = optimizer.train(x_train, y_train, max_iterations, threshold, batch_size)

epochs = range(1, len(training_loss_history) + 1)
plt.plot(epochs, training_loss_history, 'b', label='training loss')
plt.yscale('log')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.draw()
plt.savefig('loss_plot')
