import pandas
import sys
import numpy
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NN_models import *
from optimizers import *
from torch.autograd import Variable
import matplotlib.pyplot as plt

sys.path.append("./utils")

plt.rcParams.update({'font.size': 16})

# Setting for the neural network
num_neurons = 10
num_hidden_layers = 10
use_bias = True

# Parameters for optimizer
window_depth = 10
frequency = 5
max_iterations = 50000
learning_rate = 1e-4
threshold = 1e-7
batch_size = 100
weight_decay = 1e-5

# Import data as pandas data-frame
df = pandas.read_csv('./datasets/graduate_admission.csv')

# Convert a pandas data-frame into a numpy array
df_array = df.values

X_sample = df_array[:, :-1]
X_sample = X_sample[:, 1:]
y_values = df_array[:, -1]
y_values = numpy.reshape(y_values, (len(y_values), 1))

# Extract important features of the data
input_dim = X_sample.shape[1]
output_dim = y_values.shape[1]

# We split the sample points between training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_values, train_size=0.9, random_state=42)

# We scale the data
scaler = StandardScaler()

# The scaling process is based on the training set
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Doubles must be converted to Floats before passing them to a neural network model
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()


model = MultiLayerPerceptionRegression(input_dim, output_dim, num_neurons, num_hidden_layers, use_bias)
#model.set_coefficients_to_zero()

# Set up optimizer
accelerated_optimizer = Anderson(window_depth, frequency, learning_rate, weight_decay)

# Import neural network in optimizer
accelerated_optimizer.import_model(model)
accelerated_optimizer.set_loss_function('mse')
accelerated_optimizer.set_optimizer('adam')

training_loss_history = accelerated_optimizer.accelerated_train(X_train, y_train, max_iterations, threshold, batch_size)

epochs = range(1,len(training_loss_history)+1)
plt.plot(epochs, training_loss_history, 'b', label = 'training loss')
plt.yscale('log')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.draw()
plt.savefig('loss_plot')        

