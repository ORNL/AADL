import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy


def import_UCI_ML_data(data_name):
    # Import data as pandas data-frame
    if data_name == 'graduate_admission':
        df = pandas.read_csv('./datasets/graduate_admission.csv')

    # Convert a pandas data-frame into a numpy array
    df_array = df.values

    x_sample = df_array[:, :-1]
    x_sample = x_sample[:, 1:]
    y_values = df_array[:, -1]
    y_values = numpy.reshape(y_values, (len(y_values), 1))

    # Extract important features of the data
    input_dim = x_sample.shape[1]
    output_dim = y_values.shape[1]

    # We split the sample points between training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x_sample, y_values, train_size=0.75, random_state=42)

    # We scale the data
    scaler = StandardScaler()

    # The scaling process is based on the training set
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Doubles must be converted to Floats before passing them to a neural network model
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    return input_dim, output_dim, x_train, x_test, y_train, y_test
