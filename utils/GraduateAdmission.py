import pandas
import torch
import numpy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GraduateAdmission(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super(GraduateAdmission, self).__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Import data as pandas data-frame
        df = pandas.read_csv(root_dir + csv_file)

        # Convert a pandas data-frame into a numpy array
        df_array = df.values

        self.x_sample = df_array[:, :-1]
        self.x_sample = self.x_sample[:, 1:]
        self.y_values = df_array[:, -1]
        self.y_values = numpy.reshape(self.y_values, (len(self.y_values), 1))

        # We split the sample points between training set and test set
        x_train, x_test, y_train, y_test = train_test_split(self.x_sample, self.y_values, train_size=0.75,
                                                            random_state=42)

        # We scale the data
        self.transform = StandardScaler()

        # The scaling process is based on the training set
        self.transform.fit(x_train)

    def __len__(self):
        return self.y_values.shape[0]

    def __getitem__(self, index):
        x_sample = self.x_sample[index, :]
        x_sample = x_sample.reshape(1, -1)

        if self.transform:
            x_sample = self.transform.transform(x_sample)

        y_sample = self.y_values[index]

        # Doubles must be converted to Floats before passing them to a neural network model
        x_sample = torch.from_numpy(x_sample).float()
        y_sample = torch.from_numpy(y_sample).float()

        return x_sample, y_sample
