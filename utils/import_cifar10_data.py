import pickle
import numpy
import torch


# import and reshape cifar-10 data

def load():
    path = './datasets/cifar-10/'
    file = 'data_batch_'

    imagearray = numpy.zeros((0, 3072))
    labelarray = numpy.zeros(0)

    for num_file in range(1, 6):
        with open(path + file + str(num_file), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        images = dict[b'data']
        # images = np.reshape(images, (10000, 3, 32, 32))
        labels = dict[b'labels']
        imagearray_tmp = numpy.array(images)  # (10000, 3072)
        labelarray_tmp = numpy.array(labels)  # (10000,)

        imagearray = numpy.concatenate((imagearray, imagearray_tmp))
        labelarray = numpy.concatenate((labelarray, labelarray_tmp))

    with open(path + 'test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    images = dict[b'data']
    # images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict[b'labels']
    imagearray_tmp = numpy.array(images)  # (10000, 3072)
    labelarray_tmp = numpy.array(labels)  # (10000,)

    imagearray = numpy.concatenate((imagearray, imagearray_tmp))
    labelarray = numpy.concatenate((labelarray, labelarray_tmp))

    labelarray = numpy.reshape(labelarray, (labelarray.shape[0], 1))

    return imagearray, labelarray


def rearrange_images(images_old, num_channels):
    images_new = numpy.zeros((images_old.shape[0], num_channels, int(numpy.sqrt(images_old.shape[1] / num_channels)),
                              int(numpy.sqrt(images_old.shape[1] / num_channels))))

    for row in range(0, images_old.shape[0]):
        images_new[row, 0, :, :] = numpy.reshape(images_old[row, 0:int(images_old.shape[1] / num_channels)], (
            int(numpy.sqrt((int(images_old.shape[1] / num_channels)))),
            int(numpy.sqrt(int(images_old.shape[1] / num_channels)))))
        images_new[row, 1, :, :] = numpy.reshape(
            images_old[row, int(images_old.shape[1] / num_channels):2 * int(images_old.shape[1] / num_channels)], (
                int(numpy.sqrt((int(images_old.shape[1] / num_channels)))),
                int(numpy.sqrt(int(images_old.shape[1] / num_channels)))))
        images_new[row, 2, :, :] = numpy.reshape(
            images_old[row, 2 * int(images_old.shape[1] / num_channels):3 * int(images_old.shape[1] / num_channels)], (
                int(numpy.sqrt((int(images_old.shape[1] / num_channels)))),
                int(numpy.sqrt(int(images_old.shape[1] / num_channels)))))

    return images_new


def import_cifar10_data():
    images_old, labels = load()
    images = rearrange_images(images_old, 3)

    x_sample = images.astype("float")
    x_sample /= 255.0
    
    # Extract important features of the data
    input_dim = x_sample.shape[1:]
    output_dim = int(10)   

    x_train = x_sample[0:50000, :, :, :]
    x_test = x_sample[50000:, :, :, :]
    y_train = labels[0:50000, :]
    y_test = labels[50000:, :]

    # Doubles must be converted to Floats before passing them to a neural network model
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_train = torch.squeeze(y_train,1)
    y_test = torch.from_numpy(y_test).long()  
    y_test = torch.squeeze(y_test,1)

    return input_dim, output_dim, x_train, x_test, y_train, y_test
