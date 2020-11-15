from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical


def get_cifar10():
    """
    Retrieve the CIFAR dataset and pre-process data.
    """
    # Set defaults
    num_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return num_classes, batch_size, input_shape, x_train, x_test, y_train, y_test


def get_mnist():
    """
    Retrieve the MNIST dataset and pre-process data.
    """
    # Set defaults
    num_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return num_classes, batch_size, input_shape, x_train, x_test, y_train, y_test


def get_dataset(dataset: str) -> tuple:
    """
    Get the desired dataset
    :param dataset: string representing a supported dataset <mnist, cifar10>
    :return: preprocessed dataset (num_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)
    """
    if dataset == 'cifar10':
        return get_cifar10()
    elif dataset == 'mnist':
        return get_mnist()
    else:
        raise Exception('Dataset not supported yet')
