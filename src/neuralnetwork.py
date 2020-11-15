import random
import logging
from utils.dataset import get_dataset
import tensorflow.keras.callbacks as c
import tensorflow.keras.models as models
from tensorflow.keras.layers import Dense, Dropout


class NeuralNetwork:
    """
    Represent a neural network instance.
    Only works for MLP.
    """

    def __init__(self, nn_param_choices=None):
        """
        Intialize the neural network
        :param nn_param_choices: dict with network parameter choices, including:
            - num_neurons (list)
            - num_layers (list)
            - activations (list)
            - optimizers (list)
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dict): represents MLP network parameters

    def create_random(self):
        """
        Create a random network.
        """
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def set_network_parameters(self, network: dict):
        """
        Set network properties.
        :param network: network parameters dictionary
        """
        self.network = network

    def train(self, dataset: str):
        """
        Train the network and store the accuracy.
        :param dataset: Name of dataset to use.
        :return:
        """
        if self.accuracy == 0.:
            self.accuracy = self._train_and_score(dataset)

    def _train_and_score(self, dataset: str) -> float:
        """
        Private function that trains a network on a dataset and returns accuracy on test set
        :param dataset: dataset to be used for training and evaluation
        :return: test accuracy
        """
        num_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_dataset(dataset=dataset)

        model = self._compile_model(self.network, num_classes, input_shape)

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=10000,  # using early stopping, so no real limit
                  verbose=0,
                  validation_data=(x_test, y_test),
                  callbacks=[c.EarlyStopping(patience=5)])

        score = model.evaluate(x_test, y_test, verbose=0)

        return score[1]

    def _compile_model(self, network: dict, num_classes: int, input_shape: tuple) -> models.Sequential:
        """
        Builds a model with tf.keras sequential APIs and compiles it.
        :param network: dict with hyper-parameters
        :param num_classes: number of output classes
        :param input_shape: shape of input tensor
        :return: model
        """
        # Get network parameters
        nb_layers = network['num_layers']
        nb_neurons = network['num_neurons']
        activation = network['activations']
        optimizer = network['optimizers']

        model = models.Sequential()

        # Add layers
        for i in range(nb_layers):

            # Set input shape for the first layer
            if i == 0:
                model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(nb_neurons, activation=activation))

            model.add(Dropout(0.2))  # hard-coded dropout because why not

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def print_network(self):
        """
        Print out the network.
        """
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
