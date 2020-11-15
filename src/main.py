import argparse
from utils.logging import config_logger
from genetic.optimizer import GeneticOptimizer
import logging
from tqdm import tqdm


def train_networks(networks: list, dataset: str):
    """
    Trains networks on a specified dataset
    :param networks: list of networks
    :param dataset: string representing the dataset (mnist, cifar10)
    """
    for network in tqdm(networks):
        network.train(dataset)


def get_average_accuracy(networks: list):
    """
    Get the average accuracy for a list of networks
    :param networks: list of dict with network parameters
    :return: average accuracy obtained by all networks
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def print_networks(networks: list):
    """
    Print a list of networks
    :param networks: list of networks
    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()


def run(population: int, generations: int, hyperparams: dict, dataset: str):
    """
    Generates a Neural Network and trains it  with a genetic approach.
    :param population: number of networks in the population
    :param generations: number of new generations
    :param hyperparams: dict containing choices for num of neurons, num of layers, activations and optimizers
    :param dataset: dataset to be used for experimentation. We only consider simple datasets included in TensorFlow
    """
    optimizer = GeneticOptimizer(hyperparams)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("Generation {} of {}".format(i+1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out average accuracy for each generation
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-' * 80)

        # Evolve, except on the last iteration
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort out final population based on accuracy
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])


def main(arguments):
    config_logger()
    logger = logging.getLogger()

    population = arguments.population
    num_generations = arguments.generations
    logger.info("Meta-Learning started - Population size: {} & {} generations.".format(population, num_generations))
    dataset = arguments.dataset

    hyperparams_choices = {
        'num_neurons': [32, 64, 128, 256, 512],
        'num_layers': [1, 2, 3, 4],
        'activations': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizers': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    run(population, num_generations, hyperparams_choices, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=5)
    parser.add_argument('--population', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='mnist')
    args = parser.parse_args()
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
