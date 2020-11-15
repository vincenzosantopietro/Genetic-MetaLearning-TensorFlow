"""
Class that holds a genetic algorithm for evolving a population of MLP networks.
Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from neuralnetwork import NeuralNetwork


class GeneticOptimizer:
    """
    Class that implements genetic algorithm for MLP optimization.
    """

    def __init__(self, nn_param_choices: dict, retain: float = 0.4,
                 random_select: float = 0.1, mutate_chance: float = 0.2):
        """
        Create a genetic optimizer instance
        :param nn_param_choices: dict with all possible network hyperparameters
        :param retain: percentage of population to retain after each generation
        :param random_select: probability for a bad network to be kept in the population
        :param mutate_chance: probability for a network to be randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count: int) -> list:
        """
        Create a population of randomly initialised networks
        :param count: number of networks to be generated
        :return: population - list of networks
        """
        pop = []
        for _ in range(0, count):
            # Create a random network
            network = NeuralNetwork(self.nn_param_choices)
            network.create_random()

            # Add the network to our population
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network: NeuralNetwork) -> float:
        """
        Return the accuracy, which is desired fitness function.
        """
        return network.accuracy

    def grade(self, pop) -> float:
        """
        Compute average fitness of a population
        :param pop: population - list of networks
        :return: average fitness
        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother: NeuralNetwork, father: NeuralNetwork) -> list:
        """
        Create two children based on parents hyper parameters
        :param mother: mother network
        :param father: father network
        :return: list of children
        """
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object
            network = NeuralNetwork(self.nn_param_choices)
            network.set_network_parameters(child)

            # Randomly mutate some of the children
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network: NeuralNetwork) -> NeuralNetwork:
        """
        Randomly mutate one part of a given network
        :param network: input network
        :return: mutated network
        """
        # Choose a random key
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one of the params
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])
        return network

    def evolve(self, pop: list):
        """
        Evolve a population of networks
        :param pop: list of neural networks
        :return: evolved population
        """
        # Get scores for each network
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Find out how many spots we have left to fill
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks
        while len(children) < desired_length:

            # Get a random mom and dad
            p1 = random.randint(0, parents_length - 1)
            p2 = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if p1 != p2:
                p1 = parents[p1]
                p2 = parents[p2]

                # Breed them.
                babies = self.breed(p1, p2)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
