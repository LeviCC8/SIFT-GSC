from individual import Individual
import numpy as np
from copy import deepcopy


class Population:

    def __init__(self, image_shape, key_points, population_number, window_size):
        self.population_number = population_number
        self.window_size = window_size
        self.image_shape = image_shape
        self.key_points = key_points
        self.individuals = self.__generate_individuals()

    def __generate_individuals(self):
        individuals = np.array([])
        for _ in range(self.population_number):
            individual = Individual(self.image_shape[0], self.image_shape[1], self.key_points, self.window_size)
            individuals = np.append(individuals, individual)
        return individuals

    def genetic_operators(self, elite_percentage, children_percentage, mutant_percentage):
        assert elite_percentage+children_percentage+mutant_percentage == 1, 'The percentages sum must be equals to 1'
        elite_individuals = self.__elitism_selection(elite_percentage)
        mutant_individuals = self.__mutate(mutant_percentage)
        spins = self.population_number - (elite_individuals.shape[0] + mutant_individuals.shape[0])
        parents = self.__roulette_wheel_selection(spins)
        children_individuals = self.__crossover(parents)
        self.individuals = np.append(elite_individuals, mutant_individuals)
        self.individuals = np.append(self.individuals, children_individuals)

    def __elitism_selection(self, elite_percentage):
        fitness_values = self.get_fitness_values()
        maintain = int(self.population_number*elite_percentage)
        arg_indexes = np.argsort(fitness_values)[:maintain]
        elite_individuals = self.individuals[arg_indexes]
        return elite_individuals

    def __roulette_wheel_selection(self, spins):
        fitness_values = self.get_fitness_values()
        i = np.max(fitness_values)
        selected_individuals = np.random.choice(self.individuals, size=spins, p=(i-fitness_values)/np.sum(i-fitness_values))
        return selected_individuals

    def get_fitness_values(self):
        fitness_values = np.zeros(self.individuals.shape[0])
        for i, individual in enumerate(self.individuals):
            fitness_values[i] = individual.fitness_value
        return fitness_values

    def __crossover(self, parents):
        children = np.array([])
        while children.shape[0] < parents.shape[0]:
            child1, child2 = self.__generate_children(parents)
            children = np.append(children, np.array([child1, child2]))
        return children[:parents.shape[0]]

    def __generate_children(self, parents):
        parent1 = np.random.choice(parents)
        parent2 = np.random.choice(parents)
        while np.all(parent2.chromosome == parent1.chromosome):
            parent2 = np.random.choice(parents)
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        value_pivot1 = child1.chromosome[child1.index_pivot]
        child1.chromosome = np.delete(child1.chromosome, child1.index_pivot)
        value_pivot2 = child2.chromosome[child2.index_pivot]
        child2.chromosome = np.delete(child2.chromosome, child2.index_pivot)
        self.__swap_genes(child1, child2)
        child1.chromosome = np.insert(child1.chromosome, child1.index_pivot, value_pivot1)
        child1.update_fitness_value()
        child2.chromosome = np.insert(child2.chromosome, child2.index_pivot, value_pivot2)
        child2.update_fitness_value()
        return child1, child2

    def __swap_genes(self, child1, child2):
        index_split = np.random.randint(child1.chromosome.shape[0]-1)
        store = np.copy(child1.chromosome[index_split+1:])
        child1.chromosome[index_split+1:] = child2.chromosome[index_split+1:]
        child2.chromosome[index_split+1:] = store

    def __mutate(self, mutant_percentage):
        mutations = int(mutant_percentage*self.population_number)
        mutants = np.array([])
        for _ in range(mutations):
            individual = deepcopy(np.random.choice(self.individuals))
            individual.mutate()
            mutants = np.append(mutants, individual)
        return mutants

    def get_fittest_seams(self):
        fitness_values = self.get_fitness_values()
        fittest_individuals = self.individuals[fitness_values == min(fitness_values)]
        fittest_seams = self.get_seams(fittest_individuals)
        return self.__discard_crossing_seams(fittest_seams)

    def __discard_crossing_seams(self, seams):
        crossing_costs = self.__calculate_crossing_costs(seams)
        while not np.all(crossing_costs == 0):
            index = np.argmax(crossing_costs)
            seams = np.delete(seams, index, axis=0)
            crossing_costs = self.__calculate_crossing_costs(seams)
        return seams

    def __calculate_crossing_costs(self, seams):
        costs = np.tile(seams, [1, seams.shape[0]])
        costs = np.reshape(costs, [seams.shape[0], seams.shape[0], seams.shape[1]])
        costs = np.any(costs - seams == 0, axis=2)
        costs = np.sum(costs, axis=1) - 1
        return costs

    def get_seams(self, individuals):
        seams = []
        for individual in individuals:
            seam = individual.get_seam(individual.chromosome, individual.index_pivot)
            seams.append(seam)
        return np.array(seams)
