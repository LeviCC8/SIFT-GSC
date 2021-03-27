import numpy as np


class Individual:

    def __init__(self, size, max_value, key_points, window_size):
        self.size = size
        self.max_value = max_value
        self.key_points = key_points
        self.window_size = window_size
        self.fitness_value = 0
        self.chromosome, self.index_pivot = self.__generate_chromosome()
        self.update_fitness_value()

    def __generate_chromosome(self):
        index_pivot, value_pivot = self.__generate_pivot()

        chromosome = self.__random_chromosome(index_pivot, value_pivot)

        while not self.__is_valid(chromosome, index_pivot):
            chromosome = self.__random_chromosome(index_pivot, value_pivot)

        return chromosome, index_pivot

    def __generate_pivot(self):
        index_pivot = np.random.randint(0, self.size)
        value_pivot = np.random.randint(0, self.max_value)
        return index_pivot, value_pivot

    def __random_chromosome(self, index_pivot, value_pivot):
        chromosome = np.random.randint(-1, 2, self.size)
        chromosome[index_pivot] = value_pivot

        return chromosome

    def __is_valid(self, chromosome, index_pivot):
        seam = self.get_seam(chromosome, index_pivot)
        condition = lambda x: np.all(np.logical_and(x >= 0, x < self.max_value))

        return condition(seam)

    def get_seam(self, chromosome, index_pivot):
        chromosome_part1 = np.flip(np.cumsum(np.flip(chromosome[:index_pivot + 1])))
        chromosome_part2 = np.cumsum(chromosome[index_pivot:])[1:]
        return np.append(chromosome_part1, chromosome_part2)

    def mutate(self):
        genes_to_mutate = np.random.randint(1, self.size)
        indexes = np.random.choice(range(self.size), genes_to_mutate, replace=False)
        values = np.random.randint(-1, 2, genes_to_mutate)
        self.__update_genes(indexes, values)
        self.update_fitness_value()

    def __update_genes(self, indexes, values):
        np.put(self.chromosome, indexes, values)
        self.chromosome = np.delete(self.chromosome, self.index_pivot)
        self.index_pivot, value_pivot = self.__generate_pivot()
        self.chromosome = np.insert(self.chromosome, self.index_pivot, value_pivot)

    def update_fitness_value(self):
        seam = self.get_seam(self.chromosome, self.index_pivot)
        if self.__is_valid(self.chromosome, self.index_pivot):
            distances = self.__calculate_distances(seam)
            self.fitness_value = self.__penalty_function(distances)
        else:
            invalid_values = np.sum(np.logical_or(seam < 0, seam >= self.max_value))
            self.fitness_value = invalid_values + 1

    def __penalty_function(self, distances):
        values = (0.5*self.window_size - distances)/(0.5*self.window_size)
        np.put(values, np.where(values < 0), 0)
        return np.sum(values)/self.key_points.shape[0]

    def __calculate_distances(self, seam):
        distances = np.expand_dims(self.key_points, axis=1) - list(enumerate(seam))
        distances = np.linalg.norm(distances, axis=2)
        distances = np.min(distances, axis=1)
        return distances
