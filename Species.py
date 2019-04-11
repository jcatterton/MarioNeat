import math
import random


class Species:
    def __init__(self, player):
        self.players = []
        self.best_fitness = 0
        self.champ = None
        self.average_fitness = 0
        self.staleness = 0
        self.rep = None
        self.excess_coefficient = 1
        self.weight_difference_coefficient = 0.5
        self.compatibility_threshold = 3

        if player is not None:
            self.players.append(player)
            self.best_fitness = player.fitness
            self.rep = player.brain.copy()
            self.champ = player.clone_for_replay

    def add_to_species(self, player):
        self.players.append(player)

    @staticmethod
    def get_excess_disjoint(brain_1, brain_2):
        matching = 0
        for i in range(len(brain_1.connection_genes)):
            for j in range(len(brain_2.connection_genes)):
                if brain_1.connection_genes[i].innovation == brain_2.connection_genes[j].innovation:
                    matching += 1
                    break
        return len(brain_1.connection_genes) + len(brain_2.connection_genes) - 2*matching

    @staticmethod
    def average_weight_difference(brain_1, brain_2):
        if len(brain_1.connection_genes) == 0 or len(brain_2.connection_genes) == 0:
            return 0

        matching = 0
        total_difference = 0

        for i in range(len(brain_1.connection_genes)):
            for j in range(len(brain_2.connection_genes)):
                if brain_1.connection_genes[i].innovation == brain_2.connection_genes[j].innovation:
                    matching += 1
                    total_difference += abs(brain_1.connection_genes[i].weight - brain_2.connection_genes[j].weight)
                    break

        if matching == 0:
            return 100

        return total_difference/matching

    def same_species(self, genome):
        excess_and_disjoint = self.get_excess_disjoint(genome, self.rep)
        average_weight_difference = self.average_weight_difference(genome, self.rep)

        large_genome_normaliser = len(genome.connection_genes) - 20

        if large_genome_normaliser < 1:
            large_genome_normaliser = 1

        compatibility = (self.excess_coefficient * excess_and_disjoint/large_genome_normaliser) \
            + (self.weight_difference_coefficient * average_weight_difference)

        return self.compatibility_threshold > compatibility

    def sort_species(self):
        temp = []
        for i in range(len(self.players)):
            maximum = 0
            max_index = 0
            for j in range(len(self.players)):
                if self.players[j].fitness > maximum:
                    maximum = self.players[j].fitness
                    max_index = j
            temp.append(self.players[max_index])
            self.players.remove(self.players[max_index])
            i -= 1

        self.players = temp.copy()
        if len(self.players) == 0:
            self.staleness = 200
            return

        if self.players[0].fitness > self.best_fitness:
            self.staleness = 0
            self.best_fitness = self.players[0].fitness
            self.rep = self.players[0].brain.copy()
            self.champ = self.players[0].clone_for_replay()
        else:
            self.staleness += 1

    def set_average(self):
        fitness_sum = 0
        for i in range(len(self.players)):
            fitness_sum += self.players[i].fitness

        self.average_fitness = fitness_sum/len(self.players)

    def select_player(self):
        fitness_sum = 0
        for i in range(len(self.players)):
            fitness_sum += self.players[i].fitness

        rand = random.uniform(0, fitness_sum)
        running_sum = 0

        for i in range(len(self.players)):
            running_sum += self.players[i].fitness
            if running_sum > rand:
                return self.players[i]

        return self.players[0]

    def give_me_baby(self, innovation_history):
        if random.uniform(0, 1) < 0.25:
            baby = self.select_player().copy()
        else:
            parent_1 = self.select_player()
            parent_2 = self.select_player()

            if parent_1.fitness < parent_2.fitness:
                baby = parent_2.crossover(parent_1)
            else:
                baby = parent_1.crossover(parent_2)

        baby.brain.mutate(innovation_history)
        return baby

    def cull(self):
        self.sort_species()
        if len(self.players) > 2:
            x = math.floor(len(self.players)/2)
            for i in range(x):
                del self.players[-1]

    def fitness_sharing(self):
        for i in range(len(self.players)):
            self.players[i].fitness /= len(self.players)
