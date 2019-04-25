import Genome
import Node
import Player
import Population
import Species
import ConnectionGene
import ConnectionHistory
import retro
import neat
import numpy as np
import cv2
import retro

'''NEATDriver is responsible for managing the NEAT algorithm, it is here that the population
is created, and breeding occurs. After every generation, the genomes are sorted into species
and underperforming species are culled. The champion of the population is copied into the next
generation and the process repeats
'''
class NEATDriver:
    def __init__(self):
        self.next_connection_number = 1000
        self.pop = Population.Population(20)
        self.speed = 60
        self.generation = 0
        self.run(self.pop)

    def run(self, population):
        pop = population
        while not pop.done():
            self.generation += 1
            print("Generation: ", self.generation)
            champion = pop.update_alive()
            print("Champion: ", champion.fitness)
            pop.speciate()
            pop.sort_species()
            pop.kill_bad_species()
            pop.sort_species()
            for p in range(1, len(pop.pop)):
                pop.pop[p].brain.mutate(pop.innovation_history)
                pop.pop[p].fitness = 0
            pop.pop[0] = champion
            pop.pop[0].fitness = 0
            print("-----------------------")
        pop.natural_selection()

nd = NEATDriver()
