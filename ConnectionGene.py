import random
import numpy as np


class ConnectionGene:
    def __init__(self, in_node, out_node, weight, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation = innovation
        self.enabled = True

    def mutate(self):
        if random.uniform(0, 1) < 0.1:
            self.weight = random.uniform(-1, 1)
        else:
            self.weight += np.random.normal()
        if self.weight > 1:
            self.weight = 1
        elif self.weight < -1:
            self.weight = -1

    def copy(self, in_node, out_node):
        copy = ConnectionGene(in_node, out_node, self.weight, self.innovation)
        copy.enabled = True
        return copy
