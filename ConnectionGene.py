import random
import numpy as np

'''The ConnectionGene class encompases the connections which exist between neurons
in each genome. ConnectionGenes track the input node and output node, and each have
an associated weight, and keep track of the global innovation.
'''
class ConnectionGene:
    def __init__(self, in_node, out_node, weight, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation = innovation
        self.enabled = True

    '''The mutate function mutates the weight of the connection gene. The weight
    may be randomized between -1 and 1, or it may be adjusted slightly.
    '''
    def mutate(self):
        if random.uniform(0, 1) < 0.1:
            self.weight = random.uniform(-1, 1)
        else:
            self.weight += np.random.normal()
        if self.weight > 1:
            self.weight = 1
        elif self.weight < -1:
            self.weight = -1

    '''Copies the ConnectionGene for breeding'''
    def copy(self, in_node, out_node):
        copy = ConnectionGene(in_node, out_node, self.weight, self.innovation)
        copy.enabled = True
        return copy
