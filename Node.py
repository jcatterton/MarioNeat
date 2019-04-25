import numpy as np

'''Node refers to the Neuron genes of the genomes'''
class Node:

    output_value = 0

    def __init__(self, no):
        self.number = no
        self.input_sum = 0
        self.output_value = 0
        self.output_connections = []
        self.layer = 0

    '''The sigmoid function is used to help set the output value of a neuron to 1 or 0 depending
    on the input value which is most likely a decimal between 0 and 1'''
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-4.9 * x))

    '''Helper function to determine if two neurons are connected'''
    def is_connected_to(self, node):
        if node.layer == self.layer:
            return False
        if node.layer < self.layer:
            for i in range(len(node.output_connections)):
                if node.output_connections[i].to_node == self:
                    return True
        else:
            for i in range(len(self.output_connections)):
                if self.output_connections[i].to_node == node:
                    return True
        return False

    '''copies a neuron for breeding'''
    def copy(self):
        copy = Node(self.number)
        copy.layer = self.layer
        return copy

    '''Send the neuron's output to the neurons in the next layer'''
    def engage(self):
        if self.layer != 0:
            self.output_value = self.sigmoid(self.input_sum)
        for i in range(len(self.output_connections)):
            if self.output_connections[i].enabled:
                self.output_connections[i].out_node.input_sum += \
                    self.output_connections[i].weight * self.output_value
