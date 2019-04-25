import Node
import ConnectionGene
import random
import ConnectionHistory
import math
from PIL import Image, ImageDraw
import vector_2d

'''In a NEAT algorithm, the Genome is the network of neurons and connections which performs
actions. Genomes may mutate by adding new neurons and connections, and adjusting the weights
of connections. When created, genomes have only two layers (input and output), but more layers
are created as new neurons are added.'''
class Genome:
    def __init__(self, inputs, outputs):
        self.connection_genes = []
        self.node_genes = []
        self.inputs = inputs
        self.outputs = outputs
        self.layers = 2
        self.next_node = 0
        self.bias_node = None
        self.network = []

        local_connection_number = 0

        '''The input and output layer are created manually depending on the desired inputs
        and outputs provided'''
        
        for i in range(inputs):
            self.node_genes.append(Node.Node(i))
            self.next_node += 1
            self.node_genes[i].layer = 0

        for i in range(outputs):
            self.node_genes.append(Node.Node(i + inputs))
            self.node_genes[i + inputs].layer = 1
            self.next_node += 1

        self.node_genes.append(Node.Node(self.next_node))
        self.bias_node = self.next_node
        self.next_node += 1
        self.node_genes[self.bias_node].layer = 0

        '''A genome is fully connected when first created'''
        
        for i in range(inputs):
            for j in range(outputs):
                connection_gene = ConnectionGene.ConnectionGene(self.node_genes[i], self.node_genes[inputs + j],
                                                                random.uniform(-1, 1), local_connection_number)
                self.connection_genes.append(connection_gene)
                local_connection_number += 1

        for i in range(outputs):
            connection_gene = ConnectionGene.ConnectionGene(self.node_genes[self.bias_node],
                                                            self.node_genes[inputs + 1], random.uniform(-1, 1),
                                                            local_connection_number)
            self.connection_genes.append(connection_gene)
            local_connection_number += 1

    '''Returns a neuron'''
    def get_node(self, node_number):
        for i in range(len(self.node_genes)):
            if self.node_genes[i].number == node_number:
                return self.node_genes[i]
        return None

    '''Connects all neurons as defined by the ConnectionGenes, done in order to ensure no
    accidental loose neurons'''
    def connect_nodes(self):
        for i in range(len(self.node_genes)):
            self.node_genes[i].output_connections.clear()

        for i in range(len(self.connection_genes)):
            self.connection_genes[i].in_node.output_connections.append(self.connection_genes[i])

    '''Send input from the input layer through all layers of the genome and returns results
    from the output layer'''
    def feed_forward(self, input_values):
        for i in range(self.inputs):
            self.node_genes[i].output_value = input_values[i]

        self.node_genes[self.bias_node].output_value = 1

        for i in range(len(self.network)):
            self.network[i].engage()

        outputs = []
        for i in range(self.outputs):
            outputs.append(self.node_genes[self.inputs + i].output_value)

        for i in range(len(self.node_genes)):
            self.node_genes[i].input_sum = 0

        return outputs

    '''Adds all node genes to the network array'''
    def generate_network(self):
        self.connect_nodes()
        self.network = []
        for l in range(self.layers):
            for i in range(len(self.node_genes)):
                if self.node_genes[i].layer == l:
                    self.network.append(self.node_genes[i])

    '''returns innovation number'''
    def get_innovation_number(self, innovation_history, in_node, out_node):
        is_new = True
        connection_innovation_number = 1000
        for i in range(len(innovation_history)):
            if innovation_history[i].matches(self, in_node, out_node):
                is_new = False
                connection_innovation_number = innovation_history[i].innovation
                break

        if is_new:
            inno_numbers = []
            for i in range(len(self.connection_genes)):
                inno_numbers.append(self.connection_genes[i].innovation)
            innovation_history.append(ConnectionHistory.ConnectionHistory(in_node.number, out_node.number,
                                                                          connection_innovation_number, inno_numbers))
        return connection_innovation_number

    '''Adds a neuron to the genome. When a new neuron is added, an existing connection is split and the
    new neuron is added between the two new connections. This is done to avoid extraneous neurons.'''
    def add_node(self, innovation_history):
        random_connection = random.randrange(0, len(self.connection_genes))

        while self.connection_genes[random_connection].in_node == self.node_genes[self.bias_node]:
            random_connection = random.randrange(0, len(self.connection_genes))

        self.connection_genes[random_connection].enabled = False

        new_node_number = self.next_node
        new_node = Node.Node(new_node_number)
        self.node_genes.append(new_node)
        self.next_node += 1

        connection_innovation_number = self.get_innovation_number(innovation_history,
                                                                  self.connection_genes[random_connection].in_node,
                                                                  self.get_node(new_node_number))
        self.connection_genes.append(ConnectionGene.ConnectionGene(self.connection_genes[random_connection].in_node,
                                                                   self.get_node(new_node_number),
                                                                   1, connection_innovation_number))

        connection_innovation_number = self.get_innovation_number(innovation_history,
                                                                  self.get_node(new_node_number),
                                                                  self.connection_genes[random_connection].out_node)
        self.connection_genes.append(ConnectionGene.ConnectionGene(self.get_node(new_node_number),
                                                                   self.connection_genes[random_connection].out_node,
                                                                   self.connection_genes[random_connection].weight,
                                                                   connection_innovation_number))
        self.get_node(new_node_number).layer = self.connection_genes[random_connection].in_node.layer + 1

        connection_innovation_number = self.get_innovation_number(innovation_history, self.node_genes[self.bias_node],
                                                                  self.get_node(new_node_number))
        self.connection_genes.append(ConnectionGene.ConnectionGene(self.node_genes[self.bias_node],
                                                                   self.get_node(new_node_number),
                                                                   0, connection_innovation_number))

        if self.get_node(new_node_number).layer == self.connection_genes[random_connection].out_node.layer:
            for i in range(len(self.node_genes)):
                if self.node_genes[i].layer >= self.get_node(new_node_number).layer:
                    self.node_genes[i].layer += 1
            self.layers += 1

        self.connect_nodes()

    '''Helper function, used to determine if the place a connection is being added is a viable option'''
    def bad_random_connections(self, random_node_1, random_node_2):
        if self.node_genes[random_node_1].layer == self.node_genes[random_node_2].layer:
            return True
        if self.node_genes[random_node_1].is_connected_to(self.node_genes[random_node_2]):
            return True
        return False

    '''Helper function, determines if there are any available spots to add a connection'''
    def fully_connected(self):
        max_connections = 0
        nodes_in_layers = []

        for i in range(self.layers):
            nodes_in_layers.append(0)

        for i in range(len(self.node_genes)):
            nodes_in_layers[self.node_genes[i].layer] += 1

        for i in range(self.layers - 1):
            nodes_in_front = 0
            for j in range(i + 1, self.layers):
                nodes_in_front += nodes_in_layers[j]
            max_connections += nodes_in_layers[i] * nodes_in_front

        if max_connections == len(self.connection_genes):
            return True
        return False

    '''Adds a connection between two random non-connected neurons with a random weight'''
    def add_connection(self, innovation_history):
        if self.fully_connected:
            return

        random_node_1 = math.floor(random.randint(len(self.node_genes)))
        random_node_2 = math.floor(random.randint(len(self.node_genes)))

        while self.bad_random_connections(random_node_1, random_node_2):
            random_node_1 = math.floor(random.randint(len(self.node_genes)))
            random_node_2 = math.floor(random.randint(len(self.node_genes)))

        if self.node_genes[random_node_1].layer > self.node_genes[random_node_2].layer:
            temp = random_node_2
            random_node_2 = random_node_1
            random_node_1 = temp

        connection_innovation_number = self.get_innovation_number(innovation_history, self.node_genes[random_node_1],
                                                                  self.node_genes[random_node_2])

        self.connection_genes.append(ConnectionGene.ConnectionGene(self.node_genes[random_node_1],
                                                                   self.node_genes[random_node_2],
                                                                   random.uniform(-1, 1),
                                                                   connection_innovation_number))

        self.connect_nodes()

    '''Mutates the genome. Three random numbers are generated which gives the genome an 80% chance to mutate
    the connection weights, a 20% chance to add a new connection, and a 10% chance to add a new neuron. Three
    separate numbers are generated in order to assure these mutations are not mutually exclusive (One mutation
    can occur, or any combination of two, or all three, or none).'''
    def mutate(self, innovation_history):
        if len(self.connection_genes) == 0:
            self.add_connection(innovation_history)

        rand_1 = random.uniform(0, 1)

        if rand_1 < 0.8:
            for i in range(len(self.connection_genes)):
                self.connection_genes[i].mutate()

        rand_2 = random.uniform(0, 1)

        if rand_2 < 0.2:
            self.add_connection(innovation_history)

        rand_3 = random.uniform(0, 1)

        if rand_3 < 0.1:
            self.add_node(innovation_history)

    '''Used to locate a matching gene in another genome'''
    @staticmethod
    def matching_gene(parent_2, innovation_number):
        for i in range(len(parent_2.connection_genes)):
            if parent_2.connection_genes[i].innovation == innovation_number:
                return i
        return -1

    '''Breeding method, takes genes from two compatable parents to create a child genome'''
    def crossover(self, parent_2):
        child = Genome(self.inputs, self.outputs, True)
        child.connection_genes.clear()
        child.node_genes.clear()
        child.layers = self.layers
        child.next_node = self.next_node
        child.bias_node = self.bias_node

        child_genes = []
        is_enabled = []

        for i in range(len(self.connection_genes)):
            set_enabled = True

            parent_2_gene = self.matching_gene(parent_2, self.connection_genes[i].innovation)
            if parent_2_gene != -1:
                if not self.connection_genes[i].enabled or not parent_2.connection_genes[parent_2_gene].enabled:
                    if random.uniform(0, 1) < 0.75:
                        set_enabled = False

                rand = random.uniform(0, 1)
                if rand < 0.5:
                    child_genes.append(self.connection_genes[i])
                else:
                    child_genes.append(parent_2.connection_genes[parent_2_gene])
            else:
                child_genes.append(self.connection_genes[i])
                set_enabled = self.connection_genes[i].enabled
            is_enabled.append(set_enabled)

        for i in range(len(self.node_genes)):
            child.node_genes.append(self.node_genes[i].copy())

        for i in range(len(child_genes)):
            child.connection_genes.append(child_genes[i].copy(child.get_node(child_genes[i].in_node.number),
                                                              child.get_node(child_genes[i].out_node.number)))
            child.connection_genes[i].enabled = is_enabled[i]

        child.connect_nodes()
        return child

    '''Copies a genome for breeding'''
    def copy(self):
        copy = Genome(self.inputs, self.outputs)
        for i in range(len(self.node_genes)):
            copy.node_genes.append(self.node_genes[i])

        for i in range(len(self.connection_genes)):
            copy.connection_genes.\
                append(self.connection_genes[i].copy(copy.get_node(self.connection_genes[i].in_node.number),
                                                     copy.get_node(self.connection_genes[i].out_node.number)))

        copy.layers = self.layers
        copy.next_node = self.next_node
        copy.bias_node = self.bias_node
        copy.connect_nodes()

        return copy

    '''Helper function used to show information on a genome'''
    def print_genome(self):
        print("Print genome: ")
        print("Layers: ", self.layers)
        print("Bias Node: ", self.bias_node)
        print("Nodes: ")
        for i in range(len(self.node_genes)):
            print(self.node_genes[i].number, ", ", end=' ')
        print("Genes: ")
        for i in range(len(self.connection_genes)):
            print("Gene: ", self.connection_genes[i].innovation,
                  ", from node: ", self.connection_genes[i].in_node.number,
                  ", to node: ", self.connection_genes[i].out_node.number,
                  ", expresses?: ", self.connection_genes[i].enabled,
                  ", from layer: ", self.connection_genes[i].in_node.layer,
                  ", to layer: ", self.connection_genes[i].out_node.layer,
                  ", weight: ", self.connection_genes[i].weight)
