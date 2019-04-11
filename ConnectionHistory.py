import Node
import ConnectionGene


class ConnectionHistory:
    def __init__(self, in_node, out_node, innovation, innovation_numbers):
        self.in_node = in_node
        self.out_node = out_node
        self.innovation = innovation
        self.innovation_numbers = innovation_numbers

    def matches(self, genome, in_node, out_node):
        if len(genome.connection_genes) == len(self.innovation_numbers):
            if in_node.number == self.in_node and out_node.number == self.out_node:
                for i in range(len(genome.connection_genes)):
                    if not genome.connection_genes[i].innovation in self.innovation_numbers:
                        return False
                return True
        return False
