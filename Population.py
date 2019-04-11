import Player
import math
import Species
import retro


class Population:
    def __init__(self, size):
        self.pop = []
        self.best_player = None
        self.best_score = 0
        self.gen = 0
        self.innovation_history = []
        self.gen_players = []
        self.species = []
        self.mass_extinction_event = False
        self.new_stage = False
        self.env = retro.make("SuperMarioWorld-Snes", "DonutPlains1.state")
        self.env.reset()
        self.env.render()

        for i in range(size):
            self.pop.append(Player.Player(self.env))
            self.pop[i].brain.generate_network()
            self.pop[i].brain.mutate(self.innovation_history)

    def update_alive(self):
        champion = self.pop[0]
        for i in range(len(self.pop)):
            if not self.pop[i].dead:
                self.pop[i].brain.connect_nodes()
                self.pop[i].act()
                self.pop[i].calculate_fitness()
                if self.pop[i].fitness > champion.fitness:
                    champion = self.pop[i]
        return champion

    def done(self):
        for i in range(len(self.pop)):
            if not self.pop[i].dead:
                return False
        return True

    def set_best_player(self):
        temp_best = self.species[0].players[0]
        temp_best.gen = self.gen

        if temp_best.score > self.best_score:
            self.gen_players.append(temp_best.clone_for_replay())
            self.best_score = temp_best.score
            self.best_player = temp_best.clone_for_replay()

    def speciate(self):
        for s in self.species:
            s.players.clear()

        for i in range(len(self.pop)):
            species_found = False
            for s in self.species:
                if s.same_species(self.pop[i].brain):
                    s.add_to_species(self.pop[i])
                    species_found = True
                    break
            if not species_found:
                self.species.append(Species.Species(self.pop[i]))

    def calculate_fitness(self):
        for i in range(len(self.pop)):
            self.pop[i].calculate_fitness()

    def sort_species(self):
        for s in self.species:
            s.sort_species()

        temp = []
        for i in range(len(self.species)):
            max_fitness = 0
            max_index = 0
            for j in range(len(self.species)):
                if self.species[j].best_fitness > max_fitness:
                    max_fitness = self.species[j].best_fitness
                    max_index = j
            temp.append(self.species[max_index])
            self.species.remove(self.species[max_index])
            i -= 1
        self.species = temp.copy()

    def kill_stale_species(self):
        for i in range(2, len(self.species)):
            if self.species[i].staleness >= 15:
                self.species.remove(i)
                i -= 1

    def get_average_fitness_sum(self):
        average_sum = 0
        for s in self.species:
            average_sum += s.average_fitness
        return average_sum

    def kill_bad_species(self):
        average_sum = self.get_average_fitness_sum()
        if average_sum == 0:
            average_sum = 0.01
        for i in range(1, len(self.species)):
            if self.species[i].average_fitness/average_sum * len(self.pop) < 1:
                self.species.remove(self.species[i])
                i -= 1

    def cull_species(self):
        for s in self.species:
            s.cull()
            s.fitness_sharing()
            s.set_average()

    def mass_extinction(self):
        for i in range(5, len(self.species)):
            self.species.remove(i)
            i -= 1

    def natural_selection(self):
        self.speciate()
        self.calculate_fitness()
        self.sort_species()
        if self.mass_extinction_event:
            self.mass_extinction()
            self.mass_extinction_event = False
        self.cull_species()
        self.set_best_player()
        self.kill_stale_species()
        self.kill_bad_species()

        print("Generation: ", self.gen, " Number of Mutations: ",
              len(self.innovation_history),
              " Species: ", len(self.species))

        average_sum = self.get_average_fitness_sum()
        children = []
        print("Species: ")
        for i in range(len(self.species)):
            print("Best unadjusted fitness: ", self.species[i].best_fitness)
            for j in range(len(self.species[i].players)):
                print("Player ", j, " fitness: ", self.species[i].players[j].fitness, " Score ",
                      self.species[i].players[j].score)
            children.append(self.species[i].champ.clone_for_replay())

            number_of_children = math.floor(self.species[i].average_fitness/average_sum*len(self.pop)) - 1
            for j in range(number_of_children):
                children.append(self.species[i].give_me_baby(self.innovation_history))

        while len(children) < len(self.pop):
            children.append(self.species[0].give_me_baby(self.innovation_history))

        self.pop.clear()
        self.pop = children.copy()
        self.gen += 1
        for i in range(len(self.pop)):
            self.pop[i].brain.generate_network()

