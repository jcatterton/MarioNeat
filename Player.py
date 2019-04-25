import Genome
import time

'''The Player class serves as an interface between the NEAT algorithm and the game environment.
Output from the network is interpreted into input for the game, and fitness is determined which
is used for breeding.'''
class Player:
    def __init__(self, env):
        self.fitness = 0
        self.vision = []
        self.decision = []
        self.unadjusted_fitness = 0
        self.lifespan = 0
        self.best_score = 0
        self.dead = False
        self.score = 0
        self.gen = 0
        self.genome_outputs = 12
        self.genome_inputs = 6
        self.brain = Genome.Genome(self.genome_inputs, self.genome_outputs)
        self.x_pos = 0
        self.y_pos = 0
        self.layer_x = 0
        self.layer_y = 0
        self.x_speed = 0
        self.y_speed = 0
        self.env = env
        self.max_x = 0
        # self.imgarray = []

    def act(self):
        ob = self.env.reset()

        '''
        These variables are used to send a screen grab to the neural network
        as input, however doing so caused the program to become very slow during
        breeding.
        inx, iny, inc = self.env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))

        for x in ob:
            for y in x:
                self.imgarray.append(y)
        '''

        self.env.reset()
        done = False
        counter = 0

        x = 0

        while not done:
            self.env.render()

            '''info refers to info.json which is used to more easily read game memory'''
            a, b, c, info = self.env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            self.x_pos = info['x']
            self.y_pos = info['y']
            self.layer_x = info['layer1x']
            self.layer_y = info['layer1y']
            self.x_speed = info['xvelocity']
            self.y_speed = info['yvelocity']

            self.look()
            self.think()
            self.update()

            counter += 1
            self.calculate_fitness()

            '''If the player does not move for a certain amount of time, the run is killed.
            This is done to end a run when a player dies and also end runs early if the player
            is not moving at all, which is common in early genomes.'''
            if counter % 50 == 0 and counter > 0 or counter >= 500 and self.x_pos <= 150:
                if x == self.x_pos:
                    print("Fitness: ", self.fitness)
                    done = True
                else:
                    x = self.x_pos

            time.sleep(0.0166)

    def copy(self):
        copy = Player(self.env)
        copy.brain = self.brain.copy()
        copy.fitness = self.fitness
        copy.brain.generate_network()
        copy.gen = self.gen
        copy.best_score = self.best_score
        return copy

    def clone_for_replay(self):
        copy = Player(self.env)
        copy.brain = self.brain.copy()
        copy.fitness = self.fitness
        copy.brain.generate_network()
        copy.gen = self.gen
        copy.best_score = self.best_score
        return copy

    '''Fitness is set to the distance the player managed to travel'''
    def calculate_fitness(self):
        self.fitness = self.x_pos

    '''Breeds the player with another player'''
    def crossover(self, parent_2):
        child = Player()
        child.brain = self.brain.crossover(parent_2.brain)
        child.brain.generate_network()
        return child

    '''gets input for the network'''
    def look(self):
        self.vision = [self.x_pos, self.y_pos, self.layer_x, self.layer_y, self.x_speed, self.y_speed]

    '''gets output from the network'''
    def think(self):
        self.decision = self.brain.feed_forward(self.vision)
        self.env.step(self.decision)

    def update(self):
        self.score += 1

