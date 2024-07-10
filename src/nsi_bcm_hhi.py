# -*- coding: utf-8 -*-
"""Merged_code_HHI_NSI

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TR22wHQJhP-RrBkuEsgW06K1FKSo8_CW
"""

# Commented out IPython magic to ensure Python compatibility.
import random
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

params_dict = {
    "population": {
        "simulation_repetition": 1,
        "grid_size": 15,
        "accessibility": 0.7,
        "time_steps": 75000
    },
    "agent": {
        "connectivity": 0.4,
        "learning_rate": 0.3,
        "confidence_threshold": 0.5,
        "tolerance": 0.05,

    }
}

class Params:
    def __init__(self):
        self._population_config = params_dict["population"]
        self._agent_config = params_dict["agent"]

    def get_population_parameter(self, parameter_name):
        if parameter_name not in self._population_config:
            return None
        return self._population_config[parameter_name]

    def get_agent_parameter(self, parameter_name):
        if parameter_name not in self._agent_config:
            return None
        return self._agent_config[parameter_name]


parameters = Params()

class Agent:
    def __init__(self, opinion, pos, delta, grid_size,
                 k, tolerance=0, dissenter=False) -> None:
        self.opinion = opinion
        self.posx = pos[0]
        self.posy = pos[1]
        self.delta = delta
        self.grid_size = grid_size
        self.k = k
        self.tolerance = tolerance
        self.dissenter = dissenter
        self.agent_type = "disconnected"          ### New add
        self.communication_status = "offline"     ### New add

    def getNeighbors(self) -> list:
        row, col = self.posx, self.posy
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in shifts:
            nr = (row + dr) % self.grid_size
            nc = (col + dc) % self.grid_size
            neighbors.append((nr, nc))
        return neighbors



    def getOpinion(self) -> float:
        return self.opinion
    def setOpinion(self, val) -> None:
        self.opinion = val
    def getPosition(self) -> list:
        return [self.posx, self.posy]
    def getDelta(self) -> int:
        return self.delta
    def setDelta(self, delta) -> None:
        self.delta = delta
    def setDissenter(self) -> None:
        self.dissenter = True
    def remDissenter(self) -> None:
        self.dissenter = False
    def checkDissenter(self) -> bool:
        return self.dissenter
    def setTolerance(self, tol) -> None:
        self.tolerance = tol
    def getTolerance(self) -> float:
        return self.tolerance

    def set_communication_status(self):         ### New add
        self.communication_status = "offline" if self.agent_type == "disconnected" else random.choice(["offline", "online"])

    def change_agent_type(self, type = "connected"):        ### New add
        self.agent_type = type
        self.set_communication_status()

class Population:
    grid = {}
    def __init__(self, grid_size=10, Uniform=True,
                 Beta=False, Random=False, conf=0.5,
                 learn=0.25, dis_percent=0.01) -> None:
        self.Uniform = Uniform
        self.Beta = Beta
        self.Random = Random
        self.grid_size = grid_size
        self.confidence_threshold = conf
        self.learning_rate = learn
        self.dis_percent = dis_percent
        self.createPopulation()

    def createPopulation(self) -> None:
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.grid[(row, col)] = Agent(self.createOpinion(), [row, col],
                                              self.createRandom(),
                                              self.grid_size,
                                              self.createRandom(),
                                              self.createRandomTolerance())

    def setDissenters(self) -> None:
        totalDissenters = self.dis_percent * self.grid_size * self.grid_size
        for i in range(totalDissenters):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.grid[[x, y]].setDissenter()

    def getNextOpinion(self, cell) -> int:
        return round(cell.getOpinion() + cell.k *
                     (round(self.getIdealOpinion(cell), 2) - cell.getOpinion()), 2)

    def getMeanOpinion(self, cell) -> float:
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getOpinion())
        if len(data) == 0: return float('nan')
        return round(sum(data) / len(data), 2)

    def getSDOpinion(self, cell) -> int:
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getOpinion())
        if not data: return None
        mean = sum(data) / len(data)
        squared_deviations = [pow(x - mean, 2) for x in data]
        variance = sum(squared_deviations) / len(data)
        return round(pow(variance, 0.5), 2)

    def getIdealOpinion(self, cell) -> float:
        return round(self.getMeanOpinion(cell) + self.learning_rate * self.getSDOpinion(cell), 2)

    def getAvgDelta(self, cell) -> int:
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getDelta())
        return sum(data) / len(data)

    def getNextDelta(self, cell) -> int:
        # NSI
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getDelta())
        newDelta = min(max(cell.getDelta() +
                            self.learning_rate *
                            (self.getAvgDelta(cell) - cell.getDelta()), -5),
                            5)
        return int(newDelta)

    def createOpinion(self) -> float:
        if self.Beta == True:
            return self.createBetaOpinion()
        if self.Uniform == True:
            return self.createUniformOpinion()
        if self.Random == True:
            return self.createRandomOpinion()

    def createBetaOpinion(self, alpha=2, beta=2) -> float:
        # alpha, beta -> positive
        u1, u2 = random.random(), random.random()
        t1 = pow(u1, 1/(alpha-1))
        t2 = pow(u2, 1/(beta-1))
        # Combine and transform to beta distribution [0, 1]
        sample = (t1 + t2) / (1 + t1 + t2)
        # Scale and shift to desired range [-1, 1]
        transformed_value = (sample - 0.5) * 2
        return round(transformed_value, 2)

    def createUniformOpinion(self, low=-1, high=1) -> float:
        return round(random.uniform(low, high), 2)

    def createRandom(self) -> float:
        value = random.random()
        while value <= 0 or value >= 1:
            value = random.random()
        return round(value, 2)

    def createRandomTolerance(self) -> float:
        value = random.random()
        while value <= 0 or value >= 0.15:
            value = random.random()
        return round(value, 2)

    def createRandomOpinion(self, low=-1, high=1) -> float:
        value = random.random() * (high + abs(low)) + low
        while low > value or value > high:
            value = random.random() * (high + abs(low)) + low
        return round(value, 2)

### Merging
### added new argument : parameters ===> update the population class initialization
## including new argument

class Population:
    grid = {}
    def __init__(self, parameters, grid_size=10, Uniform=True,
                 Beta=False, Random=False, conf=0.5,
                 learn=0.25, dis_percent=0.01) -> None:
        self.Uniform = Uniform
        self.Beta = Beta
        self.Random = Random
        self.grid_size = grid_size
        self.confidence_threshold = conf
        self.learning_rate = learn
        self.dis_percent = dis_percent
        self.createPopulation()

        #### New add
        self.accessibility = parameters.get_population_parameter("accessibility")
        self.connectivity = parameters.get_agent_parameter("connectivity")
        self.time_steps = parameters.get_population_parameter("time_steps")
        self.tolerance = parameters.get_agent_parameter("tolerance")

        self.population_size = self.grid_size * self.grid_size
        self.num_ConnectedAgents = int(self.grid_size * self.accessibility)
        self.num_DiffusionAtOnce = self.num_ConnectedAgents * self.connectivity
        self.all_Agents = None
        self.ConnectedAgents = None # a list of positions(tuple) of all connected agents
        self.DisconnectedAgents = None

        self._initialize_Agents()

        ####
    ######  NEW add
    def _initialize_Agents(self):

        # an agent is initialized randomly as connected owing to constraint "num_connectedagents"
        self.all_Agents = [(i,j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.ConnectedAgents = random.choices(self.all_Agents, k = self.num_ConnectedAgents)


        for pos in self.ConnectedAgents:
            self.grid[pos].change_agent_type(type = "connected")

        self.DisconnectedAgents = list(set(self.all_Agents) - set(self.ConnectedAgents))

    def _diffuse_opinion(self, agent, number_diffusionAtOnce):

                x1 = agent.opinion

                ConnectedAgents_of_cell = random.choices(self.ConnectedAgents, k = int(number_diffusionAtOnce))
                for pos in ConnectedAgents_of_cell:
                     x2 = self.grid[pos].opinion
                     x2_new = x2 + self.learning_rate * (x1 - x2)

                     self.grid[pos].opinion = round(x2_new, 2)

    ###########
    def createPopulation(self) -> None:
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.grid[(row, col)] = Agent(self.createOpinion(), [row, col],
                                              self.createRandom(),
                                              self.grid_size,
                                              self.createRandom(),
                                              self.createRandomTolerance())

    def setDissenters(self) -> None:
        totalDissenters = self.dis_percent * self.grid_size * self.grid_size
        for i in range(totalDissenters):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.grid[[x, y]].setDissenter()

    def getNextOpinion(self, cell) -> int:
        return round(cell.getOpinion() + cell.k *
                     (round(self.getIdealOpinion(cell), 2) - cell.getOpinion()), 2)

    def getMeanOpinion(self, cell) -> float:
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getOpinion())
        if len(data) == 0: return float('nan')
        return round(sum(data) / len(data), 2)

    def getSDOpinion(self, cell) -> int:
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getOpinion())
        if not data: return None
        mean = sum(data) / len(data)
        squared_deviations = [pow(x - mean, 2) for x in data]
        variance = sum(squared_deviations) / len(data)
        return round(pow(variance, 0.5), 2)

    def getIdealOpinion(self, cell) -> float:
        return round(self.getMeanOpinion(cell) + 0.2 * self.getSDOpinion(cell), 2)

    def getAvgDelta(self, cell) -> int:
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getDelta())
        return sum(data) / len(data)

    def getNextDelta(self, cell) -> int:
        # NSI
        data = []
        neighbors = cell.getNeighbors()
        for i in range(4):
            data.append(self.grid[neighbors[i]].getDelta())
        newDelta = min(max(cell.getDelta() +
                            self.learning_rate *
                            (self.getAvgDelta(cell) - cell.getDelta()), -5),
                            5)
        return int(newDelta)

    def createOpinion(self) -> float:
        if self.Beta == True:
            return self.createBetaOpinion()
        if self.Uniform == True:
            return self.createUniformOpinion()
        if self.Random == True:
            return self.createRandomOpinion()

    def createBetaOpinion(self, alpha=2, beta=2) -> float:
        # alpha, beta -> positive
        u1, u2 = random.random(), random.random()
        t1 = pow(u1, 1/(alpha-1))
        t2 = pow(u2, 1/(beta-1))
        # Combine and transform to beta distribution [0, 1]
        sample = (t1 + t2) / (1 + t1 + t2)
        # Scale and shift to desired range [-1, 1]
        transformed_value = (sample - 0.5) * 2
        return round(transformed_value, 2)

    def createUniformOpinion(self, low=-1, high=1) -> float:
        return round(random.uniform(low, high), 2)

    def createRandom(self) -> float:
        value = random.random()
        while value <= 0 or value >= 1:
            value = random.random()
        return round(value, 2)

    def createRandomTolerance(self) -> float:
        value = random.random()
        while value <= 0 or value >= 0.15:
            value = random.random()
        return round(value, 2)

    def createRandomOpinion(self, low=-1, high=1) -> float:
        value = random.random() * (high + abs(low)) + low
        while low > value or value > high:
            value = random.random() * (high + abs(low)) + low
        return round(value, 2)

grid = Population(parameters, grid_size=10, Uniform=True,
                 Beta=False, Random=False, conf=0.5,
                 learn=0.25, dis_percent=0.01)

agent = Agent(0.5, [0, 0], 0, 10, 0.5)

agent.opinion

grid._diffuse_opinion(agent, grid.num_DiffusionAtOnce)

##### Modified adding cluster and hhi part

class NSI:
    def __init__(self) -> None:
        self.popl = Population(50, False, True, False, 0.5, 0.25)
    def update(self) -> None:
        pos1, cell1 = random.choice(list(self.popl.grid.items()))
        neighbors = cell1.getNeighbors()
        pos2 = random.choice(neighbors)
        cell2 = self.popl.grid[pos2]
        x1 = cell1.getOpinion()
        x2 = cell2.getOpinion()

        if not cell1.checkDissenter():
            if (abs(x1 - x2) < self.popl.confidence_threshold - cell1.tolerance * 2):
                x1_new = self.popl.getNextOpinion(cell1)
                x2_new = self.popl.getNextOpinion(cell2)
                # x1_new = x1 + self.popl.learning_rate * (x2 - x1)
                # x2_new = x2 + self.popl.learning_rate * (x1 - x2)
                cell1.setOpinion(round(x1_new, 2))
                cell2.setOpinion(round(x2_new, 2))
                cell1.setDelta(self.popl.getNextDelta(cell1))
                cell2.setDelta(self.popl.getNextDelta(cell2))
        elif cell1.checkDissenter():
            if (abs(x1 - x2) < self.popl.confidence_threshold - cell1.tolerance * 2):
                x1_new = self.popl.getNextOpinion(cell1)
                x2_new = self.popl.getNextOpinion(cell2)
                cell1.setOpinion(round(x1_new, 2))
                cell2.setOpinion(round(x2_new, 2))
                cell1.setDelta(self.popl.getNextDelta(cell1))
                cell2.setDelta(self.popl.getNextDelta(cell2))
            # Negative Influence:
            elif (abs(x1 - x2) > self.popl.confidence_threshold + cell1.tolerance * 2):
                if (x1 > x2):
                    x1_new = (x1 + self.popl.learning_rate * (x1 - x2) * (1 - x1))
                    x2_new = (x2 + self.popl.learning_rate * (x2 - x1) * x2)
                    cell1.setOpinion(round(x1_new, 2))
                    cell2.setOpinion(round(x2_new, 2))
                    cell1.setDelta(self.popl.getNextDelta(cell1))
                    cell2.setDelta(self.popl.getNextDelta(cell2))
                else:
                    x1_new = (x1 + self.popl.learning_rate * (x1 - x2) * x1)
                    x2_new = (x2 + self.popl.learning_rate * (x2 - x1) * (1 - x2))
                    cell1.setOpinion(round(x1_new, 2))
                    cell2.setOpinion(round(x2_new, 2))
                    cell1.setDelta(self.popl.getNextDelta(cell1))
                    cell2.setDelta(self.popl.getNextDelta(cell2))

    def simulate(self, timeSteps) -> None:
        for t in range(timeSteps):
            self.update()





# Using delta as a value between 0 and 1 instead of an integer

#### NSI class copy for testing , while working on population class
#### in population class added one more argument i.e parameters , add that to
#### to test whether anything is broken due to changes

class NSI:
    def __init__(self) -> None:
        self.popl = Population(parameters, 50, False, True, False, 0.5, 0.25)
    def update(self) -> None:
        pos1, cell1 = random.choice(list(self.popl.grid.items()))
        neighbors = cell1.getNeighbors()
        pos2 = random.choice(neighbors)
        cell2 = self.popl.grid[pos2]
        x1 = cell1.getOpinion()
        x2 = cell2.getOpinion()

        if not cell1.checkDissenter():
            if (abs(x1 - x2) < self.popl.confidence_threshold - cell1.tolerance * 2):
                x1_new = self.popl.getNextOpinion(cell1)
                x2_new = self.popl.getNextOpinion(cell2)
                # x1_new = x1 + self.popl.learning_rate * (x2 - x1)
                # x2_new = x2 + self.popl.learning_rate * (x1 - x2)
                cell1.setOpinion(round(x1_new, 2))
                cell2.setOpinion(round(x2_new, 2))
                cell1.setDelta(self.popl.getNextDelta(cell1))
                cell2.setDelta(self.popl.getNextDelta(cell2))
        elif cell1.checkDissenter():
            if (abs(x1 - x2) < self.popl.confidence_threshold - cell1.tolerance * 2):
                x1_new = self.popl.getNextOpinion(cell1)
                x2_new = self.popl.getNextOpinion(cell2)
                cell1.setOpinion(round(x1_new, 2))
                cell2.setOpinion(round(x2_new, 2))
                cell1.setDelta(self.popl.getNextDelta(cell1))
                cell2.setDelta(self.popl.getNextDelta(cell2))
            # Negative Influence:
            elif (abs(x1 - x2) > self.popl.confidence_threshold + cell1.tolerance * 2):
                if (x1 > x2):
                    x1_new = (x1 + self.popl.learning_rate * (x1 - x2) * (1 - x1))
                    x2_new = (x2 + self.popl.learning_rate * (x2 - x1) * x2)
                    cell1.setOpinion(round(x1_new, 2))
                    cell2.setOpinion(round(x2_new, 2))
                    cell1.setDelta(self.popl.getNextDelta(cell1))
                    cell2.setDelta(self.popl.getNextDelta(cell2))
                else:
                    x1_new = (x1 + self.popl.learning_rate * (x1 - x2) * x1)
                    x2_new = (x2 + self.popl.learning_rate * (x2 - x1) * (1 - x2))
                    cell1.setOpinion(round(x1_new, 2))
                    cell2.setOpinion(round(x2_new, 2))
                    cell1.setDelta(self.popl.getNextDelta(cell1))
                    cell2.setDelta(self.popl.getNextDelta(cell2))

    def simulate(self, timeSteps) -> None:
        for t in range(timeSteps):
            self.update()

# Using delta as a value between 0 and 1 instead of an integer

def plotHeatMap(vis) -> None:
    plt.clf()
    plt.imshow(vis, origin='lower')
    plt.show()

import collections

########################### NEW ADD ############################
def get_clusters(final_opinions_ls):

    lastevolutions_opinions = final_opinions_ls
    count_lastevolutions_opinions = collections.Counter(lastevolutions_opinions)

    return count_lastevolutions_opinions

def metrics(final_opinions_ls, population_size):

    cluster_dict = get_clusters(final_opinions_ls)

    cluster_sizes = list(cluster_dict.values())
    cluster_sizes = [i for i in cluster_sizes if i>5]  # >5 agents forms a cluster, this is an assumption
    hhi = sum([(i/population_size)**2 for i in cluster_sizes])

    return np.round(hhi, 3)


def print_metrics(final_opinions_ls, population_size):

    hhi = metrics(final_opinions_ls, population_size)

    print("#"*10 + "    METRICS     " + "#"*10)
    print("\n")
    print("HHI" + "     ===>  ", hhi)
    print("\n")
    print("#"*38)

    ################################################################

#### Modified code adding hhi and cluster part

def main() -> None:
    n = NSI()
    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]

    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = n.popl.grid.get((row, col)).getOpinion()

    ##### NEW code ###########
    final_opinions_ls = []
    for row in range(len(vis)):
      final_opinions_ls.extend(vis[row])

    size_population = n.popl.population_size
    hhi = metrics(final_opinions_ls, size_population)
    print_metrics(final_opinions_ls, size_population)


    #####################

    print(np.matrix(vis))
    plotHeatMap(vis)
    n.simulate(100000)
    print("\n\n")

    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = n.popl.grid.get((row, col)).getOpinion()
    print(np.matrix(vis))
    plotHeatMap(vis)

if __name__ == "__main__":
    main()

#### original main() for testing

def main() -> None:
    n = NSI()
    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]

    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = n.popl.grid.get((row, col)).getOpinion()
    print(np.matrix(vis))
    plotHeatMap(vis)
    n.simulate(100000)
    print("\n\n")

    vis = [[0 for x in range(n.popl.grid_size)] for y in range(n.popl.grid_size)]
    for row in range(len(vis)):
        for col in range(len(vis[0])):
            vis[row][col] = n.popl.grid.get((row, col)).getOpinion()
    print(np.matrix(vis))
    plotHeatMap(vis)

if __name__ == "__main__":
    main()



