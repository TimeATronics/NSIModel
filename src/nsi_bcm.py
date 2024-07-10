import random
import numpy as np
import matplotlib.pyplot as plt

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

def plotHeatMap(vis) -> None:
    plt.clf()
    plt.imshow(vis, origin='lower')
    plt.show()

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

# Using delta as a value between 0 and 1 instead of an integer
