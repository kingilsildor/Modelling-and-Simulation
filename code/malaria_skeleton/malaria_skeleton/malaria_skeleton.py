import matplotlib.pyplot as plt
import numpy as np
import malaria_visualize

class Model:
    def __init__(self, width=50, height=50, mosquitoPopDensity=0.10, humanPopDensity=0.23,
                 initMosquitoHungry=0.5, initHumanInfected=0.2,
                 humanInfectionProb=0.25, humanImmuneProb=0.001, humanReInfectionProb=0.15,
                 illnessDeathProb=0.03, illnessIncubationTime=4, illnessContagiousTime = 30,
                 mosquitoInfectionProb=0.9, mosquitoMinage = 14, mosquitoMaxage = 65,
                 mosquitoFeedingCycle=15, biteProb=1.0):
        """
        Model parameters
        Initialize the model with the width and height parameters.
        """
        self.height = height
        self.width = width
        self.nHuman = int(width * height * humanPopDensity)
        self.nMosquito = int(width * height * mosquitoPopDensity)
        self.humanInfectionProb = humanInfectionProb
        self.humanImmuneProb = humanImmuneProb
        self.humanReInfectionProb = humanReInfectionProb
        self.illnessDeathProb = illnessDeathProb
        self.illnessIncubationTime = illnessIncubationTime
        self.illnessContagiousTime = illnessContagiousTime
        self.mosquitoInfectionProb = mosquitoInfectionProb
        self.mosquitoFeedingCycle = mosquitoFeedingCycle
        self.mosquitoMinage = mosquitoMinage
        self.mosquitoMaxage = mosquitoMaxage
        self.biteProb = biteProb
        
        """
        Data parameters
        To record the evolution of the model
        """
        self.infectedCount = 0
        self.deathCount = 0
        self.resistCount = 0

        """
        Population setters
        Make a data structure in this case a list with the humans and mosquitos.
        """
        self.humanPopulation, self.humanCoordinates = self.set_human_population(initHumanInfected)
        self.mosquitoPopulation = self.set_mosquito_population(initMosquitoHungry)
    
    def set_human_population(self, initHumanInfected):
        """
        This function makes the initial human population, by iteratively adding
        an object of the Human class to the humanPopulation list.
        The position of each Human object is randomized. A number of Human
        objects is initialized with the "infected" state.
        """
        humanPopulation = []
        humanCoordinates = []
        for i in range(self.nHuman):
            x, y, state = self.create_new_human(i, humanCoordinates, initHumanInfected)
            humanPopulation.append(Human(x, y, state))
            humanCoordinates.append([x,y])

        return humanPopulation, humanCoordinates

    def create_new_human(self, i, humanCoordinates, initHumanInfected=0.2):
        """ Initial create new humans based on the starting condition of the simulation.
        Afterwards it will be used to create new humans after a human dies.
        For this i=-1 is used to better handle the random infection at birth."""
        x = np.random.randint(self.width)
        y = np.random.randint(self.height)
        
        # Humans may not have overlapping positions.
        while (x,y) in humanCoordinates:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
        
        if (i / self.nHuman) <= initHumanInfected:
            state = 'I'  # I for infected
        elif (i == -1) and np.random.uniform <= initHumanInfected:
            state = 'S'  # To handle new born babies
        # elif np.random.uniform() <= self.humanImmuneProb:
        #     state = 'R'  # R for removed / immune 
        else:
            state = 'S'  # S for susceptible
            
        return x, y, state
    
    def set_mosquito_population(self, initMosquitoHungry):
        """
        This function makes the initial mosquito population, by iteratively
        adding an object of the Mosquito class to the mosquitoPopulation list.
        The position of each Mosquito object is randomized.
        A number of Mosquito objects is initialized with the "hungry" state.
        """
        mosquitoPopulation = []
        for i in range(self.nMosquito):
            x,y, hungry = self.create_new_mosquito(i, initMosquitoHungry)
            mosquitoPopulation.append(Mosquito(x, y, hungry))
        return mosquitoPopulation
    
    def create_new_mosquito(self, i, initMosquitoHungry=0.5):
        """ Initial create new mosquito based on the starting condition of the simulation.
        Afterwards it will be used to create new mosquito after a mosquito dies.
        For this i=-1 is used to better handle the random hungry at birth."""              
        x = np.random.randint(self.width)
        y = np.random.randint(self.height)
        if (i / self.nMosquito) <= initMosquitoHungry:
            hungry = True
        elif (i == -1) and np.random.uniform <= initMosquitoHungry:
            hungry = True
        else:
            hungry = False
            
        return x, y, hungry

    def update(self):
        """
        Perform one timestep:
        1.  Update mosquito population. Move the mosquitos. If a mosquito is
            hungry it can bite a human with a probability biteProb.
            Update the hungry state of the mosquitos.
        2.  Update the human population. If a human dies remove it from the
            population, and add a replacement human.
        """
        
        def set_mosquito_hungry(m):
            """ Set the hungry state from False to True after a number of time steps has passed."""
            if not m.hungry:
                m.daysNotHungry += 1
            
            if m.daysNotHungry == self.mosquitoFeedingCycle:
                m.daysNotHungry = 0
                m.hungry = True
        
        def mosquito_live_cycle(m):
            m.age += 1
          
            if (m.age >= self.mosquitoMinage and np.random.uniform() <= m.indivualDeathProb) or\
                (m.age >= self.mosquitoMaxage):
                self.mosquitoPopulation.remove(h)
                x, y, hungry = self.create_new_mosquito(-1)
                self.mosquitoPopulation.append(Mosquito(x, y, hungry))
            else:
                m.indivualDeathProb += 0.001
        
        def new_person_born(h):
            self.humanCoordinates.remove(h.position)
            self.humanPopulation.remove(h)
            x, y, state = self.create_new_human(-1, self.humanCoordinates)
            
            self.humanPopulation.append(Human(x, y, state))
            self.humanCoordinates.append([x,y])
        
        
        for i, m in enumerate(self.mosquitoPopulation):
            m.move(self.height, self.width)
            for h in self.humanPopulation:
                if m.position == h.position and m.hungry\
                   and np.random.uniform() <= self.biteProb:
                    if m.bite(h, self.humanInfectionProb, self.humanReInfectionProb,
                           self.mosquitoInfectionProb):
                        self.infectedCount += 1
            set_mosquito_hungry(m)
            # let_mosquito_live_cycle(m)

        for j, h in enumerate(self.humanPopulation):
            if h.infected:
                h.daysInfected += 1
                    
            if (h.humanResistance(self.illnessContagiousTime)):
                self.resistCount += 1
                
            h.humanSymptoms(self.illnessIncubationTime)
            if np.random.uniform() <= self.illnessDeathProb and h.state == 'I':
                self.deathCount += 1
                new_person_born(h)
                
                
        """
        To implement: update the human population.
        """
        """
        To implement: update the data/statistics e.g. infectedCount,
                      deathCount, etc.
        """
        return self.infectedCount, self.deathCount, self.resistCount


class Mosquito:
    def __init__(self, x, y, hungry):
        """
        Class to model the mosquitos. Each mosquito is initialized with a random
        position on the grid. Mosquitos can start out hungry or not hungry.
        All mosquitos are initialized infection free (this can be modified).
        """
        self.position = [x, y]
        self.hungry = hungry
        self.daysNotHungry = 0
        self.age = 0
        self.indivualDeathProb = 0
        self.infected = False

    def bite(self, human, humanInfectionProb, humanReInfectionProb, mosquitoInfectionProb):
        """
        Function that handles the biting. If the mosquito is infected and the
        target human is susceptible, the human can be infected.
        If the mosquito is not infected and the target human is infected, the
        mosquito can be infected.
        After a mosquito bites it is no longer hungry.
        """
        humanInfected = False
        if self.infected and human.state == 'S':
            if np.random.uniform() <= humanInfectionProb:
                human.infected = True
                humanInfected = True
        if self.infected and human.state == 'R':
            if np.random.uniform() <= humanReInfectionProb:
                human.infected = True
                humanInfected = True
        elif not self.infected and human.state == 'I':
            if np.random.uniform() <= mosquitoInfectionProb:
                self.infected = True
        self.hungry = False
        
        return humanInfected

    def move(self, height, width):
        """
        Moves the mosquito one step in a random direction.
        """
        deltaX = np.random.randint(-1, 2)
        deltaY = np.random.randint(-1, 2)
        """
        The mosquitos may not leave the grid. There are two options:
                      - fixed boundaries: if the mosquito wants to move off the
                        grid choose a new valid move.
                      - periodic boundaries: implement a wrap around i.e. if
                        y+deltaY > ymax -> y = 0. This is the option currently implemented.
        """
        
        self.position[0] = (self.position[0] + deltaX) % width
        self.position[1] = (self.position[1] + deltaY) % height
        
class Human:
    def __init__(self, x, y, state):
        """
        Class to model the humans. Each human is initialized with a random
        position on the grid. Humans can start out susceptible or infected
        (or immune).
        """
        self.position = [x, y]
        self.state = state    
        self.infected = False
        self.daysInfected = 0
        self.humanInfected()
        
    def humanResistance(self, illnessContagiousTime):
        if self.daysInfected >= illnessContagiousTime:
            self.infected = False
            self.daysInfected = 0
            self.state = 'R'
            return True
        return False
            
    def humanSymptoms(self, illnessIncubationTime):
        if self.daysInfected >= illnessIncubationTime:
            self.state = 'I'
            return True
        return False
    
    def humanInfected(self):
        if self.state == 'I':
            self.infected = True
            return True
        return False


if __name__ == '__main__':
    # Simulation parameters
    fileName = 'simulation'
    # Amount of days
    timeSteps = 500
    t = 0
    plotData = True
    
    # Run a simulation for an indicated number of timesteps.
    file = open(fileName + '.csv', 'w')
    sim = Model(height=50, width=50, humanPopDensity=0.1)
    vis = malaria_visualize.Visualization(sim.height, sim.width)
    print('Starting simulation')
    while t < timeSteps:
        [d1, d2, d3] = sim.update()  # Catch the data
        line = str(t) + ',' + str(d1) + ',' + str(d2) + ',' + str(d3) + '\n'  # Separate the data with commas
        file.write(line)  # Write the data to a .csv file
        vis.update(t, sim.mosquitoPopulation, sim.humanPopulation)
        t += 1
    file.close()
    vis.persist()

    if plotData:
        # Make a plot by from the stored simulation data.
        data = np.loadtxt(fileName+'.csv', delimiter=',')
        time = data[:, 0]
        infectedCount = data[:, 1]
        deathCount = data[:, 2]
        resistCount = data[:, 3]
        plt.figure()
        plt.plot(time, infectedCount, label='infected')
        plt.plot(time, deathCount, label='deaths')
        plt.plot(time, resistCount, label='resisted')
        plt.legend()
        plt.show()
