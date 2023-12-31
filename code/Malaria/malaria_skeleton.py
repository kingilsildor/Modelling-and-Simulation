import matplotlib.pyplot as plt
import numpy as np
import malaria_visualize

class Model:
    def __init__(self, width=50, height=50, mosquitoPopDensity=0.35, humanPopDensity=0.23,
                 initMosquitoHungry=0.5, initHumanInfected=0.2,
                 humanInfectionProb=0.90, humanImmuneProb=0.01, humanReInfectionProb=0.30,
                 illnessDeathProb=0.0354, illnessIncubationTime=4, illnessContagiousTime = 30,
                 mosquitoInfectionProb=0.65, mosquitoMinage = 21, mosquitoMaxage = 31,
                 mosquitoFeedingCycle=7, biteProb=1.0, prevention=None):
        """
        Model parameters
        Initialize the model with the width and height parameters.
        """
        self.height = height
        self.width = width
        self.nHuman = int(width * height * humanPopDensity * 12)
        self.nMosquito = int(width * height * mosquitoPopDensity * humanPopDensity * 12)
        self.humanInfectionProb = humanInfectionProb - prevention.get_prevention_probability()
        self.humanImmuneProb = humanImmuneProb
        self.humanReInfectionProb = humanReInfectionProb
        self.illnessDeathProb = illnessDeathProb - prevention.get_deathrate_probability()
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
        self.N = -1

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
        elif np.random.uniform() <= self.humanImmuneProb:
            state = 'R'  # R for removed / immune 
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
            """Mosquitos live around three weeks, this function removes a mosquito around the time he would naturally perish."""
            m.age += 1
          
            if (m.age >= self.mosquitoMinage and np.random.uniform() <= m.indivualDeathProb) or\
                (m.age >= self.mosquitoMaxage):
                self.mosquitoPopulation.remove(m)
                x, y, hungry = self.create_new_mosquito(-1)
                self.mosquitoPopulation.append(Mosquito(x, y, hungry))
            else:
                m.indivualDeathProb += 0.001
        
        def new_person_born(h):
            """When a person dies of malaria the person gets removed and a new person spawns in the grid."""
            self.humanCoordinates.remove(h.position)
            self.humanPopulation.remove(h)
            x, y, state = self.create_new_human(-1, self.humanCoordinates)
            
            self.humanPopulation.append(Human(x, y, state))
            self.humanCoordinates.append([x,y])
        
        
        """
        When a mosquito and a person are on the same coordinates there is a probability the 
        mosquito bites the person and one of the two gets infected
        """
        for i, m in enumerate(self.mosquitoPopulation):
            m.move(self.height, self.width)
            for h in self.humanPopulation:
                if m.position == h.position and m.hungry\
                   and np.random.uniform() <= self.biteProb:
                    answer, extra = m.bite(h, self.humanInfectionProb, self.humanReInfectionProb,
                           self.mosquitoInfectionProb)
                    if answer:
                        self.infectedCount += 1
                        if extra:
                            self.resistCount -= 1
                            
                        
            set_mosquito_hungry(m)
            mosquito_live_cycle(m)

        """There is a probability an infected person is getting better from malaria and gains a form of immunity."""
        for h in self.humanPopulation:
            if h.infected:
                h.daysInfected += 1
                
                if h.state != 'R' and (h.humanResistance(self.illnessContagiousTime)):
                    if h.state == 'R':
                        self.resistCount += 1
                
            h.humanSymptoms(self.illnessIncubationTime)
                            
            if np.random.uniform() <= self.illnessDeathProb/365 and h.state == 'I':
                self.deathCount += 1
                new_person_born(h)
                
                
        """
        Every week the infected people and deceased get counted.
        """
        if self.N % 7 == 0 and not self.N == 0:
            self.infectedCount = 0
            self.deathCount = 0

        self.N +=1

        return self.infectedCount/self.nHuman, self.deathCount/self.nHuman, self.resistCount/self.nHuman


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
                return humanInfected, True
        elif not self.infected and human.state == 'I':
            if np.random.uniform() <= mosquitoInfectionProb:
                self.infected = True
        self.hungry = False
        
        return humanInfected, False

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

    def __str__(self):
        return f"Human(position={self.position}, state={self.state})"
        
    def humanResistance(self, illnessContagiousTime):
        """Function to maybe gain partial immunity."""
        if self.daysInfected >= illnessContagiousTime:
            self.infected = False
            self.daysInfected = 0
            if np.random.uniform() <= 0.5:
                self.state = 'R'
            else:
                self.state = 'S'
            return True
        return False
            
    def humanSymptoms(self, illnessIncubationTime):
        """When bitting by mosquito the person doesn't become infected right away there is an incubation period."""
        if self.daysInfected >= illnessIncubationTime:
            self.state = 'I'
            return True
        return False
    
    def humanInfected(self):
        """Infects human"""
        if self.state == 'I':
            self.infected = True
            return True
        return False
      

class Prevention:
    def __init__(self, netsPercentage=0, sprayPercentage=0, windowNetsPercentage=0, vaccinePercentage=0):
        """
        Class to model the prevention methods. Each method is initialized with a use case probability.
        Humans can start out with different no, one or more methods.
        """
        self.netsProbability = 0.3 * (7.5 / 24) * netsPercentage
        self.sprayProbability = 0.35 * (3.5 / 24) * sprayPercentage
        self.windowNetsProbability = 0.65 * (3.5 / 24) * windowNetsPercentage

        self.vaccineProbability = 0.3 * vaccinePercentage

        self.totalPreventionProbability = self.netsProbability + self.sprayProbability + self.windowNetsProbability
        self.deathRateProbability = self.vaccineProbability
    
    def get_prevention_probability(self):
        """
        Method to get the prevention probability.
        """
        print(self.totalPreventionProbability)
        return self.totalPreventionProbability

    
    def get_deathrate_probability(self):
        """
        Method to get the death rate probability.
        """
        print(self.deathRateProbability)
        return self.deathRateProbability


if __name__ == '__main__':
    # Run a simulation for an indicated number of timesteps.
    t = 0
    # amount of days
    timeSteps = 365*1
    file = open('simulation_current'+'.csv', 'w')
    prevention_instance = Prevention(netsPercentage=0.54, sprayPercentage=0.41, windowNetsPercentage=0.01, vaccinePercentage=0.02)
    sim = Model(height=50, width=50, humanPopDensity=0.1, prevention=prevention_instance)
    vis = malaria_visualize.Visualization(sim.height, sim.width)
    print('Starting simulation')
    while t < timeSteps:
        [d1, d2, d3] = sim.update() # Catch the data
        if t % 7 == 0 and not t == 0:
            line = str(t/7) + ',' + str(d1) + ',' + str(d2) + ',' + str(d3) + '\n' # Separate the data with commas
            file.write(line) # Write the data to a .csv file
        vis.update(t, sim.mosquitoPopulation, sim.humanPopulation)
        t += 1
    file.close()
    vis.persist()



    simulations = [x/100 for x in range(0,110,25)]

    for i in simulations:
        # Simulation parameters
        fileName = f'simulation_{int(i*100)}'
        t = 0
        plotData = True
        population = 214028302
        
        # Run a simulation for an indicated number of timesteps.
        file = open(fileName + '.csv', 'w')
        prevention_instance = Prevention(netsPercentage=i, sprayPercentage=i, windowNetsPercentage=i, vaccinePercentage=0.02)
        sim = Model(height=50, width=50, humanPopDensity=0.1, prevention=prevention_instance)
        vis = malaria_visualize.Visualization(sim.height, sim.width)
        print('Starting simulation')
        while t < timeSteps:
            [d1, d2, d3] = sim.update()  # Catch the data
            if t % 7 == 0 and not t == 0:
                line = str(t/7) + ',' + str(d1) + ',' + str(d2) + ',' + str(d3) + '\n'  # Separate the data with commas
                file.write(line)  # Write the data to a .csv file
            vis.update(t, sim.mosquitoPopulation, sim.humanPopulation)
            t += 1
        file.close()
        vis.persist()
    
    

    if plotData:
        """Makes plot for current intervention vs no intervention"""
        # Make a plot by from the stored simulation data.
        data = np.loadtxt('simulation_0'+'.csv', delimiter=',')
        data2 = np.loadtxt('simulation_current'+'.csv', delimiter=',')
        time = data[:, 0]

        infectedCount_no = data[:, 1]
        infectedCount_with = data2[:, 1]

        resistCount_no = data[:, 3]
        resistCount_with = data2[:, 3]

        bar_width = 0.35  
        bar_positions_no = np.arange(len(time))
        bar_positions_with = bar_positions_no + bar_width

        plt.figure()
        plt.bar(bar_positions_no, infectedCount_no, width=bar_width, label='Infected no intervention')
        plt.bar(bar_positions_with, infectedCount_with, width=bar_width, label='Infected with intervention')
        plt.xlabel('Week')
        plt.ylabel('Population Nigeria')
        plt.legend()
        plt.show()


        plt.figure()
        plt.bar(bar_positions_no, resistCount_no, width=bar_width, label='Resistant no intervention')
        plt.bar(bar_positions_with, resistCount_with, width=bar_width, label='Resistant with intervention')
        plt.xlabel('Week')
        plt.ylabel('Population Nigeria')
        plt.legend()
        plt.show()

        import plotly.graph_objects as go

        """Experiment with different percentages of intervention, sleeping nets, IRS and window nets."""
        fig = go.Figure()

        for i in simulations:
            infected_count = np.loadtxt(f'simulation_{int(i*100)}'+'.csv', delimiter=',')[:,1]
            fig.add_trace(go.Bar(
                x=time,
                y=infected_count,
                name=f'Infected {int(i*100)}% intervention'
            ))

        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=list(range(len(time))), ticktext=time),
            yaxis=dict(title='Population Nigeria'),
            barmode='group',
            xaxis_tickangle=-45,
            legend=dict(title='Simulations'),
            title='Population Infections Over Weeks'
        )

        fig.show()
