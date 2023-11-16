import numpy as np
import pandas as pd
import plotly.express as px
from pyics import Model

def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""    
    if n == 0:
        return [0]
    
    digits = []
    while n > 0:
        digits.append(n % k)
        n //= k

    return digits[::-1]

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.config = None
        self.car_flow = 0
        self.car_dict = {}

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 1000)
        self.make_param('rule', 184, setter=self.setter_rule)        
        self.make_param('density', 0.40)        

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""        
        self.rule_set_size = self.k ** (2 * self.r + 1)
        self.max_rule_number = self.k ** self.rule_set_size
        return max(0, min(val, self.max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""        
        base_arr = decimal_to_base_k(self.rule, self.k)
        rule_arr = np.zeros(self.rule_set_size)
        rule_arr[-len(base_arr):] = base_arr
        self.rule_set = rule_arr
        
    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""        
        reversed_ruleset = self.rule_set[::-1]
        base_index = 0

        for num in inp:
            base_index = base_index * self.k + num
        return reversed_ruleset[int(base_index)]

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""        
        initial = None
        if ( 0 <= self.density <= 1):
            initial = np.random.choice(self.k, size=self.width, p=[1 - self.density, self.density])        
        else:
            np.random.seed(self.density)
            initial = np.random.randint(0, self.k, size=self.width)
        return initial

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""
        self.t = 0
        self.car_flow = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()  
                  
    def draw(self):
        """Draws the current state of the grid."""
        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True     

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)
            
        """ Calculate the amount of cars that cross the right-hand side
        system boundary per unit of time."""    
        if (self.config[self.t, 0] == 0 and self.config[self.t, self.width - 1] == 1):
            self.car_flow += 1
    
def run_simulation_for_density(sim, density, sim_amount=30, N=50, T=1000):
    """Runs a simulation with specified density on a traffic simulation object
    for a given time (T) and road length (N), resetting the simulation and 
    returning the resulting car flow."""
    sim.density = density
    sim.width = N
    sim.reset()
    
    average_density = []
    for _ in range(sim_amount): 
        for _ in range(T):
            sim.step()
            average_density.append(sim.car_flow)

    return np.round(np.average(average_density),0)

def plot_df(sim, sim_amount=30, N=50, T=1000):
    """Plot the car flow for different density values"""
    densities = np.arange(0, 1.05, 0.05)

    df = pd.DataFrame(columns=['density', 'car flow'])
    for density in densities:        
        df.loc[len(df)] = [round(density, 2), run_simulation_for_density(sim, density, N=N, T=T)]
    

    fig = px.scatter(df, x="density", y="car flow",
                    labels={'car flow': 'Car Flow', 'density': 'Density'},
                    title=f'Scatter plot of Car Flow with Density range [0-1] with a road length of N={N} and T={T} time steps')

    fig.update_traces(marker={'size': 15})

    fig.show()


def estimation_graph(repeat,  N=50, T=1000):
    sim.reset()
    densities = [x/100 for x in range(0,105, 5)]
    t_ranges = [count for count in range(100, 500, 100)]

    df = pd.DataFrame(columns=['density', 'car flow', 'time steps'])

    for t in t_ranges:
        print(f"Loop 1, zit nu op t range: {t}")  
        for _ in range(repeat):
            for density in densities:        
                df.loc[len(df)] = [round(density, 2), run_simulation_for_density(sim, density, N=N, T=t), t]
        
    for t in t_ranges:  
        print(f"Loop 2, zit nu op t range: {t}")  
        for dens in densities:
            values = df.loc[(df['time steps'] == t) & (df['density'] == dens)]
            values_mean = values['car flow'].mean()
            for index in values.index:
                density = df.at[index, 'density']
                df.at[index, 'critical density'] = 0 if density == 0.0 or density == 1.0 else abs(values.at[index, 'car flow'] / values_mean- 1)

    result = df[df['critical density'] <= 0.05].groupby('time steps')['critical density'].count().reset_index(name='count')
    result['probability correct'] = result['count'] / len(densities)
    result = result.drop(['count'], axis=1)
    
    print(result) 

    fig = px.scatter(result, x='time steps', y='probability correct', title='Influence time steps amount on correctness probablility')

    fig.update_layout(
        xaxis_title='Time Steps',
        yaxis_title='Probability Correct'
    )

    fig.show()
    
    	
if __name__ == '__main__':
    sim = CASim()
    plot_df(sim, N=3, T=5)
    plot_df(sim)
    estimation_graph(10)

    from pyics import GUI
    cx = GUI(sim)
    cx.start()