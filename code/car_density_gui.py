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

        self.t = 1000
        self.N = 50

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('rule', 184)        
     

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

    def setup_initial_row(self, density):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""        
        initial = None
        if ( 0 <= density <= 1):
            initial = np.random.choice(self.k, size=self.N, p=[1 - density, density])        
        else:
            np.random.seed(density)
            initial = np.random.randint(0, self.k, size=self.N)
        return initial
                  
    def make_new_gen(self, row):
        """ Make new gen, makes a new generation of the input row according to the build ruleset"""
        new_gen_row = []
        length_row = len(row)
        
        for num in range(length_row):
            if num == 0:
                # Infer the first element in the row. 
                current_row = list(row[length_row-self.r:]) + list(row[:self.r+1])
            elif num == length_row-1:
                # Infer the last element in the row.
                current_row = list(row[length_row-self.r-1:]) + list(row[:self.r])
            else:
                # Infer the remaining elements in the row per element.
                current_row = list(row[num-self.r:num+self.r+1])

            new_value = self.check_rule(current_row)
            new_gen_row.append(new_value)
        
        return new_gen_row
    
    def run_simulation_for_density(self, density, sim_amount=30, N=50, T=1000):
        """Runs sim_amount simulations with specified density on a traffic simulation object
        for a given time (T) and road length (N), returning the rounded average resulting car 
        flow of previous mentioned simulations."""
        self.N = N
        self.setter_rule(self.rule)
        self.build_rule_set()

        densities = []

        for _ in range(sim_amount):
            row = self.setup_initial_row(density)
            count = 0

            # Count occurrences when the last element is a car and there is an available space in the 
            # first element for placing a car next generation (doesn't include last generation since the car will not move)
            for _ in range(T):
                if row[0] == 0 and row[-1] != 0:
                    count +=1

                row = self.make_new_gen(row)
                
            densities.append(count)

        return round(np.mean(densities))
        
    def plot_df(self, sim_amount=30, N=50, T=1000):
        """Plot the car flow for different density values"""
        densities = [x/100 for x in range(0,105, 5)]

        df = pd.DataFrame(columns=['density', 'car flow'])
        for density in densities:        
            df.loc[len(df)] = [round(density, 2), self.run_simulation_for_density(density=density, N=N, T=T)]
        

        fig = px.scatter(df, x="density", y="car flow",
                        labels={'car flow': 'Car Flow', 'density': 'Density'})

        fig.update_traces(marker={'size': 15})
        fig.update_layout(font=dict(size=20))

        fig.show()


    def estimation_graph(self, repeat, N=50, T=1000):
        """ First calculates different time ranges (t), repeat amount of times for 21 different densities. 
        For each time range the critical density is calculated and compared to this density the correctness probability is calculated.
        In the end a plot is made for all time ranges and their corresponding probability of correctness"""
        densities = [x/100 for x in range(0,105, 5)]
        t_ranges = [count for count in range(10, 60)]
        group = 0
        
        df = pd.DataFrame(columns=['group', 'density', 'car flow', 'time steps'])

        for t in t_ranges:
            for _ in range(repeat):
                for density in densities:        
                    df.loc[len(df)] = [group, round(density, 2), self.run_simulation_for_density(density=density, N=N, T=t), t]
                group +=1
            print(f"T-range: {t}, is done! (last one is {t_ranges[-1]})")  

        result = pd.DataFrame(columns=['time steps', 'probability correct'])
        
        for t in t_ranges:  
            # For each time range repeat group the transition phase density is calculated (The highest car flow value is chosen since the transition begins when the car flow amount declines)
            values = df.loc[(df['time steps'] == t)]
            indexes = values.groupby('group')['car flow'].idxmax()
            max_density_values = df.loc[indexes, 'density']
            
            # Critical density is choses by chosing the highest measured car flow and the corresponding density
            critical_density = values.loc[values['car flow'].idxmax(), 'density']
            crit_min, crit_max = critical_density-0.05, critical_density+0.05

            # Beneath calculates the probability of densities being in range of the critical density
            count = 0
            for val in max_density_values:
                if val >= crit_min and val <= crit_max:
                    count +=1

            prob = count/len(max_density_values)
            result.loc[len(result)] = [t, prob]
        
        fig = px.scatter(result, x='time steps', y='probability correct', title='Influence time steps amount on correctness probablility')

        fig.update_layout(
            xaxis_title='Time Steps',
            yaxis_title='Probability Correct')

        fig.update_traces(marker={'size': 15})
        fig.show()   
            

if __name__ == '__main__':
    sim = CASim()
    sim.plot_df(N=3, T=5)
    sim.plot_df()
    sim.estimation_graph(20)

