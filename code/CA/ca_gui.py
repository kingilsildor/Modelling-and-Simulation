import numpy as np
import pandas as pd

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

def get_base_combinations(base, combination_length):
    "Generate all possible combinations of values in a given base-k for a specified width."    
    import itertools
    base_values = list(range(base))
    base_combinations = list(itertools.product(base_values, repeat=combination_length))

    return [list(combination) for combination in base_combinations]

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.rule_set_size = 0
        self.max_rule_number = 0 
        self.iterations_cycle = 10 ** 6
        self.rule_set = []
        self.all_rules = []
        self.all_inits = []
        self.config = None

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)
        
        self.make_param('initial', 1)
        self.make_param('plot', False)
        self.make_param('langton', 'table')
        self.make_param('lambda_value', 0.05)
        

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
        
        if self.initial == 0:
            initial = np.zeros(self.width)            
        elif self.initial == 1:           
            initial = np.zeros(self.width)
            initial[self.width//2] = 1
        else:
            np.random.seed(self.initial)
            initial = np.random.randint(0, self.k, size=self.width)
            
        return initial

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""
        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        
        # Change the ruleset based on langton table
        if self.langton == 'random':
            self.random_table(self.lambda_value, self.max_rule_number)
        elif self.langton == 'table':
            self.walk_through(self.lambda_value)
        else:
            self.build_rule_set()
            
        if self.plot:
            self.create_graph()
            self.plot = False
        
        

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
      
    def build_rule_set2(self, rule):
        n = self.setter_rule(rule)
        rule_in_base = decimal_to_base_k(n, self.k)
        return [0 for x in range((self.rule_set_size - len(rule_in_base)))] + rule_in_base

    def check_rule2(self, inp, rule):
        ruleset = self.build_rule_set2(rule)[::-1]
        base_k = self.k
        decimal = 0

        for num in inp:
            decimal = decimal * base_k + num

        return ruleset[int(decimal)]
    
    def make_new_gen(self, row, rule):
        new_gen_row = []
        length_row = len(row)
        
        for num in range(length_row):
            
            if num == 0:
                current_row = [row[-1]] + list(row[:2])
                new_value = self.check_rule2(current_row, rule)
                new_gen_row.append(new_value)
            elif num == length_row-1:
                current_row = list(row[length_row-2:]) + [row[0]]
                new_value = self.check_rule2(current_row, rule)
                new_gen_row.append(new_value)
            else:
                current_row = list(row[num-1:num+2])
                new_value = self.check_rule2(current_row, rule)
                new_gen_row.append(new_value)
        
        return new_gen_row
    
    def calculate_cycle_length(self):
        self.all_rules = [i for i in range(self.max_rule_number)]
        system_length = self.r*2 + 1

        rows = get_base_combinations(self.k, system_length)
        max_n = 10**6

        for row in rows:
            cycle_lengths = []

            for rule in range(self.max_rule_number):
        
                generations = [row]
                new_row = row
                n = 0 

                while n < max_n:
                    new_row = self.make_new_gen(new_row, rule)

                    if new_row in generations:
                        cycle_length = len(generations) - (generations.index(row))
                        cycle_lengths.append(cycle_length)
                        break
                    generations.append(new_row)
                    n+=1
                else:
                    cycle_lengths.append(0)

            self.all_inits.append(cycle_lengths)
            
    def create_dataframe(self):
        """ Based on the initial values, create a dataframe."""
        self.calculate_cycle_length()
        
        values = np.array(self.all_inits).T.mean(axis=1)
        all_std = [np.std(row) for row in np.array(self.all_inits).T]
        df = pd.DataFrame({'mean': values, 'std': all_std, 'rule': self.all_rules}) 
          
        return df        
            
    def create_graph(self):    
        import plotly.graph_objects as go
        
        df = self.create_dataframe()        

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['rule'],
            y=df['mean'],
            mode='markers',
            marker=dict(
                color=df['mean'],
                colorscale='Viridis',
                reversescale=True,
                colorbar=dict(title='Cycle length'),
                size=7 
            ),
            name='Mean Values',
            error_y=dict(
                type='data',
                array=df['std'],
                visible=True
            )
        ))

        fig.update_layout(
            title=f'Scatter plot for mean cycle length of Wolfram rules 0-255 with standard deviations and system length {self.r*2 +1}',
            scene=dict(
                xaxis_title='Rule',
                yaxis_title='Cycle length'
            ),
            width=1600,
            height=800
        )

        fig.show()        
    
    def random_table(self, langton_lambda, K):
        """ In the random-table method, lambda is interpreted as a bias on the 
        random selection of states from all possible states as we sequentially fill in the 
        transitions that make up a delta function."""
        import random
        rule_set = np.zeros(self.k ** (self.r *2 +1))
        # Generate uniform random number g in [0, 1]
        for cell in range(len(rule_set)):
            r = random.random()
            if r >= langton_lambda:
                rule_set[cell] = K
            else:
                K_minus_one = list(range(self.k))
                K_minus_one.remove(K)
                rule_set[cell] = random.choice(K_minus_one)
        self.rule_set = rule_set               
         
        
    def walk_through(self, Lambda):
        """In the table-walk-through method, we start with a delta function consisting entirely of
        transitions to the quiescent state, so that lambda = 0.0  (but  note restrictions  below)."""
        KN = self.k ** (self.r *2 +1)
        n = round(abs(Lambda * KN - KN))
        ruleset = np.zeros(KN)
        index_lijst = [index for index in range(KN)]
        
        import random
        random_indexes = set()
        
        while len(random_indexes) < n:
            random_indexes.add(random.choice(index_lijst))

        for index in random_indexes:
            ruleset[index] = np.random.randint(1, self.k)
        
        self.rule_set = ruleset
        
    
    def return_df(self):
        """When creating a dataframe out of the csv files, the first row are used as head
        To circumvent this the following function is created"""        
        df = pd.read_csv('rule_class_wolfram.csv')
        df.rename(columns={'0': 'rule', '1': 'wolfram class'}, inplace=True)

        # New row data
        new_row = {'rule': 0, 'complexity': 1}

        # Insert the new row at the beginning
        df.loc[-1] = new_row
        df.index = df.index + 1
        df = df.sort_index()
        return df
        
        
    def calculate_lambda(self):
        """Let's define the parameter lambda as follows: We choose an arbitrary state
        s in the set of all the states and designate it as the quiescent state Sq. 
        In the transition function delta, there are n transitions to this special quiescent state.
        if n contains all the states than lambda = 0. if n contains no states, than lambda = 1"""
        
        df = self.return_df()
        
        # Count the number of rules in delta that produce this particular quiescent state, and call it n
        for rule, _ in df.iterrows():
            self.rule = rule
            self.build_rule_set()
            
            n = sum(self.rule_set)  
            lambda_delta = round(((self.rule_set_size - n) / self.rule_set_size), 2)   
            # print(rule, lambda_delta)
            df.loc[rule, 'lambda parameter'] = lambda_delta

        import plotly.express as px
        x_axis_start = -1
        x_axis_step_size = 10

        fig = px.scatter(df, y="lambda parameter", x="rule", color="wolfram class")
        fig.update_traces(marker_size=10)
        fig.update_xaxes(range=[x_axis_start, df['rule'].max() + 1], dtick=x_axis_step_size)
        fig.update_layout(coloraxis_colorbar=dict(dtick=1))
        fig.show()
    

    def entropy_graph(self):
        """The experiment entailed the average single-cell Shannon entropy for lambda with the table-walk-through method 
        for the ruleset and the parameters lambda 0.1 until 0.9 with steps of 0.02, k=2, r=1, matrix shape 64x64 excluding 
        the initial state and 10 different random initial states."""
        all_runs = []

        for _ in range(10):
            initial_row = np.random.randint(0, self.k, size=64)

            for lambda_value in range(10, 91, 2):
                lambda_value /= 100
                self.walk_through(lambda_value)
                rule = sum(bit * (2 ** i) for i, bit in enumerate(self.rule_set[::-1])) # This is to convert binary list back to rule number
                matrix = [self.make_new_gen(initial_row, rule)]

                for _ in range(63):
                    matrix.append(self.make_new_gen(matrix[-1], rule))
                
                entropies = []
                for cell_states in np.array(matrix).T:
                    rest, state_counts = np.unique(cell_states, return_counts=True)
                    probs = state_counts / len(cell_states)
                    
                    average_entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probs]) # Shannon entropy formula
                    entropies.append(average_entropy)

                all_runs.append([lambda_value, np.mean(entropies)])    
        
        df = pd.DataFrame(all_runs, columns=['Lambda', 'Average_H'])

        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Lambda'],
            y=df['Average_H'],
            mode='markers',
            marker=dict(size=10, color='#FF4500')
        ))

        fig.update_layout(
            title='Average Single Cell Entropy (H) over Lambda Space for 25600 CA generations',
            xaxis=dict(title='Lambda'),
            yaxis=dict(title='Average H'),
        )

        # Show the plot
        fig.show()

    
if __name__ == '__main__':
    sim = CASim()    
    sim.entropy_graph()
    sim.calculate_lambda()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()