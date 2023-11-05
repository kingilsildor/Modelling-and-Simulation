import numpy as np
import itertools

from pyics import Model


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""

    digits = []
    while n > 0:
        digits.append(n%k)
        n //= k

    return digits[::-1]

def get_base_combinations(base, combination_length):
    "Generate all possible combinations of values in a given base-k for a specified width."
    base_values = list(range(base))
    base_combinations = list(itertools.product(base_values, repeat=combination_length))

    return [list(combination) for combination in base_combinations]


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
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
        self.make_param('plot', True)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""
        
        base_arr = decimal_to_base_k(self.rule, self.k)
        rule_arr = np.zeros(self.k ** (2 * self.r + 1))
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
        if self.initial == 1:           
            initial  = np.zeros(self.width)
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
        rule_set_size = self.k ** (2 * self.r + 1)
        rule_in_base = decimal_to_base_k(n, self.k)
        return [0 for x in range((rule_set_size - len(rule_in_base)))] + rule_in_base

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
            if not num:
                current_row = [row[length_row-1]] + row[:length_row-2]
                new_value = self.check_rule2(current_row, rule)
                new_gen_row.append(new_value)
            elif num == length_row-1:
                current_row = row[length_row-2:length_row-1] + row[:length_row-3]
                new_value = self.check_rule2(current_row, rule)
                new_gen_row.append(new_value)
            else:
                new_value = self.check_rule2(row, rule)
                new_gen_row.append(new_value)
        
        return new_gen_row
    
    def create_graph(self):
        import pandas as pd

        rules = [i for i in range(256)]
        rules = self.k ** self.k ** (self.r * 2 + 1)
        system_length = self.r*2 +1

        rows = get_base_combinations(self.k, system_length)
        max_n = 10**6
        all_inits = []

        for row in rows:
            cycle_lengths = []

            for rule in range(256):
        
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

            all_inits.append(cycle_lengths)
        
        values = np.array(all_inits).T.mean(axis=1)
        all_std = [np.std(row) for row in np.array(all_inits).T]
        df = pd.DataFrame({'mean': values, 'std': all_std, 'rule': rules})
            
        
        import plotly.graph_objects as go

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
            title=f'Scatter plot for mean cycle length of Wolfram rules 0-255 with standard deviations and system length {system_length}',
            scene=dict(
                xaxis_title='Rule',
                yaxis_title='Cycle length'
            ),
            width=1600,
            height=800
        )

        fig.show()


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
