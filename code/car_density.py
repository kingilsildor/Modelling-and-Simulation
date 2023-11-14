import numpy as np
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
        self.r = 1
        self.k = 2
        self.width = 50
        self.rule = 184

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

    def setup_initial_row(self, percentage):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""        
        zeros = round((1-percentage) * self.width) * [0]
        ones = round(percentage * self.width) * [1]
        initial_row = zeros + ones

        import random
        random.shuffle(initial_row)
        
        return initial_row[:50]

    def make_new_gen(self, row):
        new_gen_row = []
        length_row = len(row)
        
        for num in range(length_row):
            if num == 0:
                current_row = [row[-1]] + list(row[:2])
                new_value = self.check_rule(current_row)
                new_gen_row.append(new_value)
            elif num == length_row-1:
                current_row = list(row[length_row-2:]) + [row[0]]
                new_value = self.check_rule(current_row)
                new_gen_row.append(new_value)
            else:
                current_row = list(row[num-1:num+2])
                new_value = self.check_rule(current_row)
                new_gen_row.append(new_value)
        
        return new_gen_row
                  
    def density_plot(self):
        self.setter_rule(self.rule)
        self.build_rule_set()

        density_dict = {}
        
        for percentage in range(0,105, 5):
            
            percentage /=100
            row = self.setup_initial_row(percentage)
            matrix = [row]
            
            for _ in range(self.t):
                row = self.make_new_gen(row)
                matrix.append(row)

            last_row = np.array(matrix).T[-1]
            count = 0
            status = True
            
            for space in last_row:
                if status and space and len(last_row) != sum(last_row):
                    count +=1
                    status = False
                elif not status and not space:
                    status = True
            
            density_dict[percentage] = count
   
        import pandas as pd
 
        df = pd.DataFrame.from_dict(data=density_dict, orient='index')
        
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Scatter(x=df.index, y= df.iloc[:, 0]))
        fig.show()


                
                



            
            



 
if __name__ == '__main__':
    CASim().density_plot()
    

