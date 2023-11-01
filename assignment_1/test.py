import random

random_array = [random.randint(0, 2) for _ in range(27)]
check = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,1]

def check_rule(r, k, inp, check):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
        next_gen = []
        len_arr = len(inp)
        for index in range(len_arr):
            left_of_i = (index - r) % len_arr
            right_and_i = (index + r + 1) % len_arr
            
            if left_of_i < right_and_i:
                neighborhood = inp[left_of_i:right_and_i]
            else:
                neighborhood = inp[left_of_i:] + inp[:right_and_i]

            test = 0
            for base in range(len(neighborhood)):
                test = test + neighborhood[base] * k ** base
            next_gen.append(check[test])              
            
        return next_gen
    
check_rule(1, 2, random_array, check)


        # next_gen = []
        # print(inp)
        # len_arr = len(inp)
        # for index in range(len_arr):
        #     left_of_i = (index - self.r) % len_arr
        #     right_and_i = (index + self.r) % len_arr
            
        #     if left_of_i < right_and_i:
        #         neighborhood = inp[left_of_i:right_and_i]
        #     else:
        #         neighborhood = inp[left_of_i:] + inp[:right_and_i]

        #     rule_index = 0
        #     for base in range(len(neighborhood)):
        #         rule_index = rule_index + neighborhood[base] * self.k ** base
        #     next_gen.append(self.rule_set[rule_index])             
            
        # return next_gen