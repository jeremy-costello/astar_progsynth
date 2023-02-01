import sys, time, math, copy
from dsl import *
from tqdm import tqdm
from a_star import a_star

class prog_search:
    def __init__(self, bound, grammar_constants, evaluator):
        # selfs
        self.bound = bound
        self.grammar_constants = grammar_constants
        self.evaluator = evaluator
    
    def initialize_hyperparameters(self, performance_type, performance_log_base,
                                   regularization_divisor, regularization_power,
                                   floor_individually, compare_normalized_costs):
        assert performance_type in ['zero', 'one']
        self.performance_type = performance_type
        self.performance_log_base = performance_log_base
        self.regularization_divisor = regularization_divisor
        self.regularization_power = regularization_power
        self.floor_individually = floor_individually
        self.compare_normalized_costs = compare_normalized_costs
    
    def search(self):
        # best average number of steps
        self.best_avg_steps = float('inf')
        self.best_avg_steps_upon_last_reset = float('inf')
        
        # both plists
        self.plist = {}
        self.plist[1] = self.grammar_constants
        
        # check which programs have been evaulated
        self.progs_evaled = set()
                
        # the main search
        self.current_size = 0
        self.reset_size = False
        while self.current_size < self.bound:
            
            self.current_size += 1
            print(f'SIZE: {self.current_size}')
            
            current_plist = copy.deepcopy(self.plist)
            self.get_valid_program_sizes(current_plist)
            prog_generator = self.generate_new_programs(current_plist)
            for new_heuristic in tqdm(prog_generator, total=self.total_new_programs):
                if new_heuristic.toString() not in self.progs_evaled:
                    self.progs_evaled.add(new_heuristic.toString())
                    avg_steps, add_to_plist, num_iters = self.evaluator.run_a_star(new_heuristic,
                                                                                   compare_normalized_costs=self.compare_normalized_costs)
                    if add_to_plist:
                        '''
                        this mod_size is probably the most important thing
                        it determines what cost each program is given
                        there is a linear and exponential version currently
                        linear: (* x) is the hyperparameter
                            [x is the iteration split - e.g. 5 means it is split into 20% tiers]
                            higher x means higher costs
                            lower x means lower costs
                        exponential: (, x) is the hyperparameter
                            [x is the base of the logarithm]
                            higher x means higher costs
                            lower x means lower costs
                            + epsilon used to be outside of brackets
                                this would cause costs of 0
                                maybe this worked better?
                            changed to add 1 to top and bottom
                        '''
                        ### CAN FLOOR BOTH INDIVIDUALLY, OR AT THE END
                        #performance_size = (self.evaluator.num_pairs - num_iters) * 5 // self.evaluator.num_pairs + 1
                        if self.performance_type == 'zero':
                            performance_size = math.log(num_iters / self.evaluator.num_pairs + 1e-4, self.performance_log_base)
                        elif self.performance_type == 'one':
                            performance_size = math.log((num_iters + 1) / (self.evaluator.num_pairs + 1), self.performance_log_base)
                        
                        # regularization - higher size for longer programs
                        # can change // x to some other value
                        true_heuristic_size = new_heuristic.getSize()
                        #regularization_size = 0
                        #regularization_size = mod_size + true_heuristic_size // 10
                        regularization_size = (true_heuristic_size / self.regularization_divisor)**self.regularization_power
                        if self.floor_individually:
                            performance_size = performance_size // 1
                            regularization_size = regularization_size // 1
                        mod_size = (performance_size + regularization_size + 1) // 1
                        # make it so size can't exceed real size
                        mod_size = min(mod_size, true_heuristic_size)
                        
                        # make sure mod_size is 1 or greater
                        if mod_size < 0: # 1
                            print('INVALID MOD SIZE')
                            sys.exit()
                        
                        # debug check if abs(state_y - goal_y) is size 1
                        if new_heuristic.toString() in ['abs((state_y - goal_y))',
                                                        'abs((goal_y - state_y))']:
                            print('YYYYYYYYYYYYYYYYYYYYYYYYYYYY')
                            print(f'MOD SIZE: {mod_size}')
                            print('YYYYYYYYYYYYYYYYYYYYYYYYYYYY')
                        
                        if mod_size not in self.plist.keys():
                            self.plist[mod_size] = []
                        self.plist[mod_size].append(new_heuristic)       
                        '''
                        if mod_size < self.current_size:
                            self.reset_size = True
                        '''
                    if avg_steps < self.best_avg_steps:
                        # changed reset to when a new best is found
                        # not only when a lower cost program is generated
                        ### could only reset when the decrease in steps
                        ### is a certain amount - such as 5% ???
                        ### compare to best upon last reset, not current best
                        ### could be a bunch of incremental <5% increases
                        if avg_steps / self.best_avg_steps_upon_last_reset <= 0.95:
                            self.reset_size = True
                        
                        # update best program
                        print('\nNEW BEST FOUND:')
                        print(new_heuristic.toString())
                        print(avg_steps)
                        self.best_avg_steps = avg_steps
                        self.best_program = new_heuristic
            
            # WHETHER THIS IS AT START OR END MATTERS
            if self.reset_size:
                self.reset_size = False
                self.current_size = 0
                self.best_avg_steps_upon_last_reset = self.best_avg_steps
    
    def get_valid_program_sizes(self, prog_dict):
        ### lists
        # plus, minus, times, maximum, minimum
        valid_sizes_set = set()
        valid_sizes_list = []
        
        ### valid program sizes
        for size1 in prog_dict.keys():
            for size2 in prog_dict.keys():
                if size1 + size2 + 1 == self.current_size:
                    new_valid_size = tuple(sorted((size1, size2)))
                    if new_valid_size not in valid_sizes_set:
                        valid_sizes_set.add(new_valid_size)
                        valid_sizes_list.append(new_valid_size)
        
        total_new_programs = 0
        for (size1, size2) in valid_sizes_list:
            total_new_programs += len(prog_dict[size1]) * len(prog_dict[size2]) * 11
        
        self.valid_sizes_list = valid_sizes_list
        self.total_new_programs = total_new_programs
    
    def generate_new_programs(self, prog_dict):
        ### generate programs
        # plus, minus, times, maximum, minimum
        for (size1, size2) in self.valid_sizes_list:
            for s1 in prog_dict[size1]:
                for s2 in prog_dict[size2]:
                    # plus
                    yield Plus(s1, s2)
                    yield Abs(Plus(s1, s2))
                    # minus
                    yield Minus(s1, s2)
                    yield Minus(s2, s1)
                    yield Abs(Minus(s1, s2))
                    # times
                    yield Times(s1, s2)
                    yield Abs(Times(s1, s2))
                    # max
                    yield Max(s1, s2)
                    yield Abs(Max(s1, s2))
                    # min
                    yield Min(s1, s2)
                    yield Abs(Min(s1, s2))