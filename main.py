import time, sys
from dsl import *
from a_star import a_star
from search import prog_search

### INPUTS
# A* hyperparameters
map_name = 'isound1'
num_pairs = 10
# GBUS hyperparameters
bound = 0
performance_type = 'zero'
performance_log_base = 0.25
regularization_divisor = 10
regularization_power = 1.5
floor_individually = True
compare_normalized_costs = True
# TEST
test_more_pairs = False
show_test_graphs = False

### GRAMMAR
# grammar constants
grammar_constants = [Var('state_x'), Var('state_y'),
                     Var('goal_x'), Var('goal_y'),
                     Num(0.5), Num(2)]

### PRE-COMPUTES
map_file_location = f'dao-map/{map_name}.map'

### OUTPUTS
# generate A* evaluator
evaluator = a_star(num_pairs)
evaluator.load_map_from_file(map_file_location, show_graphs=True)
evaluator.generate_start_goal_lists()
evaluator.additional_gets()

# test if the maze and all (start,goal) pairs are connected
dx = Abs(Minus(Var('state_x'), Var('goal_x')))
dy = Abs(Minus(Var('state_y'), Var('goal_y')))
manhat_heur = Plus(dx, dy)
manhat_avg_steps, maze_works, _ = evaluator.run_a_star(manhat_heur, update_best=False, show_graphs=show_test_graphs)
print(f'MANHATTAN: {manhat_avg_steps}')

if maze_works is None:
    print('Broken maze!')
    sys.exit()

manhat_diag_heur = Plus(Max(dx, dy), Times(Num(0.5), Min(dx, dy)))
manhat_diag_avg_steps, _, _ = evaluator.run_a_star(manhat_diag_heur, update_best=False, show_graphs=show_test_graphs)
print(f'MANHATTAN DIAGONAL: {manhat_diag_avg_steps}')

### TEST PAIRS
if test_more_pairs:
    assert map_name in ['isound1', 'orz302d', 'brc501d']
    if map_name == 'isound1':
        test_heuristic = Times(Max(Abs(Minus(Var('state_x'), Var('state_y'))), dx), Plus(dy, Abs(Minus(dx, dy))))
    elif map_name == 'orz302d':
        test_heuristic = Times(Plus(dx, dy), Max(Plus(Var('state_x'), Var('state_x')), Times(Var('state_y'), Minus(Var('goal_x'), Var('state_x')))))
    elif map_name == 'brc501d':
        test_heuristic = Plus(Max(Minus(Var('goal_x'), Var('state_x')), dy), Max(Minus(Var('goal_x'), Var('state_x')), Abs(Minus(Minus(Var('goal_x'), Var('state_x')), dy))))
    print(map_name)
    print(test_heuristic.toString())
    test_heur_avg_steps, _, _ = evaluator.run_a_star(test_heuristic, update_best=False, show_graphs=show_test_graphs)
    print(f"Best heuristic synthesized for map '{map_name}': {test_heur_avg_steps}")

# sleep for tqdm
time.sleep(1)

# run bottom-up search
synthesizer = prog_search(bound, grammar_constants, evaluator)
synthesizer.initialize_hyperparameters(performance_type, performance_log_base,
                                       regularization_divisor, regularization_power,
                                       floor_individually, compare_normalized_costs)
synthesizer.search()

# evaulate best program
_, _, _ = evaluator.run_a_star(synthesizer.best_program, update_best=False, show_graphs=True)