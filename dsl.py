import numpy as np

'''
DSL:
    plus
    minus
    times
    maximum
    minimum
    absolute value
    state_x
    state_y
    goal_x
    goal_y
    0.5
    1
    2
'''

# parent class of nodes on the tree (parts of the CFG)
class Node:
    # converts program to a string
    def toString(self):
        raise Exception('Unimplemented method')
    
    # interprets the output of a program
    # env is a dict of variable values
    def interpret(self):
        raise Exception('Unimplemented method')
    
    # computes the size of a program
    def getSize(self):
        raise Exception('Unimplemented method')

# addition
class Plus(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def toString(self):
        return "(" + self.left.toString() + " + " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) + self.right.interpret(env)
    
    def getSize(self):
        return self.left.getSize() + self.right.getSize() + 1

# subtraction
class Minus(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def toString(self):
        return "(" + self.left.toString() + " - " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) - self.right.interpret(env)
    
    def getSize(self):
        return self.left.getSize() + self.right.getSize() + 1

# multiplication
class Times(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def toString(self):
        return "(" + self.left.toString() + " * " + self.right.toString() + ")"

    def interpret(self, env):
        return self.left.interpret(env) * self.right.interpret(env)
    
    def getSize(self):
        return self.left.getSize() + self.right.getSize() + 1

# maximum
class Max(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def toString(self):
        return "max(" + self.left.toString() + ", " + self.right.toString() + ")"

    def interpret(self, env):
        return np.maximum(self.left.interpret(env), self.right.interpret(env))
    
    def getSize(self):
        return self.left.getSize() + self.right.getSize() + 1

# variable
class Min(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def toString(self):
        return "min(" + self.left.toString() + ", " + self.right.toString() + ")"

    def interpret(self, env):
        return np.minimum(self.left.interpret(env), self.right.interpret(env))
    
    def getSize(self):
        return self.left.getSize() + self.right.getSize() + 1

# absolute value
class Abs(Node):
    def __init__(self, value):
        self.value = value

    def toString(self):
        return "abs(" + self.value.toString() + ")"

    def interpret(self, env):
        return np.abs(self.value.interpret(env))
    
    def getSize(self):
        return self.value.getSize() + 1

# variable
class Var(Node):
    def __init__(self, name):
        self.name = name

    def toString(self):
        return self.name

    def interpret(self, env):
        return env[self.name]
    
    def getSize(self):
        return 1

# number
class Num(Node):
    def __init__(self, value):
        self.value = value

    def toString(self):
        return str(self.value)

    def interpret(self, env):
        return self.value
    
    def getSize(self):
        return 1