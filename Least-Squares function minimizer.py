import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import Union, List
from scipy.optimize import root


class Regression():
    def __init__(self, function: Union[sp.Expr, sp.Symbol], x_data: Union[list, np.array], y_data: Union[list, np.array], 
                 variables: List[sp.Symbol], parameters: Union[List[sp.Symbol]]):
        self.function = function
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.parameters = parameters
        self.variables = variables
        self.free_symbols = function.free_symbols
        self.parameter_counts = len(self.parameters)
    
    def decision_function(self):
        decision = sum([(self.function.subs(zip(self.variables, self.x_data[i])) - self.y_data[i])**2 for i in range(len(self.x_data))])
        return decision
    
    def estimates(self, approx_method='lm'):
        decision = self.decision_function()
        equations = [sp.diff(decision, variable) for variable in self.parameters]
        equations_matrix = sp.Matrix(equations)
        try:
            solution = sp.nsolve(equations_matrix, self.parameters, [0 for x in self.parameters])
            return np.array(solution.tolist())
        except ZeroDivisionError:
            equations = lambda params: [sp.diff(decision, variable).subs(zip(self.parameters, params)) for variable in self.parameters]
            initial_guess = [1 for i in range(self.parameter_counts)]
            solutions = root(equations, initial_guess, method=approx_method)
            return np.array(solutions.x)

    
    def regression_function(self):
        return self.function.subs(zip(self.parameters, self.estimates()))
    
    def plot_regression(self):
        ...
    