import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List
from scipy.optimize import root


class Regression():
    default_functions = {'linear', 'quadratic'}
    
    def __init__(self, function: Union[sp.Expr, sp.Symbol, str], x_data: Union[list, np.array, pd.DataFrame], y_data: Union[list, np.array], 
                 variables: List[sp.Symbol], parameters: Union[List[sp.Symbol]]):
        
        if isinstance(x_data, pd.DataFrame):
            x_data = x_data.values
        if np.array(x_data).shape == (len(x_data),):
            self.x_data = np.array([[t] for t in x_data])
        else:
            self.x_data = x_data
        self.y_data = np.array(y_data)
        self.parameters = parameters
        self.variables = variables
        self.free_symbols = function.free_symbols
        self.parameter_counts = len(self.parameters)
        
        if isinstance(function, str):
            if function.lower() not in self.default_functions:
                raise ValueError("Invalid function.")
            elif function.lower() == 'linear':
                self.function = Linear_Regression(self.x_data, self.y_data, self.variables, self.parameters)
            elif function.lower() == 'quadratic':
                self.function = Quadratic_Regression(self.x_data, self.y_data, self.variables, self.parameters)
        else:
            self.function = function

    def decision_function(self):
        decision = sum([(self.function.subs(zip(self.variables, self.x_data[i])) - self.y_data[i])**2 for i in range(len(self.x_data))])
        return decision

    def estimates(self, approx_method='lm'):
        decision = self.decision_function()
        equations = [sp.diff(decision, variable) for variable in self.parameters]
        equations_matrix = sp.Matrix(equations)
        try:
            solution = sp.nsolve(equations_matrix, self.parameters, [0 for x in self.parameters])
            return np.array([x[0] for x in solution.tolist()])
        except (ZeroDivisionError, ValueError):
            equations = lambda params: [sp.diff(decision, variable).subs(zip(self.parameters, params)) for variable in self.parameters]
            initial_guess = [1 for i in range(self.parameter_counts)]
            solutions = root(equations, initial_guess, method=approx_method)
            return np.array(solutions.x)

    def regression_function(self):
        sympified_estimates = sp.sympify(self.estimates().tolist())
        return self.function.subs(zip(self.parameters, sympified_estimates))
    
    def plot_regression(self):
        ...


class Linear_Regression(Regression):
    def __init__(self, x_data, y_data, variables, parameters):
        super().__init__(None, x_data, y_data, variables, parameters)


class Quadratic_Regression(Regression):
    def __init__(self, x_data, y_data, variables, parameters):
        super().__init__(None, x_data, y_data, variables, parameters)