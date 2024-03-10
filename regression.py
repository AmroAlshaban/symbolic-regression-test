import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from typing import Union, List
import warnings


class Regression():
    default_functions = {'linear', 'quadratic', 'circular'}
    
    def __init__(self, function: Union[sp.Expr, sp.Symbol, str], x_data: Union[list, np.array, pd.DataFrame], y_data: Union[list, np.array], 
                 variables: List[sp.Symbol]=None, parameters: Union[List[sp.Symbol]]=None):
        
        if isinstance(x_data, pd.DataFrame):
            x_data = x_data.values
        if np.array(x_data).shape == (len(x_data),):
            x_data = np.array([[t] for t in x_data])

        self.x_data = x_data
        self.y_data = np.array(y_data)
        if variables:
            self.variables = variables
        if parameters:
            self.parameters = parameters
            self.parameter_counts = len(self.parameters)
        
        if isinstance(function, str):
            if function.lower() not in self.default_functions:
                raise ValueError("Invalid function.")
            elif function.lower() == 'linear':
                self.function = Linear_Regression(self.x_data, self.y_data, self.variables, self.parameters)
            elif function.lower() == 'quadratic':
                self.function = Quadratic_Regression(self.x_data, self.y_data, self.variables, self.parameters)
            elif function.lower() == 'circular':
                self.coordinates = [[self.x_data[i][0], self.y_data[i]] for i in range(len(self.x_data))]
                self.function = Circular_Regression(self.coordinates)
        else:
            self.function = function
            self.free_symbols = function.free_symbols

    def decision_function(self):
        decision = sum([(self.function.subs(zip(self.variables, self.x_data[i])) - self.y_data[i])**2 for i in range(len(self.x_data))])
        return decision

    def estimates(self, alpha=0.5, iterations=100, guess=None, use_sympy: bool=False):
        decision = self.decision_function()
        message = None
        if use_sympy:
            try:
                equations = [sp.diff(decision, variable) for variable in self.parameters]
                equations_matrix = sp.Matrix(equations)
                solution = sp.nsolve(equations_matrix, self.parameters, [0 for x in self.parameters])
                return np.array([x[0] for x in solution.tolist()])
            except Exception as e:
                message = "No Exact Solution"
        if message:
            warnings.warn(message, UserWarning)
        
        return gradient_descent(function=decision, parameters=self.parameters, alpha=alpha, iterations=iterations, guess=guess)

    def regression_function(self):
        sympified_estimates = sp.sympify(self.estimates().tolist())
        return self.function.subs(zip(self.parameters, sympified_estimates))

    def mse(self):
        ...
    
    def plot_regression():
        ...
    

class Linear_Regression(Regression):
    def __init__(self, x_data, y_data, variables, parameters):
        super().__init__(None, x_data, y_data, variables, parameters)


class Quadratic_Regression(Regression):
    def __init__(self, x_data, y_data, variables, parameters):
        super().__init__(None, x_data, y_data, variables, parameters)


class Circular_Regression():
    def __init__(self, coordinates: Union[List[float], np.array, pd.DataFrame]):
        if isinstance(coordinates, pd.DataFrame):
            coordinates = coordinates.values
        self.coordinates = np.array(coordinates)
        self.x_center, self.y_center, self.radius = sp.symbols('x_center y_center radius')
        self.data_plotter = pd.DataFrame(self.coordinates)
    
    def decision_function(self):
        return sum([(sp.sqrt((self.coordinates[i][0] - self.x_center)**2 + (self.coordinates[i][1] - self.y_center)**2) - self.radius)**2
                    for i in range(len(self.coordinates))])

    def estimates(self, alpha=0.5, iterations=100, guess=None):
        decision = self.decision_function()
        return gradient_descent(function=decision, parameters=[self.x_center, self.y_center, self.radius], alpha=alpha, iterations=iterations, guess=guess)


    def mse(self):
        ...

    def regression_function(self):
        ...
    
    def plot_regression(self, estimates, dataset_color='red', legend: bool=False, dataset_label=None, xlabel=None, ylabel=None, **kwargs):
        x_center, y_center, radius = estimates
        regression_x = np.linspace(int(x_center - radius) - 2, int(x_center + radius) + 2, int(100*radius))
        regression_y_upper = np.array([y_center + np.sqrt(radius**2 - (element - x_center)**2) for element in regression_x])
        regression_y_lower = np.array([y_center - np.sqrt(radius**2 - (element - x_center)**2) for element in regression_x])

        color = kwargs.pop('color', 'blue')
        
        plt.scatter(self.data_plotter[0], self.data_plotter[1], label=dataset_label, color=dataset_color)
        plt.plot(regression_x, regression_y_upper, color=color, **kwargs)
        plt.plot(regression_x, regression_y_lower, color=color, **kwargs)

        if legend:
            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.show()


def gradient_descent(function: Union[sp.Expr, sp.Symbol], parameters: List[sp.Symbol], alpha: float, iterations: int=100, guess=None):
    if not guess:
        guess = np.array([1 for i in range(len(parameters))])

    gradient = [sp.diff(function, variable) for variable in parameters]
    gradient_ = sp.lambdify(parameters, gradient, 'numpy')
    
    for i in range(iterations):
        grad_values = gradient_(*guess)
        guess = guess - alpha * np.array(grad_values)
    
    return guess
    

