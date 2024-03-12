import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import warnings
import inspect
from typing import Union, List, Callable
from optimizers import gradient_descent
from custom_errors import *


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

        if len(self.x_data) == len(self.y_data):
            self.dataset_length = len(self.x_data)
        else:
            raise ImbalanceError("x_data and y_data have different sizes.")
        
        if variables:
            self.variables = variables
        if parameters:
            self.parameters = parameters
            self.parameter_counts = len(self.parameters)
        
        if isinstance(function, str):
            if function.lower() not in self.default_functions:
                raise ValueError("Invalid function.")
            elif function.lower() == 'linear':
                self.special_function = Linear_Regression(self.x_data, self.y_data, self.variables, self.parameters)
            elif function.lower() == 'quadratic':
                self.special_function = Quadratic_Regression(self.x_data, self.y_data, self.variables, self.parameters)
            elif function.lower() == 'circular':
                self.coordinates = [[self.x_data[i][0], self.y_data[i]] for i in range(len(self.x_data))]
                self.special_function = Circular_Regression(self.coordinates)
        else:
            self.function = function
            self.free_symbols = function.free_symbols
            self.lambdified_function = sp.lambdify(self.variables + self.parameters, self.function, 'numpy')

    def decision_function(self, lambdified: bool=False):
        if lambdified:
            decision = lambda *params: sum([(self.lambdified_function(*self.x_data[i], *params) - self.y_data[i])**2 for i in range(len(self.x_data))])
            return decision
        decision = sum([(self.function.subs(zip(self.variables, self.x_data[i])) - self.y_data[i])**2 for i in range(len(self.x_data))])
        return decision

    def estimates(self, alpha=0.5, iterations=100, guess=None, use_sympy: bool=False, lambdified: bool=True):
        decision = self.decision_function(lambdified)
        message = None
        if use_sympy and lambdified:
            warnings.warn('SymPy solver got ignored; using the lambdified function overwrites the SymPy solver.')
        elif use_sympy:
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
    
    def regression_function(self, estimates):
        estimates = sp.sympify(estimates.tolist())
        return self.function.subs(zip(self.parameters, estimates))
    
    def mse(self, biased=False):
        ...
    
    def plot_regression(self, estimates, background_style=None, dataset_color='red', legend: bool=False, dataset_label=None, xlabel=None, ylabel=None, **kwargs): 
        if all([(type(x) == np.ndarray) and (len(x) == 1) for x in self.x_data]) or all([type(x) in [np.float64, np.int32] for x in self.x_data]):
            x_data = np.array([x[0] for x in self.x_data])
            regression_x_all = np.linspace(int(x_data.min()) - 2, int(x_data.max()) + 2, int(10*len(x_data)))
            regression_y_all = np.array([self.regression_function(estimates).subs(self.variables[0], x0) for x0 in regression_x_all])
            package = np.array([(regression_x_all[i], sp.re(regression_y_all[i])) for i in range(len(regression_x_all)) if abs(sp.im(regression_y_all[i])) < 1e-10])
            regression_x, regression_y = np.array([[value[0] for value in package], [value[1] for value in package]])
            
            color = kwargs.pop('color', 'blue')

            if background_style:
                plt.style.use(background_style)
            
            plt.scatter(x_data, self.y_data, label=dataset_label, color=dataset_color)
            plt.plot(regression_x, regression_y, color=color, **kwargs)
    
            if legend:
                plt.legend()

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axis('equal')
            plt.show()
            return
        raise DimensionError("x_data is not 1D.")
    

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
