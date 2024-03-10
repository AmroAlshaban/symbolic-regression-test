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


class Circular_Regression():
    def __init__(self, coordinates: Union[List[float], np.array, pd.DataFrame]):
        if isinstance(coordinates, pd.DataFrame):
            coordinates = coordinates.values
        self.coordinates = np.array(coordinates)
        self.x_center, self.y_center, self.radius = sp.symbols('x_center y_center radius')
    
    def decision_function(self):
        return sum([(sp.sqrt((self.coordinates[i][0] - self.x_center)**2 + (self.coordinates[i][1] - self.y_center)**2) - self.radius)**2
                    for i in range(len(self.coordinates))])

    def estimates(self):
        eq1 = lambda x_0, y_0, r: sum([np.sqrt((self.coordinates[i][0] - x_0)**2 + (self.coordinates[i][1] - y_0)**2) - r for i in range(len(self.coordinates))])
        eq2 = lambda x_0, y_0, r: sum([np.sqrt((self.coordinates[i][0] - x_0)**2 + (self.coordinates[i][1] - y_0)**2 - r) * (self.coordinates[i][0] - x_0)/np.sqrt((self.coordinates[i][0] - x_0)**2 + (self.coordinates[i][1] - y_0)**2) for i in range(len(self.coordinates))])
        eq3 = lambda x_0, y_0, r: sum([np.sqrt((self.coordinates[i][0] - x_0)**2 + (self.coordinates[i][1] - y_0)**2 - r) * (self.coordinates[i][1] - y_0)/np.sqrt((self.coordinates[i][0] - x_0)**2 + (self.coordinates[i][1] - y_0)**2) for i in range(len(self.coordinates))])

        max_iterations = 100
        j = 1
    
        while j <= max_iterations:
            equations = lambda x: [eq1(*x), eq2(*x), eq3(*x)]
            initial_guess = [1 + j, 1 - j, 1]
            solutions = root(equations, initial_guess, method='lm')
            
            if not np.isnan(solutions.x).any():
                return solutions.x
            j += 1

    def plot_regression(self, dataset_color='red', legend: bool=False, dataset_label=None, xlabel=None, ylabel=None, **kwargs):
        x_center, y_center, radius = self.estimates()
        data_plotter = pd.DataFrame(self.coordinates)
        regression_x = np.linspace(int(x_center - radius) - 2, int(x_center + radius) + 2, int(100*radius))
        regression_y_upper = np.array([y_center + np.sqrt(radius**2 - (element - x_center)**2) for element in regression_x])
        regression_y_lower = np.array([y_center - np.sqrt(radius**2 - (element - x_center)**2) for element in regression_x])

        color = kwargs.pop('color', 'blue')
        
        plt.scatter(data_plotter[0], data_plotter[1], label=dataset_label, color=dataset_color)
        plt.plot(regression_x, regression_y_upper, color=color, **kwargs)
        plt.plot(regression_x, regression_y_lower, color=color, **kwargs)

        if legend:
            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.show()
    

x_eg = np.linspace(-2, 10, 50)
y_eg_1 = np.array([5 + np.sqrt(36 - (element - 4)**2) + np.random.normal(0, 0.5) for element in x_eg]) 
y_eg_2 = np.array([5 - np.sqrt(36 - (element - 4)**2) + np.random.normal(0, 0.5) for element in x_eg]) 
data = np.array([[x_eg[i], y_eg_1[i]] for i in range(len(x_eg))] + [[x_eg[i], y_eg_2[i]] for i in range(len(x_eg))])
    
circular_regressor = Circular_Regression(data)
circular_regressor.plot_regression()