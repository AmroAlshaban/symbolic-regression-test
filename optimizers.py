import numpy as np
import sympy as sp
from typing import Union, List, Callable


def gradient_descent(function: Union[sp.Expr, sp.Symbol, Callable], alpha: float, parameters: List[sp.Symbol], iterations: int=100, guess=None, h=10**(-12)):    
    if not guess:
        guess = np.array([1 for i in range(len(parameters))])

    if type(function) not in [sp.Expr, sp.Symbol]:
        for i in range(iterations):
            grad_values = gradient_approximation(function, at=guess, s=h)
            guess = guess - alpha * grad_values

        return guess

    gradient = [sp.diff(function, variable) for variable in parameters]
    gradient_ = sp.lambdify(parameters, gradient, 'numpy')
    
    for i in range(iterations):
        grad_values = gradient_(*guess)
        guess = guess - alpha * np.array(grad_values)

    return guess


def gradient_approximation(function: Union[sp.Expr, sp.Symbol, Callable], at=None, s=10**(-12)):
    if type(function) in [sp.Expr, sp.Symbol]:
        function = sp.lambdify(list(function.free_symbols), function)
    
    number_of_parameters = len(inspect.signature(function).parameters)
    if at is None:
        at = [1 for _ in range(number_of_parameters)]

    gradient = []
    for i in range(number_of_parameters):
        at_s = at[:]
        at_s[i] = at_s[i] + s
        value = np.round((function(*at_s) - function(*at))/s, 2)
        gradient.append(value)

    return np.array(gradient)