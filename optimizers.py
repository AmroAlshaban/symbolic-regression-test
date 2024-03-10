import numpy as np
import sympy as sp


def gradient_descent(function: Union[sp.Expr, sp.Symbol], parameters: List[sp.Symbol], alpha: float, iterations: int=100, guess=None):
    if not guess:
        guess = np.array([1 for i in range(len(parameters))])

    gradient = [sp.diff(function, variable) for variable in parameters]
    gradient_ = sp.lambdify(parameters, gradient, 'numpy')
    
    for i in range(iterations):
        grad_values = gradient_(*guess)
        guess = guess - alpha * np.array(grad_values)
    
    return guess
