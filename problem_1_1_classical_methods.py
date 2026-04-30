import numpy as np
import matplotlib.pyplot as plt

# Defines the right-hand side of the ODE:
def f(t, u):
    return -5*u + 5*np.cos(t) - np.sin(t)

# Defines the exact solution
# Used to compare against the numerical approximation.
def exact_solution(t):
    return np.cos(t) - np.exp(-5*t)