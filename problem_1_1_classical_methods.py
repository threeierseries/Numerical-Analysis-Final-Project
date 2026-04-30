import numpy as np
import matplotlib.pyplot as plt

# Defines the right-hand side of the ODE:
def f(t, u):
    return -5*u + 5*np.cos(t) - np.sin(t)

# Defines the exact solution
# Used to compare against the numerical approximation
def exact(t):
    return np.cos(t) - np.exp(-5*t)

# Forward Euler method for solving IVP
# Inputs:
#   f = function defining u'(t) = f(t, u)
#   a = time interval left endpoint 
#   b = time interval right endpoint 
#   h = step size
#   u0 = initial condition 
# Outputs:
#   t = time point array
#   w = Array of Forward Euler approximations at each time point
def forward_euler(f, a, b, h, u0):

    # Compute the number of time steps
    N = int((b - a) / h)

    # Create the time grid
    t = np.linspace(a, b, N + 1)

    # Initialize the numerical solution array w[n] which will approximate u(t[n])
    w = np.zeros(N + 1)

    # Apply the initial condition
    w[0] = u0

    # Forward Euler iteration
    for n in range(N):
        w[n + 1] = w[n] + h * f(t[n], w[n])

    return t, w



# Computes the Forward Euler approximation with h = 0.01 and plots it together with the exact solution
def forward_euler_plot():

    # Problem interval and step size
    a = 0.0
    b = 5.0
    h = 0.01

    # Initial condition u(0) = 0
    u0 = 0.0

    # Computes the Forward Euler approximation
    t, w = forward_euler(f, a, b, h, u0)

    # Evaluate the exact solution
    u_exact = exact(t)

    # Plot the exact solution and the numerical approximation
    plt.figure(figsize=(8, 5))
    plt.plot(t, u_exact, label="Exact solution", linewidth=2)
    plt.plot(t, w, "--", label="Forward Euler, h = 0.01", linewidth=2)

    # Make the Plot Labels
    plt.xlabel("t", fontsize=14)
    plt.ylabel("u(t)", fontsize=14)
    plt.title("First-Order ODE Forward Euler Approximation", fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the figure to put in report
    plt.savefig("ode_forward_euler_h001.png", dpi=300)

    # Display
    plt.show()


# Main Guard
if __name__ == "__main__":
    forward_euler_plot()