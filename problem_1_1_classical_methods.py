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

# Classical fourth-order Runge-Kutta method for solving IVP
# Inputs:
#   f = function defining u'(t) = f(t, u)
#   a = time interval left endpoint
#   b = time interval right endpoint
#   h = step size
#   u0 = initial condition
# Outputs:
#   t = time point array
#   w = array of RK4 approximations at each time point
def runge_katta_4th_order(f, a, b, h, u0):

    # Compute the number of time steps
    N = int((b - a) / h)

    # Create the time grid t_0, t_1, ..., t_N
    t = np.linspace(a, b, N + 1)

    # Initialize the numerical solution array w[n] which will approximate u(t[n])
    w = np.zeros(N + 1)

    # Apply the initial condition
    w[0] = u0

    # RK4 iteration
    for n in range(N):
        k1 = f(t[n], w[n])
        k2 = f(t[n] + h/2, w[n] + (h/2)*k1)
        k3 = f(t[n] + h/2, w[n] + (h/2)*k2)
        k4 = f(t[n] + h, w[n] + h*k3)

        w[n + 1] = w[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, w


# Computes the RK4 approximation with h = 0.01 and plots it with the exact solution
def runge_katta_4th_order_plot():

    # Problem interval and step size
    a = 0.0
    b = 5.0
    h = 0.01

    # Initial condition u(0) = 0
    u0 = 0.0

    # Computes the RK4 approximation
    t, w = runge_katta_4th_order(f, a, b, h, u0)

    # Evaluate the exact solution
    u_exact = exact(t)

    # Plot the exact solution and the numerical approximation
    plt.figure(figsize=(8, 5))
    plt.plot(t, u_exact, label="Exact solution", linewidth=2)
    plt.plot(t, w, "--", label="RK4, h = 0.01", linewidth=2)

    # Make the plot labels
    plt.xlabel("t", fontsize=14)
    plt.ylabel("u(t)", fontsize=14)
    plt.title("First-Order ODE RK4 Approximation", fontsize=15)
    plt.legend(fontsize=12)

    # Adjust the layout so labels and title fit nicely
    plt.tight_layout()

    # Save the figure as a PNG file for the report
    plt.savefig("ode_rk4_h001.png", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()

# Computes the global error
def global_error(t, w):
    u_exact = exact(t)
    error = np.max(np.abs(w - u_exact))
    return error


# Computes the observed convergence order
def observed_order(error_old, error_new, h_old, h_new):
    return np.log(error_old / error_new) / np.log(h_old / h_new)


# Compare global errors and observed convergence orders
def convergence_table():

    # Interval and initial condition
    a = 0.0
    b = 5.0
    u0 = 0.0

    # Step sizes 
    h_values = [0.01, 0.005, 0.001]

    # Store errors 
    euler_errors = []
    rk4_errors = []

    # Compute global errors for each h
    for h in h_values:

        # Forward Euler
        t_euler, w_euler = forward_euler(f, a, b, h, u0)
        euler_error = global_error(t_euler, w_euler)
        euler_errors.append(euler_error)

        # RK4
        t_rk4, w_rk4 = runge_katta_4th_order(f, a, b, h, u0)
        rk4_error = global_error(t_rk4, w_rk4)
        rk4_errors.append(rk4_error)

    # Print table
    print()
    print("Part 1.1(c): Global Errors and Observed Convergence Orders")
    print()
    print(f"{'h':<12} {'Euler Error':<18} {'Euler Order':<15} {'RK4 Error':<18} {'RK4 Order':<15}")
    print("-" * 80)

    for i in range(len(h_values)):
        h = h_values[i]

        if i == 0:
            euler_order = "--"
            rk4_order = "--"
        else:
            euler_order_value = observed_order(
                euler_errors[i - 1],
                euler_errors[i],
                h_values[i - 1],
                h_values[i]
            )

            rk4_order_value = observed_order(
                rk4_errors[i - 1],
                rk4_errors[i],
                h_values[i - 1],
                h_values[i]
            )

            euler_order = f"{euler_order_value:.4f}"
            rk4_order = f"{rk4_order_value:.4f}"

        print(
            f"{h:<12.5f} "
            f"{euler_errors[i]:<18.6e} "
            f"{euler_order:<15} "
            f"{rk4_errors[i]:<18.6e} "
            f"{rk4_order:<15}"
        )
        
# Main Guard
if __name__ == "__main__":
    forward_euler_plot()
    runge_katta_4th_order_plot()

    convergence_table()