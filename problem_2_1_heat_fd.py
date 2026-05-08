import numpy as np
import matplotlib.pyplot as plt

# Defines the exact solution for the heat equation
def exact_heat(x, t, nu=0.01):
    return (
        np.exp(-nu * np.pi**2 * t) * np.sin(np.pi * x)
        + 0.5 * np.exp(-9.0 * nu * np.pi**2 * t) * np.sin(3.0 * np.pi * x)
    )


# Defines the initial condition
def initial_condition(x):
    return np.sin(np.pi * x) + 0.5 * np.sin(3.0 * np.pi * x)


# Forward Euler finite-difference method for solving the heat equation
# Inputs:
#   nu = diffusion coefficient
#   Nx = number of spatial subintervals
#   T = final time
# Outputs:
#   x = spatial grid
#   t = time grid
#   U = numerical solution array, where U[n,j] approximates u(x_j,t_n)
#   dx = spatial step size
#   dt = time step size
#   r = stability ratio 
#   Nt = number of time steps
def heat_forward_euler(nu=0.01, Nx=64, T=0.5):

    # Compute the spatial step size
    dx = 1.0 / Nx

    # Choose dt so that r <= 1/2 and the final time is reached exactly
    dt_max = 0.5 * dx**2 / nu
    Nt = int(np.ceil(T / dt_max))
    dt = T / Nt

    # Compute the stability ratio
    r = nu * dt / dx**2

    # Create the spatial and time grids
    x = np.linspace(0.0, 1.0, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    # Initialize the numerical solution array
    U = np.zeros((Nt + 1, Nx + 1))

    # Apply the initial condition
    U[0, :] = initial_condition(x)

    # Apply the boundary conditions
    U[:, 0] = 0.0
    U[:, -1] = 0.0

    # Forward Euler finite-difference iteration
    for n in range(Nt):
        U[n + 1, 1:-1] = (
            U[n, 1:-1]
            + r * (U[n, 2:] - 2.0 * U[n, 1:-1] + U[n, :-2])
        )

    return x, t, U, dx, dt, r, Nt


# Computes the discrete L2 error at final time T
def heat_l2_error(x, U_final, T=0.5, nu=0.01):

    # Compute dx from the spatial grid
    dx = x[1] - x[0]

    # Evaluate the exact solution at final time
    U_exact = exact_heat(x, T, nu)

    # Compute the discrete L2 error
    error = np.sqrt(dx * np.sum((U_final - U_exact)**2))

    return error


# Plots the numerical solution as a heatmap over the (x,t) domain
def heatmap_plot():

    # Problem parameters
    nu = 0.01
    Nx = 64
    T = 0.5

    # Compute the finite-difference solution
    x, t, U, dx, dt, r, Nt = heat_forward_euler(nu, Nx, T)

    # Create meshgrid for plotting
    X, T_grid = np.meshgrid(x, t)

    # Plot the heatmap
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(X, T_grid, U, shading="auto")
    plt.colorbar(label="u(x,t)")

    # Make the plot labels
    plt.xlabel("x", fontsize=14)
    plt.ylabel("t", fontsize=14)
    plt.title("Heat Equation Forward Euler Approximation", fontsize=15)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure to put in report
    plt.savefig("heat_forward_euler_heatmap.png", dpi=300, bbox_inches="tight")

    # Display
    plt.show()


# Plots the final-time numerical solution with the exact solution
def heat_final_time_plot():

    # Problem parameters
    nu = 0.01
    Nx = 64
    T = 0.5

    # Compute the finite-difference solution
    x, t, U, dx, dt, r, Nt = heat_forward_euler(nu, Nx, T)

    # Evaluate the exact solution at final time
    U_exact = exact_heat(x, T, nu)

    # Plot the exact solution and the numerical approximation
    plt.figure(figsize=(8, 5))
    plt.plot(x, U_exact, label="Exact solution", linewidth=2)
    plt.plot(x, U[-1, :], "--", label="Forward Euler FD", linewidth=2)

    # Make the plot labels
    plt.xlabel("x", fontsize=14)
    plt.ylabel("u(x,0.5)", fontsize=14)
    plt.title("Heat Equation Solution at t = 0.5", fontsize=15)
    plt.legend(fontsize=12)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure to put in report
    plt.savefig("heat_forward_euler_final_time.png", dpi=300, bbox_inches="tight")

    # Display
    plt.show()


# Prints the grid parameters and L2 error
def heat_results_table():

    # Problem parameters
    nu = 0.01
    Nx = 64
    T = 0.5

    # Compute the finite-difference solution
    x, t, U, dx, dt, r, Nt = heat_forward_euler(nu, Nx, T)

    # Compute the L2 error at final time
    l2_error = heat_l2_error(x, U[-1, :], T, nu)

    # Print table
    print()
    print("Problem 2.1: Forward Euler Finite-Difference Heat Equation")
    print()
    print(f"{'Delta x':<18} {'Delta t':<18} {'Nt':<10} {'r':<18} {'L2 Error':<18}")
    print("-" * 85)
    print(
        f"{dx:<18.8e} "
        f"{dt:<18.8e} "
        f"{Nt:<10d} "
        f"{r:<18.8e} "
        f"{l2_error:<18.8e}"
    )


# Main Guard
if __name__ == "__main__":
    heatmap_plot()
    heat_final_time_plot()

    heat_results_table()