"""
PINN Final Project
EN 553.481/681 Numerical Analysis
"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""I put 1.1 and 2.1 on separate python files when creating this project, 
   so I import the relevant functions here for Problem 3 when we need to compare
   
"""
from problem_1_1_classical_methods import (
    f as classical_ode_rhs,
    forward_euler as classical_forward_euler,
    runge_katta_4th_order as classical_rk4,
    global_error as classical_global_error
)

from problem_2_1_heat_fd import (
    heat_forward_euler as heat_fd_forward_euler,
    heat_l2_error as heat_fd_l2_error
)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_pinn(model, loss_fn, epochs, lr=1e-3, log_every=2000):
    """Train a PINN model.
    Returns: (loss_history, wall_clock_time_seconds)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % log_every == 0:
            print(f"  Epoch {epoch}/{epochs}, Loss = {loss.item():.6e}")
    wall_time = time.time() - t_start
    print(f"  Training time: {wall_time:.1f}s")
    return loss_history, wall_time

def plot_loss_curve(loss_history, title="Training Loss"):
    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_ode_comparison(model, exact_fn, t_range=(0, 5), label="PINN"):
    t = torch.linspace(*t_range, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        u_pred = model(t).cpu().numpy().flatten()
    t_np = t.cpu().numpy().flatten()
    u_ex = exact_fn(t_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_np, u_ex, 'k-', lw=2, label='Exact')
    axes[0].plot(t_np, u_pred, 'r--', lw=1.5, label=label)
    axes[0].set_xlabel('t'); axes[0].set_ylabel('u(t)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{label} vs Exact')

    err = np.abs(u_pred - u_ex)
    axes[1].plot(t_np, err, 'b-')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|error|')
    axes[1].set_title(f'Pointwise Error (max = {err.max():.4e})')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"  Max absolute error: {err.max():.6e}")
    return err.max()

def plot_heat_comparison(model, exact_fn, label="PINN"):
    """Plot PINN vs exact for heat eq. Returns relative L2 error."""
    Ntest = 100
    x = np.linspace(0, 1, Ntest)
    t = np.linspace(0, 0.5, Ntest)
    X, T = np.meshgrid(x, t)
    xt = np.column_stack([X.ravel(), T.ravel()])
    xt_t = torch.tensor(xt, dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = model(xt_t).cpu().numpy().reshape(Ntest, Ntest)
    u_ex = exact_fn(X, T)
    err = np.abs(u_pred - u_ex)
    rel_l2 = np.sqrt(np.sum((u_pred - u_ex)**2)) / np.sqrt(np.sum(u_ex**2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    c0 = axes[0].pcolormesh(X, T, u_pred, shading='auto', cmap='viridis')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
    axes[0].set_title(f'{label} Prediction'); plt.colorbar(c0, ax=axes[0])
    c1 = axes[1].pcolormesh(X, T, u_ex, shading='auto', cmap='viridis')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
    axes[1].set_title('Exact Solution'); plt.colorbar(c1, ax=axes[1])
    c2 = axes[2].pcolormesh(X, T, err, shading='auto', cmap='hot')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
    axes[2].set_title(f'|Error| (rel L2 = {rel_l2:.4e})'); plt.colorbar(c2, ax=axes[2])
    plt.tight_layout()
    print(f"  Relative L2 error: {rel_l2:.6e}")
    return rel_l2

# =============================================================
# Loss Functions
# =============================================================

def compute_loss_ode_ad(model, Nr=500):
    """PINN loss for ODE using AUTOGRAD.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    IC:  u(0) = 0
    Loss: L = L_r + 50 L_ic
    """

    # Collocation points sampled uniformly from [0, 5]
    t_r = 5.0 * torch.rand((Nr, 1), device=device)
    t_r.requires_grad_(True)

    # Network prediction
    u = model(t_r)

    # Automatic differentiation: du_theta/dt
    du_dt = torch.autograd.grad(
        u,
        t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Residual:
    # du/dt + 5u - 5cos(t) + sin(t) = 0
    residual = du_dt + 5.0 * u - 5.0 * torch.cos(t_r) + torch.sin(t_r)
    loss_r = torch.mean(residual ** 2)

    # Initial condition loss
    t0 = torch.zeros((1, 1), device=device)
    u0 = model(t0)
    loss_ic = torch.mean(u0 ** 2)

    return loss_r + 50.0 * loss_ic


def compute_loss_ode_fdm(model, epsilon=1e-3,  Nr=500):
    """PINN loss for ODE using FINITE DIFFERENCES.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    IC:  u(0) = 0
    Derivative approximation:
        du/dt(t) ≈ [u(t + epsilon) - u(t - epsilon)] / (2 * epsilon)
    Loss: L = L_r + 50 L_ic
    """

    # Points sampled uniformly from [0, 5]
    t_r = 5.0 * torch.rand((Nr, 1), device=device)

    # Network predictions
    u_plus = model(t_r + epsilon)
    u_minus = model(t_r - epsilon)

    # Central difference approximation to du/dt
    du_dt_fd = (u_plus - u_minus) / (2.0 * epsilon)

    # Network prediction at t_r
    u = model(t_r)

    # Residual
    residual = du_dt_fd + 5.0 * u - 5.0 * torch.cos(t_r) + torch.sin(t_r)
    loss_r = torch.mean(residual ** 2)

    # Initial condition loss
    t0 = torch.zeros((1, 1), device=device)
    u0 = model(t0)
    loss_ic = torch.mean(u0 ** 2)

    return loss_r + 50.0 * loss_ic

def compute_loss_heat_ad(model, Nr=10000, Nic=200, Nbc=200):
    """PINN loss for heat equation using AUTOGRAD.

    PDE: u_t = 0.01 * u_xx  on (0,1) x (0, 0.5]
    IC:  u(x, 0) = sin(pi*x) + 0.5*sin(3*pi*x)
    BC:  u(0, t) = u(1, t) = 0
    """

    nu = 0.01

    # PDE residual points: (x,t) in (0,1) x (0,0.5]
    x_r = torch.rand((Nr, 1), device=device)
    t_r = 0.5 * torch.rand((Nr, 1), device=device)

    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    # First derivatives
    grad_u = torch.autograd.grad(
        u,
        xt_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    # Second derivative
    grad_u_x = torch.autograd.grad(
        u_x,
        xt_r,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    u_xx = grad_u_x[:, 0:1]

    # PDE residual calculated
    residual = u_t - nu * u_xx
    loss_r = torch.mean(residual ** 2)


    # Initial condition
    x_ic = torch.rand((Nic, 1), device=device)
    t_ic = torch.zeros((Nic, 1), device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)

    u_ic_exact = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3.0 * torch.pi * x_ic)

    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)


    # Boundary condition: u(0,t)=0 and u(1,t)=0
    t_bc = 0.5 * torch.rand((Nbc, 1), device=device)

    x_left = torch.zeros((Nbc, 1), device=device)
    x_right = torch.ones((Nbc, 1), device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

    return loss_r + 20.0 * loss_ic + 20.0 * loss_bc

def compute_loss_heat_fdm(model, epsilon=1e-3, Nr=10000, Nic=200, Nbc=200):
    """PINN loss for heat equation using FINITE DIFFERENCES.

    Same PDE, IC, BC as above. Approximate derivatives:

        u_t(x,t)  ≈ (u(x, t+eps) - u(x, t-eps)) / (2*eps)
        u_xx(x,t) ≈ (u(x+eps, t) - 2*u(x,t) + u(x-eps, t)) / eps^2
    """

    nu = 0.01

    # Making sure central-difference stays inside the domain
    if epsilon <= 0 or epsilon >= 0.25:
        raise ValueError("epsilon must satisfy 0 < epsilon < 0.25 for this heat equation domain.")


    # PDE residual points ensuring keep away from boundary by epsilon
    x_r = epsilon + (1.0 - 2.0 * epsilon) * torch.rand((Nr, 1), device=device)
    t_r = epsilon + (0.5 - 2.0 * epsilon) * torch.rand((Nr, 1), device=device)

    # Network evaluations for finite differences
    xt = torch.cat([x_r, t_r], dim=1)

    xt_t_plus = torch.cat([x_r, t_r + epsilon], dim=1)
    xt_t_minus = torch.cat([x_r, t_r - epsilon], dim=1)

    xt_x_plus = torch.cat([x_r + epsilon, t_r], dim=1)
    xt_x_minus = torch.cat([x_r - epsilon, t_r], dim=1)

    u = model(xt)
    u_t_plus = model(xt_t_plus)
    u_t_minus = model(xt_t_minus)
    u_x_plus = model(xt_x_plus)
    u_x_minus = model(xt_x_minus)

    # Central difference approximations
    u_t_fd = (u_t_plus - u_t_minus) / (2.0 * epsilon)
    u_xx_fd = (u_x_plus - 2.0 * u + u_x_minus) / (epsilon ** 2)

    # PDE residual
    residual = u_t_fd - nu * u_xx_fd
    loss_r = torch.mean(residual ** 2)


    # Initial condition: u(x,0) = sin(pi x) + 0.5 sin(3 pi x)
    x_ic = torch.rand((Nic, 1), device=device)
    t_ic = torch.zeros((Nic, 1), device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)

    u_ic_exact = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3.0 * torch.pi * x_ic)

    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)


    # Boundary condition: u(0,t)=0 and u(1,t)=0
    t_bc = 0.5 * torch.rand((Nbc, 1), device=device)

    x_left = torch.zeros((Nbc, 1), device=device)
    x_right = torch.ones((Nbc, 1), device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

    return loss_r + 20.0 * loss_ic + 20.0 * loss_bc


# Exact Solutions
def ode_exact_solution(t):
    """Exact solution."""
    return np.cos(t) - np.exp(-5.0 * t)

def heat_exact_solution(x, t, nu=0.01):
    """Exact solution for the heat equation."""
    return (
        np.exp(-nu * np.pi**2 * t) * np.sin(np.pi * x)
        + 0.5 * np.exp(-9.0 * nu * np.pi**2 * t) * np.sin(3.0 * np.pi * x)
    )

# Helper Output Functions
def run_ode_ad():
    """Train and output results for Problem 1.2."""
    model = PINN(input_dim=1, hidden_dim=32, num_layers=3).to(device)

    loss_history, wall_time = train_pinn(
        model,
        compute_loss_ode_ad,
        epochs=10000,
        lr=1e-3,
        log_every=2000
    )

    # Loss curve, log scale
    plot_loss_curve(loss_history, title="Problem 1.2: ODE PINN with AD")
    plt.savefig("problem_1_2_loss_ad.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PINN prediction vs exact solution and pointwise error
    max_err = plot_ode_comparison(
        model,
        ode_exact_solution,
        t_range=(0, 5),
        label="AD-PINN"
    )
    plt.savefig("problem_1_2_ode_ad_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    final_loss = loss_history[-1]

    print("\nProblem 1.2 AD-PINN Summary")
    print(f"Final training loss: {final_loss:.6e}")
    print(f"Max absolute error on 1000 test points: {max_err:.6e}")
    print(f"Wall-clock training time: {wall_time:.2f} seconds")

    return model, loss_history, final_loss, max_err, wall_time

def run_ode_fdm():
    """Train and output results for Problem 1.3."""
    model = PINN(input_dim=1, hidden_dim=32, num_layers=3).to(device)

    epsilon = 1e-3

    loss_history, wall_time = train_pinn(
        model,
        lambda m: compute_loss_ode_fdm(m, epsilon=epsilon),
        epochs=10000,
        lr=1e-3,
        log_every=2000
    )

    # Loss curve, log scale
    plot_loss_curve(loss_history, title="Problem 1.3: ODE PINN with FDM")
    plt.savefig("problem_1_3_loss_fdm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PINN prediction vs exact solution and pointwise error
    max_err = plot_ode_comparison(
        model,
        ode_exact_solution,
        t_range=(0, 5),
        label="FDM-PINN"
    )
    plt.savefig("problem_1_3_ode_fdm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    final_loss = loss_history[-1]

    print("\nProblem 1.3 FDM-PINN Summary")
    print(f"Epsilon: {epsilon:.1e}")
    print(f"Final training loss: {final_loss:.6e}")
    print(f"Max absolute error on 1000 test points: {max_err:.6e}")
    print(f"Wall-clock training time: {wall_time:.2f} seconds")

    return model, loss_history, final_loss, max_err, wall_time

def run_heat_ad():
    """Train and output results for Problem 2.2."""
    model = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)

    loss_history, wall_time = train_pinn(
        model,
        compute_loss_heat_ad,
        epochs=20000,
        lr=1e-3,
        log_every=2000
    )

    # Loss curve
    plot_loss_curve(loss_history, title="Problem 2.2: Heat PINN with AD")
    plt.savefig("problem_2_2_loss_ad.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Prediction heatmap, exact heatmap, and pointwise error heatmap
    rel_l2 = plot_heat_comparison(
        model,
        heat_exact_solution,
        label="AD-PINN"
    )
    plt.savefig("problem_2_2_heat_ad_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    final_loss = loss_history[-1]

    print("\nProblem 2.2 AD-PINN Heat Summary")
    print(f"Final training loss: {final_loss:.6e}")
    print(f"Relative L2 error on 100 x 100 test grid: {rel_l2:.6e}")
    print(f"Wall-clock training time: {wall_time:.2f} seconds")

    return model, loss_history, final_loss, rel_l2, wall_time

def run_heat_fdm():
    """Train and output results for Problem 2.3."""
    model = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)

    epsilon = 1e-3

    loss_history, wall_time = train_pinn(
        model,
        lambda m: compute_loss_heat_fdm(m, epsilon=epsilon),
        epochs=20000,
        lr=1e-3,
        log_every=2000
    )

    # Loss curve
    plot_loss_curve(loss_history, title="Problem 2.3: Heat PINN with FDM")
    plt.savefig("problem_2_3_loss_fdm.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Prediction heatmap, exact heatmap, and pointwise error heatmap
    rel_l2 = plot_heat_comparison(
        model,
        heat_exact_solution,
        label="FDM-PINN"
    )
    plt.savefig("problem_2_3_heat_fdm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    final_loss = loss_history[-1]

    print("\nProblem 2.3 FDM-PINN Heat Summary")
    print(f"Epsilon: {epsilon:.1e}")
    print(f"Final training loss: {final_loss:.6e}")
    print(f"Relative L2 error on 100 x 100 test grid: {rel_l2:.6e}")
    print(f"Wall-clock training time: {wall_time:.2f} seconds")

    return model, loss_history, final_loss, rel_l2, wall_time

def evaluate_ode_max_error(model, exact_fn, t_range=(0, 5), ntest=1000):
    """Compute max |u_theta(t) - u(t)| on ntest number of evenly spaced points."""
    t = torch.linspace(*t_range, ntest, device=device).unsqueeze(1)

    with torch.no_grad():
        u_pred = model(t).cpu().numpy().flatten()

    t_np = t.cpu().numpy().flatten()
    u_ex = exact_fn(t_np)

    max_err = np.max(np.abs(u_pred - u_ex))
    return max_err


def print_problem_14_comparison(
    ode_ad_final_loss,
    ode_ad_error,
    ode_ad_time,
    ode_fdm_final_loss,
    ode_fdm_error,
    ode_fdm_time
):
    """Print comparison table for Problem 1.4(a)."""
    print("\nProblem 1.4(a): AD-PINN vs FDM-PINN")
    print(f"{'Method':<15} {'Final Loss':<18} {'Max Abs Error':<18} {'Time (s)':<12}")
    print("-" * 70)
    print(f"{'AD-PINN':<15} {ode_ad_final_loss:<18.6e} {ode_ad_error:<18.6e} {ode_ad_time:<12.2f}")
    print(f"{'FDM-PINN':<15} {ode_fdm_final_loss:<18.6e} {ode_fdm_error:<18.6e} {ode_fdm_time:<12.2f}")


def train_ode_fdm_for_epsilon(epsilon, epochs=10000):
    """Train one FDM-PINN for a given epsilon and return final loss, max error, and time."""
    model = PINN(input_dim=1, hidden_dim=32, num_layers=3).to(device)

    loss_history, wall_time = train_pinn(
        model,
        lambda m: compute_loss_ode_fdm(m, epsilon=epsilon),
        epochs=epochs,
        lr=1e-3,
        log_every=2000
    )

    final_loss = loss_history[-1]
    max_err = evaluate_ode_max_error(
        model,
        ode_exact_solution,
        t_range=(0, 5),
        ntest=1000
    )

    return model, final_loss, max_err, wall_time


def run_problem_14_epsilon_sweep():
    """Train FDM-PINN for several epsilon values and plot max error vs epsilon."""
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    final_losses = []
    max_errors = []
    wall_times = []

    print("\nProblem 1.4(b): FDM epsilon sweep")
    print(f"{'epsilon':<12} {'Final Loss':<18} {'Max Abs Error':<18} {'Time (s)':<12}")
    print("-" * 70)

    for eps in epsilons:
        # Reset seeds so each epsilon starts from the same initial random setup
        torch.manual_seed(42)
        np.random.seed(42)

        model, final_loss, max_err, wall_time = train_ode_fdm_for_epsilon(
            epsilon=eps,
            epochs=10000
        )

        final_losses.append(final_loss)
        max_errors.append(max_err)
        wall_times.append(wall_time)

        print(f"{eps:<12.1e} {final_loss:<18.6e} {max_err:<18.6e} {wall_time:<12.2f}")

    # Plot max error vs epsilon
    plt.figure(figsize=(6, 4))
    plt.loglog(epsilons, max_errors, marker="o")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"Max absolute error")
    plt.title(r"Problem 1.4: FDM-PINN Error vs. $\epsilon$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("problem_1_4_fdm_epsilon_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save numerical results in case you want to copy them into Overleaf
    results = np.column_stack([epsilons, final_losses, max_errors, wall_times])
    np.savetxt(
        "problem_1_4_epsilon_sweep_results.txt",
        results,
        header="epsilon final_loss max_absolute_error wall_time_seconds",
        fmt="%.8e"
    )

    return epsilons, final_losses, max_errors, wall_times

def evaluate_heat_rel_l2(model, exact_fn, ntest=100):
    """Compute relative L2 error for the heat equation on a ntest x ntest grid."""
    x = np.linspace(0, 1, ntest)
    t = np.linspace(0, 0.5, ntest)
    X, T = np.meshgrid(x, t)

    xt = np.column_stack([X.ravel(), T.ravel()])
    xt_t = torch.tensor(xt, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(xt_t).cpu().numpy().reshape(ntest, ntest)

    u_ex = exact_fn(X, T)

    rel_l2 = np.sqrt(np.sum((u_pred - u_ex) ** 2)) / np.sqrt(np.sum(u_ex ** 2))
    return rel_l2

def print_problem_24_comparison(
    heat_ad_final_loss,
    heat_ad_rel_l2,
    heat_ad_time,
    heat_fdm_final_loss,
    heat_fdm_rel_l2,
    heat_fdm_time
):
    """Print comparison table for Problem 2.4(a)."""
    print("\nProblem 2.4(a): Heat AD-PINN vs Heat FDM-PINN")
    print(f"{'Method':<15} {'Final Loss':<18} {'Relative L2 Error':<20} {'Time (s)':<12}")
    print("-" * 75)
    print(f"{'AD-PINN':<15} {heat_ad_final_loss:<18.6e} {heat_ad_rel_l2:<20.6e} {heat_ad_time:<12.2f}")
    print(f"{'FDM-PINN':<15} {heat_fdm_final_loss:<18.6e} {heat_fdm_rel_l2:<20.6e} {heat_fdm_time:<12.2f}")

def train_heat_fdm_for_epsilon(epsilon, epochs=20000):
    """Train one heat FDM-PINN for a given epsilon and return final loss, relative L2 error, and time."""
    model = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)

    loss_history, wall_time = train_pinn(
        model,
        lambda m: compute_loss_heat_fdm(m, epsilon=epsilon),
        epochs=epochs,
        lr=1e-3,
        log_every=2000
    )

    final_loss = loss_history[-1]
    rel_l2 = evaluate_heat_rel_l2(
        model,
        heat_exact_solution,
        ntest=100
    )

    return model, final_loss, rel_l2, wall_time

def run_problem_24_epsilon_sweep():
    """Train heat FDM-PINN for several epsilon values and plot relative L2 error vs epsilon."""
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    final_losses = []
    rel_l2_errors = []
    wall_times = []

    print("\nProblem 2.4(b): Heat FDM epsilon sweep")
    print(f"{'epsilon':<12} {'Final Loss':<18} {'Relative L2 Error':<20} {'Time (s)':<12}")
    print("-" * 75)

    for eps in epsilons:
        # Reset seeds so each epsilon starts from the same initial random setup
        torch.manual_seed(42)
        np.random.seed(42)

        model, final_loss, rel_l2, wall_time = train_heat_fdm_for_epsilon(
            epsilon=eps,
            epochs=20000
        )

        final_losses.append(final_loss)
        rel_l2_errors.append(rel_l2)
        wall_times.append(wall_time)

        print(f"{eps:<12.1e} {final_loss:<18.6e} {rel_l2:<20.6e} {wall_time:<12.2f}")

    # Plot relative L2 error vs epsilon
    plt.figure(figsize=(6, 4))
    plt.loglog(epsilons, rel_l2_errors, marker="o")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title(r"Problem 2.4: Heat FDM-PINN Error vs. $\epsilon$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("problem_2_4_heat_fdm_epsilon_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save numerical results
    results = np.column_stack([epsilons, final_losses, rel_l2_errors, wall_times])
    np.savetxt(
        "problem_2_4_epsilon_sweep_results.txt",
        results,
        header="epsilon final_loss relative_l2_error wall_time_seconds",
        fmt="%.8e"
    )

    return epsilons, final_losses, rel_l2_errors, wall_times

def run_classical_ode_methods_for_summary():
    """Run Problem 1.1 Forward Euler and RK4 for the Section 3(a) summary."""
    a = 0.0
    b = 5.0
    h = 0.01
    u0 = 0.0

    # Forward Euler
    t_start = time.time()
    t_euler, w_euler = classical_forward_euler(classical_ode_rhs, a, b, h, u0)
    euler_time = time.time() - t_start
    euler_error = classical_global_error(t_euler, w_euler)

    # RK4
    t_start = time.time()
    t_rk4, w_rk4 = classical_rk4(classical_ode_rhs, a, b, h, u0)
    rk4_time = time.time() - t_start
    rk4_error = classical_global_error(t_rk4, w_rk4)

    return euler_error, euler_time, rk4_error, rk4_time


def run_heat_fd_for_summary():
    """Run Problem 2.1 Forward Euler finite-difference method for Section 3(a)."""
    nu = 0.01
    Nx = 64
    T = 0.5

    t_start = time.time()
    x, t, U, dx, dt, r, Nt = heat_fd_forward_euler(nu=nu, Nx=Nx, T=T)
    heat_fd_time = time.time() - t_start

    heat_fd_error = heat_fd_l2_error(x, U[-1, :], T=T, nu=nu)

    return heat_fd_error, heat_fd_time, dx, dt, r, Nt


def print_problem_3a_error_comparison(
    euler_error,
    euler_time,
    rk4_error,
    rk4_time,
    ode_ad_error,
    ode_ad_time,
    ode_fdm_error,
    ode_fdm_time,
    heat_fd_error,
    heat_fd_time,
    heat_ad_rel_l2,
    heat_ad_time,
    heat_fdm_rel_l2,
    heat_fdm_time
):
    """Print and save the combined error comparison table for Section 3(a)."""

    rows = [
        ["ODE", "Forward Euler", "Max absolute error", euler_error, euler_time],
        ["ODE", "RK4", "Max absolute error", rk4_error, rk4_time],
        ["ODE", "AD-PINN", "Max absolute error", ode_ad_error, ode_ad_time],
        ["ODE", "FDM-PINN", "Max absolute error", ode_fdm_error, ode_fdm_time],
        ["Heat", "Forward Euler FD", "L2 error at t=0.5", heat_fd_error, heat_fd_time],
        ["Heat", "AD-PINN", "Relative L2 error", heat_ad_rel_l2, heat_ad_time],
        ["Heat", "FDM-PINN", "Relative L2 error", heat_fdm_rel_l2, heat_fdm_time],
    ]

    print("\nSection 3(a): Error Comparison Across All Methods")
    print(f"{'Problem':<10} {'Method':<22} {'Error Metric':<25} {'Error':<18} {'Time (s)':<12}")
    print("-" * 95)

    for problem, method, metric, error, wall_time in rows:
        print(f"{problem:<10} {method:<22} {metric:<25} {error:<18.6e} {wall_time:<12.2f}")

    # Save as CSV-style text for copying into LaTeX/Overleaf
    with open("problem_3a_error_comparison_table.txt", "w") as f:
        f.write("Problem,Method,Error Metric,Error,Time (s)\n")
        for problem, method, metric, error, wall_time in rows:
            f.write(f"{problem},{method},{metric},{error:.8e},{wall_time:.8e}\n")

    return rows

# Helper Functions for 3b and 3c
def train_ode_variant(
    method,
    Nr=500,
    hidden_dim=32,
    num_layers=3,
    epochs=10000,
    epsilon=1e-3,
    lr=1e-3
):
    """Train one ODE PINN variant and return final loss, max error, and time."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = PINN(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    method = method.upper()

    if method == "AD":
        loss_fn = lambda m: compute_loss_ode_ad(m, Nr=Nr)
    elif method == "FDM":
        loss_fn = lambda m: compute_loss_ode_fdm(m, epsilon=epsilon, Nr=Nr)
    else:
        raise ValueError("method must be 'AD' or 'FDM'")

    loss_history, wall_time = train_pinn(
        model,
        loss_fn,
        epochs=epochs,
        lr=lr,
        log_every=max(1, epochs // 5)
    )

    final_loss = loss_history[-1]
    max_error = evaluate_ode_max_error(
        model,
        ode_exact_solution,
        t_range=(0, 5),
        ntest=1000
    )

    return {
        "problem": "ODE",
        "method": method,
        "Nr": Nr,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "final_loss": final_loss,
        "error": max_error,
        "time": wall_time
    }


def train_heat_variant(
    method,
    Nr=10000,
    hidden_dim=32,
    num_layers=3,
    epochs=20000,
    epsilon=1e-3,
    lr=1e-3
):
    """Train one heat-equation PINN variant and return final loss, relative L2 error, and time."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = PINN(input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    method = method.upper()

    if method == "AD":
        loss_fn = lambda m: compute_loss_heat_ad(m, Nr=Nr)
    elif method == "FDM":
        loss_fn = lambda m: compute_loss_heat_fdm(m, epsilon=epsilon, Nr=Nr)
    else:
        raise ValueError("method must be 'AD' or 'FDM'")

    loss_history, wall_time = train_pinn(
        model,
        loss_fn,
        epochs=epochs,
        lr=lr,
        log_every=max(1, epochs // 5)
    )

    final_loss = loss_history[-1]
    rel_l2 = evaluate_heat_rel_l2(
        model,
        heat_exact_solution,
        ntest=100
    )

    return {
        "problem": "Heat",
        "method": method,
        "Nr": Nr,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "final_loss": final_loss,
        "error": rel_l2,
        "time": wall_time
    }


def save_rows_as_csv(filename, rows):
    """Save list-of-dict results to a CSV-style text file."""
    if len(rows) == 0:
        return

    headers = list(rows[0].keys())

    with open(filename, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

# Functions for 3b and 3c
def run_problem_3b_collocation_sweep(
    ode_epochs=10000,
    heat_epochs=20000
):
    """Problem 3(b): vary Nr and compare AD-PINN vs FDM-PINN."""

    # -------------------------
    # ODE collocation sweep
    # -------------------------
    ode_Nr_values = [100, 500, 2000, 10000]
    ode_rows = []

    print("\nProblem 3(b): ODE Collocation Point Sweep")
    print(f"{'Method':<10} {'Nr':<10} {'Final Loss':<18} {'Max Error':<18} {'Time (s)':<12}")
    print("-" * 75)

    for method in ["AD", "FDM"]:
        for Nr in ode_Nr_values:
            result = train_ode_variant(
                method=method,
                Nr=Nr,
                hidden_dim=32,
                num_layers=3,
                epochs=ode_epochs,
                epsilon=1e-3
            )

            ode_rows.append(result)

            print(
                f"{method:<10} "
                f"{Nr:<10d} "
                f"{result['final_loss']:<18.6e} "
                f"{result['error']:<18.6e} "
                f"{result['time']:<12.2f}"
            )

    save_rows_as_csv("problem_3b_ode_collocation_results.txt", ode_rows)

    plt.figure(figsize=(7, 5))

    for method in ["AD", "FDM"]:
        xs = [row["Nr"] for row in ode_rows if row["method"] == method]
        ys = [row["error"] for row in ode_rows if row["method"] == method]
        plt.loglog(xs, ys, marker="o", label=f"{method}-PINN")

    plt.xlabel(r"Number of collocation points $N_r$")
    plt.ylabel("Max absolute error")
    plt.title("Problem 3(b): ODE Error vs. Collocation Points")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("problem_3b_ode_collocation_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Heat equation collocation sweep
    # -------------------------
    heat_Nr_values = [500, 2000, 10000, 50000]
    heat_rows = []

    print("\nProblem 3(b): Heat Equation Collocation Point Sweep")
    print(f"{'Method':<10} {'Nr':<10} {'Final Loss':<18} {'Rel L2 Error':<18} {'Time (s)':<12}")
    print("-" * 75)

    for method in ["AD", "FDM"]:
        for Nr in heat_Nr_values:
            result = train_heat_variant(
                method=method,
                Nr=Nr,
                hidden_dim=32,
                num_layers=3,
                epochs=heat_epochs,
                epsilon=1e-3
            )

            heat_rows.append(result)

            print(
                f"{method:<10} "
                f"{Nr:<10d} "
                f"{result['final_loss']:<18.6e} "
                f"{result['error']:<18.6e} "
                f"{result['time']:<12.2f}"
            )

    save_rows_as_csv("problem_3b_heat_collocation_results.txt", heat_rows)

    plt.figure(figsize=(7, 5))

    for method in ["AD", "FDM"]:
        xs = [row["Nr"] for row in heat_rows if row["method"] == method]
        ys = [row["error"] for row in heat_rows if row["method"] == method]
        plt.loglog(xs, ys, marker="o", label=f"{method}-PINN")

    plt.xlabel(r"Number of collocation points $N_r$")
    plt.ylabel(r"Relative $L^2$ error")
    plt.title("Problem 3(b): Heat Equation Error vs. Collocation Points")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("problem_3b_heat_collocation_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()

    return ode_rows, heat_rows

def run_problem_3c_network_size_sweep(
    ode_epochs=10000,
    heat_epochs=20000
):
    """Problem 3(c): compare small and large networks for AD and FDM variants."""

    network_configs = [
        ("Small", 16, 2),
        ("Large", 64, 5)
    ]

    rows = []

    print("\nProblem 3(c): Effect of Network Size")
    print(
        f"{'Problem':<10} "
        f"{'Method':<10} "
        f"{'Network':<10} "
        f"{'Layers':<10} "
        f"{'Neurons':<10} "
        f"{'Final Loss':<18} "
        f"{'Error':<18} "
        f"{'Time (s)':<12}"
    )
    print("-" * 110)

    # ODE runs
    for method in ["AD", "FDM"]:
        for network_name, hidden_dim, num_layers in network_configs:
            result = train_ode_variant(
                method=method,
                Nr=500,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=ode_epochs,
                epsilon=1e-3
            )

            result["network"] = network_name
            rows.append(result)

            print(
                f"{result['problem']:<10} "
                f"{method:<10} "
                f"{network_name:<10} "
                f"{num_layers:<10d} "
                f"{hidden_dim:<10d} "
                f"{result['final_loss']:<18.6e} "
                f"{result['error']:<18.6e} "
                f"{result['time']:<12.2f}"
            )

    # Heat equation runs
    for method in ["AD", "FDM"]:
        for network_name, hidden_dim, num_layers in network_configs:
            result = train_heat_variant(
                method=method,
                Nr=10000,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=heat_epochs,
                epsilon=1e-3
            )

            result["network"] = network_name
            rows.append(result)

            print(
                f"{result['problem']:<10} "
                f"{method:<10} "
                f"{network_name:<10} "
                f"{num_layers:<10d} "
                f"{hidden_dim:<10d} "
                f"{result['final_loss']:<18.6e} "
                f"{result['error']:<18.6e} "
                f"{result['time']:<12.2f}"
            )

    save_rows_as_csv("problem_3c_network_size_results.txt", rows)

    return rows

# Functions for Bonus
def heat_exact_solution_torch(x, t, nu=0.01):
    """Torch version of the exact heat solution, used to generate noisy data."""
    return (
        torch.exp(-nu * torch.pi**2 * t) * torch.sin(torch.pi * x)
        + 0.5 * torch.exp(-9.0 * nu * torch.pi**2 * t) * torch.sin(3.0 * torch.pi * x)
    )


def make_noisy_heat_data(Ndata=50, sigma=0.01, nu_true=0.01):
    """Generate Ndata noisy observations from the exact heat solution."""
    torch.manual_seed(123)
    np.random.seed(123)

    x_data = torch.rand((Ndata, 1), device=device)
    t_data = 0.5 * torch.rand((Ndata, 1), device=device)

    xt_data = torch.cat([x_data, t_data], dim=1)

    u_clean = heat_exact_solution_torch(x_data, t_data, nu=nu_true)
    u_noisy = u_clean + sigma * torch.randn_like(u_clean)

    return xt_data, u_noisy

def compute_loss_heat_inverse_ad(
    model,
    log_nu,
    xt_data,
    u_data,
    Nr=10000,
    Nic=200,
    Nbc=200,
    lambda_data=100.0
):
    """Inverse heat PINN loss using AD, where nu is trainable."""

    # Using exp(log_nu) keeps nu positive during training
    nu = torch.exp(log_nu)

    # PDE residual points
    x_r = torch.rand((Nr, 1), device=device)
    t_r = 0.5 * torch.rand((Nr, 1), device=device)

    x_r.requires_grad_(True)
    t_r.requires_grad_(True)

    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)

    # First derivatives
    grad_u = torch.autograd.grad(
        u,
        xt_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]

    # Second x derivative
    grad_u_x = torch.autograd.grad(
        u_x,
        xt_r,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    u_xx = grad_u_x[:, 0:1]

    # PDE residual with trainable nu
    residual = u_t - nu * u_xx
    loss_r = torch.mean(residual ** 2)

    # Initial condition loss
    x_ic = torch.rand((Nic, 1), device=device)
    t_ic = torch.zeros((Nic, 1), device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)

    u_ic_exact = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3.0 * torch.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

    # Boundary condition loss
    t_bc = 0.5 * torch.rand((Nbc, 1), device=device)

    x_left = torch.zeros((Nbc, 1), device=device)
    x_right = torch.ones((Nbc, 1), device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

    # Data loss from noisy observations
    u_data_pred = model(xt_data)
    loss_data = torch.mean((u_data_pred - u_data) ** 2)

    return loss_r + 20.0 * loss_ic + 20.0 * loss_bc + lambda_data * loss_data

def compute_loss_heat_inverse_fdm(
    model,
    log_nu,
    xt_data,
    u_data,
    epsilon=1e-3,
    Nr=10000,
    Nic=200,
    Nbc=200,
    lambda_data=100.0
):
    """Inverse heat PINN loss using FDM, where nu is trainable."""

    nu = torch.exp(log_nu)

    if epsilon <= 0 or epsilon >= 0.25:
        raise ValueError("epsilon must satisfy 0 < epsilon < 0.25.")

    # Keep central-difference stencil inside the domain
    x_r = epsilon + (1.0 - 2.0 * epsilon) * torch.rand((Nr, 1), device=device)
    t_r = epsilon + (0.5 - 2.0 * epsilon) * torch.rand((Nr, 1), device=device)

    xt = torch.cat([x_r, t_r], dim=1)

    xt_t_plus = torch.cat([x_r, t_r + epsilon], dim=1)
    xt_t_minus = torch.cat([x_r, t_r - epsilon], dim=1)

    xt_x_plus = torch.cat([x_r + epsilon, t_r], dim=1)
    xt_x_minus = torch.cat([x_r - epsilon, t_r], dim=1)

    u = model(xt)
    u_t_plus = model(xt_t_plus)
    u_t_minus = model(xt_t_minus)
    u_x_plus = model(xt_x_plus)
    u_x_minus = model(xt_x_minus)

    # Finite-difference derivatives
    u_t_fd = (u_t_plus - u_t_minus) / (2.0 * epsilon)
    u_xx_fd = (u_x_plus - 2.0 * u + u_x_minus) / (epsilon ** 2)

    residual = u_t_fd - nu * u_xx_fd
    loss_r = torch.mean(residual ** 2)

    # Initial condition loss
    x_ic = torch.rand((Nic, 1), device=device)
    t_ic = torch.zeros((Nic, 1), device=device)

    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)

    u_ic_exact = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3.0 * torch.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

    # Boundary condition loss
    t_bc = 0.5 * torch.rand((Nbc, 1), device=device)

    x_left = torch.zeros((Nbc, 1), device=device)
    x_right = torch.ones((Nbc, 1), device=device)

    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)

    u_left = model(xt_left)
    u_right = model(xt_right)

    loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)

    # Data loss
    u_data_pred = model(xt_data)
    loss_data = torch.mean((u_data_pred - u_data) ** 2)

    return loss_r + 20.0 * loss_ic + 20.0 * loss_bc + lambda_data * loss_data

def train_heat_inverse(
    method,
    epochs=20000,
    epsilon=1e-3,
    nu_true=0.01,
    nu_initial=0.02,
    Ndata=50,
    sigma=0.01,
    lr=1e-3
):
    """Train inverse heat PINN and recover nu."""

    torch.manual_seed(42)
    np.random.seed(42)

    model = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)

    # Train log(nu), not nu directly, so recovered nu stays positive
    log_nu = nn.Parameter(
        torch.tensor(np.log(nu_initial), dtype=torch.float32, device=device)
    )

    xt_data, u_data = make_noisy_heat_data(
        Ndata=Ndata,
        sigma=sigma,
        nu_true=nu_true
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [log_nu],
        lr=lr
    )

    loss_history = []
    nu_history = []

    method = method.upper()

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        if method == "AD":
            loss = compute_loss_heat_inverse_ad(
                model,
                log_nu,
                xt_data,
                u_data
            )
        elif method == "FDM":
            loss = compute_loss_heat_inverse_fdm(
                model,
                log_nu,
                xt_data,
                u_data,
                epsilon=epsilon
            )
        else:
            raise ValueError("method must be 'AD' or 'FDM'")

        loss.backward()
        optimizer.step()

        nu_current = torch.exp(log_nu).item()

        loss_history.append(loss.item())
        nu_history.append(nu_current)

        if epoch % 2000 == 0:
            print(
                f"  Epoch {epoch}/{epochs}, "
                f"Loss = {loss.item():.6e}, "
                f"nu = {nu_current:.8e}"
            )

    wall_time = time.time() - t_start

    nu_recovered = torch.exp(log_nu).item()
    relative_nu_error = abs(nu_recovered - nu_true) / nu_true

    print(f"\nBonus Inverse Problem: Heat PINN with {method}")
    print(f"True nu: {nu_true:.8e}")
    print(f"Recovered nu: {nu_recovered:.8e}")
    print(f"Relative nu error: {relative_nu_error:.8e}")
    print(f"Training time: {wall_time:.2f} seconds")

    # Plot loss curve
    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Bonus: Inverse Heat PINN Loss ({method})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"bonus_inverse_heat_loss_{method.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot nu recovery over training
    plt.figure(figsize=(6, 4))
    plt.plot(nu_history, label=r"Recovered $\nu$")
    plt.axhline(nu_true, linestyle="--", label=r"True $\nu$")
    plt.xlabel("Epoch")
    plt.ylabel(r"$\nu$")
    plt.title(f"Bonus: Recovery of Diffusion Coefficient ({method})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bonus_inverse_heat_nu_{method.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close()

    return model, nu_recovered, relative_nu_error, loss_history, nu_history, wall_time

def run_bonus_inverse_problem():
    """Run the bonus inverse problem for both AD and FDM."""

    print("\nBonus: Inverse Problem for Heat Equation")
    print(f"{'Method':<10} {'nu_true':<15} {'nu_recovered':<18} {'Rel Error':<18} {'Time (s)':<12}")
    print("-" * 80)

    nu_true = 0.01

    _, nu_ad, rel_err_ad, _, _, time_ad = train_heat_inverse(
        method="AD",
        epochs=20000,
        nu_true=nu_true,
        nu_initial=0.02,
        Ndata=50,
        sigma=0.01
    )

    _, nu_fdm, rel_err_fdm, _, _, time_fdm = train_heat_inverse(
        method="FDM",
        epochs=20000,
        epsilon=1e-3,
        nu_true=nu_true,
        nu_initial=0.02,
        Ndata=50,
        sigma=0.01
    )

    print("\nBonus Summary")
    print(f"{'Method':<10} {'nu_true':<15} {'nu_recovered':<18} {'Rel Error':<18} {'Time (s)':<12}")
    print("-" * 80)
    print(f"{'AD':<10} {nu_true:<15.8e} {nu_ad:<18.8e} {rel_err_ad:<18.8e} {time_ad:<12.2f}")
    print(f"{'FDM':<10} {nu_true:<15.8e} {nu_fdm:<18.8e} {rel_err_fdm:<18.8e} {time_fdm:<12.2f}")

    results = np.array([
        [nu_true, nu_ad, rel_err_ad, time_ad],
        [nu_true, nu_fdm, rel_err_fdm, time_fdm]
    ])

    np.savetxt(
        "bonus_inverse_heat_results.txt",
        results,
        header="nu_true nu_recovered relative_nu_error wall_time_seconds\nRows: AD, FDM",
        fmt="%.8e"
    )

    return nu_ad, rel_err_ad, time_ad, nu_fdm, rel_err_fdm, time_fdm

if __name__ == "__main__":
    ode_exact = ode_exact_solution
    nu = 0.01
    heat_exact = heat_exact_solution

    # --- Problem 1.1: Classical ODE methods ---
    print("\n" + "=" * 50)
    print("Problem 1.1: Classical ODE Methods")
    print("=" * 50)

    euler_error, euler_time, rk4_error, rk4_time = run_classical_ode_methods_for_summary()

    print(f"Forward Euler max error: {euler_error:.6e}")
    print(f"Forward Euler time: {euler_time:.2f} seconds")
    print(f"RK4 max error: {rk4_error:.6e}")
    print(f"RK4 time: {rk4_time:.2f} seconds")


    # --- Problem 2.1: Heat finite-difference method ---
    print("\n" + "=" * 50)
    print("Problem 2.1: Heat Forward Euler FD")
    print("=" * 50)

    heat_fd_error, heat_fd_time, dx, dt, r, Nt = run_heat_fd_for_summary()

    print(f"dx = {dx:.6e}")
    print(f"dt = {dt:.6e}")
    print(f"r = {r:.6e}")
    print(f"Nt = {Nt}")
    print(f"Heat FD L2 error at t=0.5: {heat_fd_error:.6e}")
    print(f"Heat FD time: {heat_fd_time:.2f} seconds")

    
    # --- Problem 1.2: ODE with AD ---
    print("=" * 50)
    print("Problem 1.2: ODE PINN (Autograd)")
    print("=" * 50)
    ode_ad_model, ode_ad_loss_history, ode_ad_final_loss, ode_ad_error, ode_ad_time = (
        run_ode_ad()
    )
    
    # --- Problem 1.3: ODE with FDM ---
    print("\n" + "=" * 50)
    print("Problem 1.3: ODE PINN (FDM)")
    print("=" * 50)
    ode_fdm_model, ode_fdm_loss_history, ode_fdm_final_loss, ode_fdm_error, ode_fdm_time = (
        run_ode_fdm()
    )

    # --- Problem 1.4: Comparison ---
    print("\n" + "=" * 50)
    print("Problem 1.4: Comparison")
    print("=" * 50)

    print_problem_14_comparison(
        ode_ad_final_loss,
        ode_ad_error,
        ode_ad_time,
        ode_fdm_final_loss,
        ode_fdm_error,
        ode_fdm_time
    )
    epsilons, final_losses, max_errors, wall_times = run_problem_14_epsilon_sweep()
    
    # --- Problem 2.2: Heat with AD ---
    print("\n" + "=" * 50)
    print("Problem 2.2: Heat PINN (Autograd)")
    print("=" * 50)
    heat_ad_model, heat_ad_loss_history, heat_ad_final_loss, heat_ad_rel_l2, heat_ad_time = (
        run_heat_ad()
    )
    
    # --- Problem 2.3: Heat with FDM ---
    print("\n" + "=" * 50)
    print("Problem 2.3: Heat PINN (FDM)")
    print("=" * 50)
    heat_fdm_model, heat_fdm_loss_history, heat_fdm_final_loss, heat_fdm_rel_l2, heat_fdm_time = (
        run_heat_fdm()
    )

    # --- Problem 2.4: Comparison and Stability ---
    print("\n" + "=" * 50)
    print("Problem 2.4: Comparison and Stability")
    print("=" * 50)

    print_problem_24_comparison(
        heat_ad_final_loss,
        heat_ad_rel_l2,
        heat_ad_time,
        heat_fdm_final_loss,
        heat_fdm_rel_l2,
        heat_fdm_time
    )

    epsilons_heat, heat_final_losses, heat_rel_l2_errors, heat_wall_times = (
        run_problem_24_epsilon_sweep()
    )

    problem_3a_rows = print_problem_3a_error_comparison(
        euler_error,
        euler_time,
        rk4_error,
        rk4_time,
        ode_ad_error,
        ode_ad_time,
        ode_fdm_error,
        ode_fdm_time,
        heat_fd_error,
        heat_fd_time,
        heat_ad_rel_l2,
        heat_ad_time,
        heat_fdm_rel_l2,
        heat_fdm_time
    )

    # --- Problem 3(b): Effect of collocation points ---
    print("\n" + "=" * 50)
    print("Problem 3(b): Effect of Collocation Points")
    print("=" * 50)

    problem_3b_ode_rows, problem_3b_heat_rows = run_problem_3b_collocation_sweep()


    # --- Problem 3(c): Effect of network size ---
    print("\n" + "=" * 50)
    print("Problem 3(c): Effect of Network Size")
    print("=" * 50)

    problem_3c_rows = run_problem_3c_network_size_sweep()

    # --- Bonus: Inverse problem ---
    print("\n" + "=" * 50)
    print("Bonus: Inverse Problem")
    print("=" * 50)

    run_bonus_inverse_problem()

    print("\nDone! All plots saved.")