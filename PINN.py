"""
PINN Final Project
EN 553.481/681 Numerical Analysis
"""
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

def compute_loss_ode_ad(model):
    """PINN loss for ODE using AUTOGRAD.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    IC:  u(0) = 0
    Loss: L = L_r + 50 L_ic
    """
    Nr = 500

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


def compute_loss_ode_fdm(model, epsilon=1e-3):
    """PINN loss for ODE using FINITE DIFFERENCES.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0
    IC:  u(0) = 0
    Derivative approximation:
        du/dt(t) ≈ [u(t + epsilon) - u(t - epsilon)] / (2 * epsilon)
    Loss: L = L_r + 50 L_ic
    """
    Nr = 500

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

def compute_loss_heat_ad(model):
    """PINN loss for heat equation using AUTOGRAD.

    PDE: u_t = 0.01 * u_xx  on (0,1) x (0, 0.5]
    IC:  u(x, 0) = sin(pi*x) + 0.5*sin(3*pi*x)
    BC:  u(0, t) = u(1, t) = 0
    """

    nu = 0.01
    Nr = 10000
    Nic = 200
    Nbc = 200


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

def compute_loss_heat_fdm(model, epsilon=1e-3):
    """PINN loss for heat equation using FINITE DIFFERENCES.

    Same PDE, IC, BC as above. Approximate derivatives:

        u_t(x,t)  ≈ (u(x, t+eps) - u(x, t-eps)) / (2*eps)
        u_xx(x,t) ≈ (u(x+eps, t) - 2*u(x,t) + u(x-eps, t)) / eps^2
    """

    nu = 0.01
    Nr = 10000
    Nic = 200
    Nbc = 200

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

if __name__ == "__main__":
    ode_exact = ode_exact_solution
    nu = 0.01
    heat_exact = heat_exact_solution
    """
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
    """
    # --- Problem 2.3: Heat with FDM ---
    print("\n" + "=" * 50)
    print("Problem 2.3: Heat PINN (FDM)")
    print("=" * 50)
    heat_fdm_model, heat_fdm_loss_history, heat_fdm_final_loss, heat_fdm_rel_l2, heat_fdm_time = (
        run_heat_fdm()
    )

    # --- Summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Method':<25} {'Problem':<10} {'Error':<15} {'Time (s)':<10}")
    print("-" * 60)
    ## TODO: Print a summary table comparing the 4 methods (ODE-AD, ODE-FDM, Heat-AD, Heat-FDM) in terms of max error and training time.

    print("\nDone! All plots saved.")
