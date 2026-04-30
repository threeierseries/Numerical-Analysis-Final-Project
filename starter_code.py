"""
Starter Code: PINN Final Project
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
# TODO: Implement these four loss functions
# =============================================================

def compute_loss_ode_ad(model):
    """PINN loss for ODE using AUTOGRAD.

    ODE: du/dt = -5u + 5cos(t) - sin(t),  u(0) = 0

    """
    raise NotImplementedError("TODO: implement this")


def compute_loss_ode_fdm(model, epsilon=1e-3):
    """PINN loss for ODE using FINITE DIFFERENCES.

    Same ODE as above. Instead of autograd, approximate du/dt
    using the central difference formula:

        du/dt(t) ≈ (u(t + epsilon) - u(t - epsilon)) / (2 * epsilon)
    """
    raise NotImplementedError("TODO: implement this")


def compute_loss_heat_ad(model):
    """PINN loss for heat equation using AUTOGRAD.

    PDE: u_t = 0.01 * u_xx  on (0,1) x (0, 0.5]
    IC:  u(x, 0) = sin(pi*x) + 0.5*sin(3*pi*x)
    BC:  u(0, t) = u(1, t) = 0
    """
    raise NotImplementedError("TODO: implement this")


def compute_loss_heat_fdm(model, epsilon=1e-3):
    """PINN loss for heat equation using FINITE DIFFERENCES.

    Same PDE, IC, BC as above. Approximate derivatives:

        u_t(x,t)  ≈ (u(x, t+eps) - u(x, t-eps)) / (2*eps)
        u_xx(x,t) ≈ (u(x+eps, t) - 2*u(x,t) + u(x-eps, t)) / eps^2
    """
    raise NotImplementedError("TODO: implement this")


if __name__ == "__main__":
    ode_exact = NotImplementedError("TODO: implement the exact solution for the ODE")
    nu = 0.01
    heat_exact = NotImplementedError("TODO: implement the exact solution for the heat equation")

    # --- Problem 1.2: ODE with AD ---
    print("=" * 50)
    print("Problem 1.2: ODE PINN (Autograd)")
    print("=" * 50)
   ## TODO: Experiments for Problem 1.2: train the AD-PINN for the ODE, plot loss curve and results

    # --- Problem 1.3: ODE with FDM ---
    print("\n" + "=" * 50)
    print("Problem 1.3: ODE PINN (FDM)")
    print("=" * 50)
    ## TODO: Experiments for Problem 1.3: train the FDM-PINN for the ODE, plot loss curve and results

    # --- Problem 2.2: Heat with AD ---
    print("\n" + "=" * 50)
    print("Problem 2.2: Heat PINN (Autograd)")
    print("=" * 50)
    ## TODO: Experiments for Problem 2.2: train the AD-PINN for the heat equation, plot loss curve and results

    # --- Problem 2.3: Heat with FDM ---
    print("\n" + "=" * 50)
    print("Problem 2.3: Heat PINN (FDM)")
    print("=" * 50)
   ## TODO: Experiments for Problem 2.3: train the FDM-PINN for the heat equation, plot loss curve and results

    # --- Summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Method':<25} {'Problem':<10} {'Error':<15} {'Time (s)':<10}")
    print("-" * 60)
    ## TODO: Print a summary table comparing the 4 methods (ODE-AD, ODE-FDM, Heat-AD, Heat-FDM) in terms of max error and training time.

    print("\nDone! All plots saved.")
