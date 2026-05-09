# Numerical Analysis Final Project: Physics-Informed Neural Networks

Alex Zhu  
EN 553.481 Numerical Analysis

This repository contains the source code for my Numerical Analysis final project on Physics-Informed Neural Networks (PINNs) for solving differential equations.

The project compares classical numerical methods with PINN methods using automatic differentiation and finite-difference derivative approximations.

## Files

### `problem_1_1_classical_methods.py`

This file solves the first-order ODE problem using classical numerical methods. It includes:

- Forward Euler method
- Fourth-order Runge-Kutta method
- Global error computation
- Observed convergence order table

It produces plots for Problem 1.1 and prints the convergence table.

### `problem_2_1_heat_fd.py`

This file solves the one-dimensional heat equation using the Forward Euler finite-difference method.

It includes:

- Stable Forward Euler finite-difference heat equation solver
- Final-time discrete \(L^2\) error
- Heatmap of the solution
- CFL-violating Forward Euler run for Problem 2.4(c)

This file should be run separately to generate the Problem 2.1 plots and the Problem 2.4(c) CFL-violation plot/table.

### `PINN.py`

This is the main PINN file. It contains:

- PINN neural network class
- ODE AD-PINN loss
- ODE FDM-PINN loss
- Heat equation AD-PINN loss
- Heat equation FDM-PINN loss
- Epsilon sweeps
- Collocation-point sweeps
- Network-size sweeps
- Bonus inverse problem for recovering \(\nu\)

It also imports functions from `problem_1_1_classical_methods.py` and `problem_2_1_heat_fd.py` for the final comparison tables.

## Requirements

The code uses Python 3 and the following packages:

- numpy
- matplotlib
- torch

Install the required packages as required with pip.

## How to Run the Code

To reproduce the classical ODE results for Problem 1.1, run:

```bash
python problem_1_1_classical_methods.py
```

To reproduce the heat-equation finite-difference results for Problem 2.1 and the CFL-violation experiment for Problem 2.4(c), run:

```bash
python problem_2_1_heat_fd.py
```

To run the main PINN experiments, run:

```bash
python PINN.py
```

The full `PINN.py` script trains all the necessary neural networks, including the epsilon sweeps, collocation-point sweeps, network-size sweeps, and the bonus inverse problem. 

## Terminal Output Explanation

During training, the terminal prints progress updates. For example, the terminal may show lines like:

- Epoch 2000/20000, Loss = ...
- Epoch 4000/20000, Loss = ...

These epoch lines show the loss during a single model training run.

For most experiments, the loss is printed every 2000 epochs. For some sweeps in Problem 3(b) and Problem 3(c), such as during 20000-epoch runs, the code prints every 4000 epochs.

After each individual training run finishes, the code prints the training time, such as:

- Training time: ...

Then the corresponding table row is printed. For example, in an epsilon sweep, the row contains values such as:

- epsilon, Final Loss, Relative L2 Error, Time (s)

This row is printed only after that particular epsilon value has finished training.

As an example, during Problem 2.4(b), the terminal first prints the epoch losses for one epsilon value. After that run finishes, the code prints the final row for that epsilon, including its final loss, relative \(L^2\) error, and training time. Then it moves to the next epsilon value and repeats the process. Once the row appears, that specific run is finished.

This same pattern applies to the longer sweep experiments, including:

- Problem 1.4(b), the ODE FDM epsilon sweep
- Problem 2.4(b), the heat FDM epsilon sweep
- Problem 3(b), the collocation-point sweep
- Problem 3(c), the network-size sweep
- Bonus inverse problem

## Output Files

Running the scripts creates `.png` plot files and `.txt` result files in the same directory.

Generated plot files include:

- ode_forward_euler_h001.png
- ode_rk4_h001.png
- heat_forward_euler_heatmap.png
- heat_forward_euler_final_time.png
- problem_1_2_loss_ad.png
- problem_1_2_ode_ad_comparison.png
- problem_1_3_loss_fdm.png
- problem_1_3_ode_fdm_comparison.png
- problem_1_4_fdm_epsilon_sweep.png
- problem_2_2_loss_ad.png
- problem_2_2_heat_ad_comparison.png
- problem_2_3_loss_fdm.png
- problem_2_3_heat_fdm_comparison.png
- problem_2_4_heat_fdm_epsilon_sweep.png
- problem_2_4_cfl_violation_heatmap.png
- problem_3b_ode_collocation_sweep.png
- problem_3b_heat_collocation_sweep.png
- bonus_inverse_heat_loss_ad.png
- bonus_inverse_heat_loss_fdm.png
- bonus_inverse_heat_nu_ad.png
- bonus_inverse_heat_nu_fdm.png


Generated text result files include:

- problem_1_4_epsilon_sweep_results.txt
- problem_2_4_epsilon_sweep_results.txt
- problem_3a_error_comparison_table.txt
- problem_3b_ode_collocation_results.txt
- problem_3b_heat_collocation_results.txt
- problem_3c_network_size_results.txt
- bonus_inverse_heat_results.txt

For local file management, I organized the generated `.png` and `.txt` output files into a separate `Plots/` folder after running the scripts. The Python scripts themselves save output files to the current working directory. The `Plots/` folder is ignored by `.gitignore`, so it is used for my local organization rather than as a required part of the GitHub submission.

## Note on Timing and Hardware

Some experiments were run on different hardware environments. Earlier runs were performed on CPU, while later experiments, including Problem 3(b), Problem 3(c), and the Bonus inverse problem, were run with GPU acceleration when available. Because of this, wall-clock training times from CPU runs should not be directly compared with GPU runs.

For GPU runs, PyTorch CUDA operations may be asynchronous, so exact timing can depend on whether CUDA synchronization is used. The reported times should therefore be interpreted as approximate wall-clock measurements for the hardware environment used in each run. The relative comparisons within the same section are still meaningful when the experiments were run under the same environment.

## Randomness and Reproducibility

The code sets random seeds using:

```python
torch.manual_seed(42)
np.random.seed(42)
```

These seeds help make the experiments more reproducible, but they do not guarantee that every run will produce exactly the same final numbers. PINN training involves random neural-network initialization, random collocation-point sampling, and nonconvex optimization. Results can also vary slightly across CPU/GPU environments, PyTorch versions, and hardware. Small differences in final loss, error, or wall-clock time are expected.

Also, the classical numerical methods in `problem_1_1_classical_methods.py` and `problem_2_1_heat_fd.py` are deterministic because they use fixed grids and fixed update formulas. Therefore, they do not require random seeds in the same way the PINN experiments do.

### Note on Repeated Runs

In my report, each individual problem section was run separately to produce representative results for that method. For example, Problems 1.2, 1.3, 2.2, and 2.3 show standalone runs for the AD-PINN and FDM-PINN methods.

The later comparison sections, such as Problem 1.4, Problem 2.4, and Problem 3(a), were produced from separate composite comparison runs so that the methods being compared were run within the same comparison workflow. Because these comparison runs are not necessarily the exact same training runs as the earlier standalone sections, the final losses, errors, and wall-clock times may differ slightly.

Therefore, small differences are present, for instance between Problem 1.3 and Problem 1.4, but they only come from separate runs of the same methods rather than from different algorithms.

Even when the same seed is used, results can still differ as more experiments are run at once and thus in a different order, because the random number generator state changes as earlier models are initialized and earlier collocation points are sampled. GPU computation can also introduce small numerical differences. For this reason, the reported values should be interpreted as representative results from the corresponding run, and the most meaningful comparisons are the relative comparisons made within the same table or section.

## Deliverable

The main LaTeX report explains the numerical results, plots, tables, and comparisons. The source code here is intended to reproduce the experiments used in the report.

The written project report is included in the root directory of this repository as:

[Numerical_Analysis_Final.pdf](Numerical_Analysis_Final.pdf)

The version submitted on Gradescope is named `Numerical_Analysis_Final (5).pdf`. This is the same report file, but renamed during download/upload.