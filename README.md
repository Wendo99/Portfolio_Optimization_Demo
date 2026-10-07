# Portfolio Optimization Demo

WORK IN PROGRESS

A Python demonstration of portfolio optimization using the **Markowitz mean-variance model**, **Quadratic Unconstrained Binary Optimization (QUBO)**, and the **Quantum Approximate Optimization Algorithm (QAOA)** with Qiskit.

The project explores how a classical financial optimization problem can be transformed into a combinatorial optimization problem and subsequently formulated for a quantum optimization algorithm.

## Overview

Portfolio optimization aims to find an allocation that balances expected return and risk.

Historical market data is obtained using [`yfinance`](https://github.com/ranaroussi/yfinance).

The optimization uses a fixed four-asset instance. Each binary variable indicates whether an asset is selected; the budget is exactly two assets.

The optimization objective combines expected returns and portfolio risk and is formulated as a **QUBO** problem. The resulting QUBO is then mapped to an Ising Hamiltonian and used as the cost Hamiltonian for QAOA.

The project therefore follows the workflow:

```text
Market Data
    ↓
Returns & Covariance Matrix
    ↓
Portfolio Optimization Model
    ↓
QUBO Formulation
    ↓
Ising Hamiltonian
    ↓
QAOA
    ↓
Measurement & Evaluation
```

## Project Structure

```text
Portfolio_Optimization_Demo/
├── data/       # Input and processed data
├── images/     # Figures and visualizations
├── results/    # Generated results
├── src/        # Source code
├── pyproject.toml
└── uv.lock
```

## Installation and Run

Clone the repository:

```bash
git clone https://github.com/Wendo99/Portfolio_Optimization_Demo.git
cd Portfolio_Optimization_Demo
```

Install the project and its dependencies using your preferred Python package manager.

For example, with `uv`:

```bash
uv sync
```

Run the corresponding Python module or script from the `src` directory according to the project configuration.

Generated figures and results are stored in the respective `images/` and `results/` directories.

## Scope and Limitations

This is a small demonstration of encoding a binary asset-selection problem as a QUBO and sampling it with QAOA. The default instance has four assets and selects exactly two. It uses historical estimates from a single year and a statevector simulator, and it does not model asset weights, share counts, transaction costs, trading constraints, or a monetary budget. It is not a production portfolio tool and does not demonstrate quantum advantage; the exact feasible optimum is also found by enumerating all 16 portfolios.

## Disclaimer

This project is for educational and technical demonstration purposes only. It does not constitute financial advice or an investment recommendation.
