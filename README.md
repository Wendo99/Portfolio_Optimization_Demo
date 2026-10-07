# Portfolio Optimization Demo

WORK IN PROGRESS

A Python demonstration of portfolio optimization using the **Markowitz mean-variance model**, **Quadratic Unconstrained Binary Optimization (QUBO)**, and the **Quantum Approximate Optimization Algorithm (QAOA)** with Qiskit.

The project explores how a classical financial optimization problem can be transformed into a combinatorial optimization problem and subsequently formulated for a quantum optimization algorithm.

## Overview

Portfolio optimization aims to find an allocation that balances expected return and risk.

In this project, a portfolio selection problem is formulated using binary decision variables:

- $xᵢ = 1$ : asset $i$ is selected
- $xᵢ = 0$ : asset $i$ is not selected

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

## Method

The project combines classical financial modeling with quantum optimization:

### 1. Data

Historical market data is obtained using [`yfinance`](https://github.com/ranaroussi/yfinance).

The data is processed to obtain quantities such as expected returns and the covariance matrix required for portfolio optimization.

### 2. Portfolio Model

The portfolio selection problem is formulated using a mean-variance objective. The model balances expected return against portfolio risk.

A typical objective has the form

$$\min_x \; \lambda x^\top \Sigma x - \mu^\top x$$

where:

- $x$ is the binary asset-selection vector,
- $μ$ is the expected-return vector,
- $Σ$ is the covariance matrix,
- $λ$ controls the trade-off between risk and return.

Additional constraints can be incorporated into the QUBO formulation using penalty terms.

### 3. QUBO

The portfolio problem is transformed into a **Quadratic Unconstrained Binary Optimization** problem.

This formulation is particularly useful for QAOA because a QUBO can be mapped to an Ising Hamiltonian whose ground state corresponds to a low-cost solution of the original optimization problem.

### 4. QAOA

The QUBO is converted into an Ising Hamiltonian and used as the cost Hamiltonian in the **Quantum Approximate Optimization Algorithm**.

QAOA alternates between:

- a cost unitary based on the problem Hamiltonian, and
- a mixer unitary.

The circuit parameters are optimized classically to minimize the expected cost.

The implementation uses **Qiskit** and its optimization framework.

## Technologies

- **Python**
- **NumPy**
- **Pandas**
- **SciPy**
- **Matplotlib**
- **yfinance**
- **Qiskit**
- **Qiskit Optimization**

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

## Installation

Python **3.14 or later** is required.

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

## Running the Project

Run the corresponding Python module or script from the `src` directory according to the project configuration.

Generated figures and results are stored in the respective `images/` and `results/` directories.

## Scope and Limitations

This project is intended as a **demonstration and study project**, not as a production-grade portfolio management system or financial advisory tool.

The focus is on the computational formulation of portfolio selection and its implementation using QUBO and QAOA. The problem size is deliberately limited, making the approach suitable for experimentation and comparison with classical solutions.

In particular, the project does **not** demonstrate a quantum advantage over classical optimization methods.

## Disclaimer

This project is for educational and technical demonstration purposes only. It does not constitute financial advice or an investment recommendation.
