import itertools

import numpy as np
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo



n = 4
B = 2
lam = 0.5
p = 1

# load data
daten = np.load("../data/portfolio_daten.npz")
ticker = daten["ticker"]
mu = daten["mu"]
sigma = daten["sigma"]

# create the portfolio_objective_function
def C(x):
  x = np.array(x)
  return (1 - lam) * x @ sigma @ x - lam * mu @ x


def create_values():
  values = [C(x) for x in itertools.product([0, 1], repeat=n)]
  return values

values = create_values()

def create_alpha():
  alpha = max(values) - min(values) + 0.01  # (Hodson)
  return alpha

alpha = create_alpha()

# MVPO as an optimization problem, then QUBO
def create_qubo():
  qp = QuadraticProgram("portfolio")
  for a in ticker:
    qp.binary_var(a)
  qp.minimize(linear=-lam * mu, quadratic=(1 - lam) * sigma)
  qp.linear_constraint(linear=[1] * n, sense="==", rhs=B)
  qubo = QuadraticProgramToQubo(penalty=alpha).convert(qp)
  return qubo

qubo =create_qubo()

# Ising-Modell and Cost-Hamiltonian

H_C, offset = qubo.to_ising()
print(H_C)

def is_valid_func():
  diag = np.real(np.diag(H_C.to_matrix())) + offset
  for k in range(2**n):
    x = [(k >> i) & 1 for i in range(n)]      # Qubit i = Bit i
    assert np.isclose(diag[k], C(x) + alpha * (sum(x) - B) ** 2)

is_valid_func()