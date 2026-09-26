import itertools
import json
import sys
from pathlib import Path

import numpy as np
import qiskit
import qiskit_optimization
import scipy
from matplotlib import pyplot as plt
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import minimize

datafile = "portfolio_daten.npz"
root = Path(__file__).parent.parent

# Load data
daten = np.load(root / "data" / datafile)
ticker = daten["ticker"]
mu = daten["mu"]
sigma = daten["sigma"]

# MVPO / QUBO Parameter
n = len(ticker)
B = 2
lam = 0.5

# Circuit Parameter
p = 1


# Portfolio objective function
def objective_function(w):
    x = np.array(w)
    return (1 - lam) * x @ sigma @ x - lam * mu @ x


def create_values():
    values = [objective_function(x) for x in
              itertools.product([0, 1], repeat=n)]
    return values


def create_alpha():
    values = create_values()
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


qubo = create_qubo()

# Ising-Modell and Cost-Hamiltonian

H_C, offset = qubo.to_ising()


def check_hamiltonian():
    diag = np.real(np.diag(H_C.to_matrix())) + offset
    for k in range(2 ** n):
        x = [(k >> i) & 1 for i in range(n)]  # Qubit i = Bit i
        expected = objective_function(x) + alpha * (sum(x) - B) ** 2
        if not np.isclose(diag[k], expected):
            raise ValueError(
                f"H_C does not match the QUBO for x = {x}: {diag[k]} != {expected}"
            )


check_hamiltonian()

ansatz = qaoa_ansatz(cost_operator=H_C, reps=p)

estimator = StatevectorEstimator()
sampler = StatevectorSampler(seed=42)
history = []  # <H_C> per iteration (convergence plot)


def exp_value(parameter):
    """<ψ(β,γ)|H_C|ψ(β,γ)> + Konstante = erwartete Kosten C."""
    b, g = parameter
    ansatz_circuit = ansatz.assign_parameters((b, g))
    ew = estimator.run([(ansatz_circuit, H_C)]).result()[0].data.evs
    return float(ew) + offset


# %% 1. Rough search: β ∈ [0, π), γ ∈ [0, 16π)
#    β is π-periodic. γ is NOT periodic because the coefficients
#    of H_C are not integers -> a large search window is required.
#    (γ < 0 yields no new results due to symmetry)
GAMMA_MAX = 16 * np.pi
gammas = np.linspace(0, GAMMA_MAX, 800)
betas = np.linspace(0, np.pi, 40, endpoint=False)
raster = np.array([[exp_value((b, g)) for g in gammas] for b in betas])
ib, ig = np.unravel_index(raster.argmin(), raster.shape)
start = (betas[ib], gammas[ig])
print(f"Startpunkt aus Raster: β = {start[0]:.3f}, γ = {start[1]:.3f}")


# %% 2. Classical optimizer: COBYLA
def cost_function(parameter):
    wert = exp_value(parameter)
    history.append(wert)
    return wert


# rhobeg: initial step size, chosen to match the grid spacing
result = minimize(
    cost_function, x0=start, method="COBYLA",
    options={"maxiter": 1000, "rhobeg": 0.1}
)
b_opt, g_opt = result.x
print(
    f"Optimiert: γ* = {g_opt:.4f}, β* = {b_opt:.4f}, "
    f"<C> = {result.fun:.4f} nach {len(history)} Auswertungen"
)

# %% 3. Measuring with the Optimal Parameters
SHOTS = 10_000
meas_circuit = ansatz.assign_parameters((b_opt, g_opt))
meas_circuit.measure_all()
counts = sampler.run([meas_circuit], shots=SHOTS).result()[
    0].data.meas.get_counts()


def bits_to_x(bitstring):
    """Qiskit reads from the right: last character = qubit 0 = ticker[0]."""
    return tuple(int(bitstring[n - 1 - i]) for i in range(n))


# %% 4. Analysis
all_x = list(itertools.product([0, 1], repeat=n))
permissible = [x for x in all_x if sum(x) == B]
x_opt = min(permissible, key=objective_function)  # Brute Force
prob = {x: 0.0 for x in all_x}
for bitstring, num in counts.items():
    prob[bits_to_x(bitstring)] = num / SHOTS

P_zul = sum(prob[x] for x in permissible)
P_opt = prob[x_opt]


def name(x):
    """Portfolio as a string of ticker symbols, e.g., 'GOOG+NVDA'."""
    return "+".join(t for t, xi in zip(ticker, x, strict=True) if xi) or "–"


print(f"Optimum (Brute Force): {name(x_opt)}")
print(
    f"P_zul = {P_zul:.1%}   "
    f"(Raten: {len(permissible)}/{2 ** n} = {len(permissible) / 2 ** n:.1%})"
)
print(f"P_opt = {P_opt:.1%}   (Raten: 1/{2 ** n} = {1 / 2 ** n:.1%})")
print("\nZulässige Portfolios, sortiert nach Kosten:")
for x in sorted(permissible, key=objective_function):
    print(
        f"  {name(x):10s} C = {objective_function(x):+.4f}   gemessen: {prob[x]:.1%}"
    )

# %% 5. Ergebnisse festhalten (Quelle für alle Zahlen auf den Folien)
ergebnisse = {
    "parameter": {"n": n, "B": B, "lambda": lam, "alpha": alpha, "p": p,
                  "shots": SHOTS, "seed": 42,
                  "suchfenster_gamma": [0.0, GAMMA_MAX],
                  "raster": [len(betas), len(gammas)]},
    "optimierung": {"gamma_opt": g_opt, "beta_opt": b_opt,
                    "erwartete_kosten": result.fun,
                    "auswertungen_cobyla": len(history)},
    "ergebnis": {"optimum": name(x_opt), "P_zul": P_zul, "P_opt": P_opt,
                 "P_zul_raten": len(permissible) / 2 ** n,
                 "P_opt_raten": 1 / 2 ** n,
                 "zulaessige_portfolios": {
                     name(x): {"C": objective_function(x),
                               "gemessen": prob[x]}
                     for x in sorted(permissible, key=objective_function)}},
    "versionen": {"python": sys.version.split()[0],
                  "qiskit": qiskit.__version__,
                  "qiskit_optimization": qiskit_optimization.__version__,
                  "numpy": np.__version__, "scipy": scipy.__version__},
}
(root / "results").mkdir(exist_ok=True)
with open(root / "results" / "ergebnisse.json", "w", encoding="utf-8") as f:
    json.dump(ergebnisse, f, indent=2, ensure_ascii=False, default=float)

# %% Convergence plot (according to Stein)
plt.figure(figsize=(6, 3.5))
plt.plot(history, marker=".")
plt.xlabel("Iteration (COBYLA)")
plt.ylabel("⟨C⟩ – erwartete Kosten")
plt.title(f"Konvergenz der Parameteroptimierung (p = {p})")
plt.tight_layout()
plt.savefig(root / "images" / "konvergenz.png", dpi=200)

# %% Energy Landscape (Backup Slide)
plt.figure(figsize=(10, 3.5))
plt.pcolormesh(gammas, betas, raster, shading="nearest", cmap="viridis")
plt.colorbar(label="⟨C⟩ – erwartete Kosten")
plt.plot(g_opt, b_opt, marker="*", color="red", markersize=14,
         label="Optimum (COBYLA)")
plt.axvline(2 * np.pi, color="white", linestyle="--",
            label="γ = 2π (übliches Suchfenster)")
plt.xlabel("γ")
plt.ylabel("β")
plt.title(f"Energielandschaft ⟨C⟩(γ, β), p = {p}")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(root / "images" / "energielandschaft.png", dpi=200)

# %% Measurement Distribution
# Permitted ones first, sorted by cost
x_sorted = sorted(all_x, key=lambda x: (sum(x) != B, objective_function(x)))
colors = [
    "tab:green" if x == x_opt else "tab:blue" if sum(x) == B else "tab:red"
    for x in x_sorted
]
plt.figure(figsize=(10, 4))
plt.bar([name(x) for x in x_sorted], [prob[x] for x in x_sorted],
        color=colors)
plt.axhline(1 / 2 ** n, color="gray", linestyle="--",
            label=f"Raten (1/{2 ** n})")
plt.xticks(rotation=60, ha="right")
plt.ylabel("Messwahrscheinlichkeit")
plt.title(f"Messverteilung nach Optimierung (p = {p}, α_min, {SHOTS} Shots)")
plt.legend()
plt.tight_layout()
plt.savefig(root / "images" / "messverteilung.png",
            dpi=200)
