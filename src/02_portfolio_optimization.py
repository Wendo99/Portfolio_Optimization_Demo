import itertools
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import minimize

datafile = "portfolio_daten.npz"

# Load data
daten = np.load(Path(__file__).parent.parent / "data" / datafile)
ticker = daten["ticker"]
mu = daten["mu"]
sigma = daten["sigma"]

# MVPO / QUBO Parameter
n = len(ticker)
B = 2
lam = 0.5

# Circuit Parameter
p = 1


# Portfolio objective function C(x)
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


def is_valid_func():
    diag = np.real(np.diag(H_C.to_matrix())) + offset
    for k in range(2 ** n):
        x = [(k >> i) & 1 for i in range(n)]  # Qubit i = Bit i
        expected = objective_function(x) + alpha * (sum(x) - B) ** 2
        if not np.isclose(diag[k], expected):
            raise ValueError(
                f"H_C does not match the QUBO for x = {x}: {diag[k]} != {expected}"
            )


is_valid_func()

qc = QuantumCircuit(n)

qc.h(range(n))

ansatz = qaoa_ansatz(cost_operator=H_C, reps=p, initial_state=qc)

estimator = StatevectorEstimator()
sampler = StatevectorSampler(seed=42)
verlauf = []  # <H_C> je Iteration (Konvergenzplot)


def erwartungswert(parameter):
    """<ψ(γ,β)|H_C|ψ(γ,β)> + Konstante = erwartete Kosten C."""
    b, g = parameter
    gebundener_kreis = ansatz.assign_parameters((b, g))
    ew = estimator.run([(gebundener_kreis, H_C)]).result()[0].data.evs
    return float(ew) + offset


# %% 1. Grobsuche: β ∈ [0, π), γ ∈ [0, 2π)
#    (γ < 0 liefert wegen Symmetrie nichts Neues)
gammas = np.linspace(0, 2 * np.pi, 60)
betas = np.linspace(0, np.pi, 60, endpoint=False)
raster = np.array([[erwartungswert((b, g)) for g in gammas] for b in betas])
ib, ig = np.unravel_index(raster.argmin(), raster.shape)
start = (betas[ib], gammas[ig])
print(f"Startpunkt aus Raster: β = {start[0]:.3f}, γ = {start[1]:.3f}")


# %% 2. Klassischer Optimierer: COBYLA
def cost_function(parameter):
    wert = erwartungswert(parameter)
    verlauf.append(wert)
    return wert


# rhobeg: anfängliche Schrittweite, passend zur Rasterweite gewählt
ergebnis = minimize(
    cost_function, x0=start, method="COBYLA",
    options={"maxiter": 1000, "rhobeg": 0.2}
)
b_opt, g_opt = ergebnis.x
print(
    f"Optimiert: γ* = {g_opt:.4f}, β* = {b_opt:.4f}, "
    f"<C> = {ergebnis.fun:.4f} nach {len(verlauf)} Auswertungen"
)

# %% 3. Messen mit den optimalen Parametern
SHOTS = 10_000
messkreis = ansatz.assign_parameters((b_opt, g_opt))
messkreis.measure_all()
counts = sampler.run([messkreis], shots=SHOTS).result()[
    0].data.meas.get_counts()


def bits_zu_x(bitstring):
    """Qiskit liest von rechts: letztes Zeichen = Qubit 0 = ticker[0]."""
    return tuple(int(bitstring[n - 1 - i]) for i in range(n))


# %% 4. Auswertung
alle_x = list(itertools.product([0, 1], repeat=n))
zulaessig = [x for x in alle_x if sum(x) == B]
x_opt = min(zulaessig, key=objective_function)  # Brute Force
wahrsch = {x: 0.0 for x in alle_x}
for bitstring, anzahl in counts.items():
    wahrsch[bits_zu_x(bitstring)] = anzahl / SHOTS

P_zul = sum(wahrsch[x] for x in zulaessig)
P_opt = wahrsch[x_opt]


def name(x):
    """Portfolio als Ticker-Kette, z. B. 'GOOG+NVDA'."""
    return "+".join(t for t, xi in zip(ticker, x, strict=True) if xi) or "–"


print(f"Optimum (Brute Force): {name(x_opt)}")
print(
    f"P_zul = {P_zul:.1%}   "
    f"(Raten: {len(zulaessig)}/{2 ** n} = {len(zulaessig) / 2 ** n:.1%})"
)
print(f"P_opt = {P_opt:.1%}   (Raten: 1/{2 ** n} = {1 / 2 ** n:.1%})")
print("\nZulässige Portfolios, sortiert nach Kosten:")
for x in sorted(zulaessig, key=objective_function):
    print(
        f"  {name(x):10s} C = {objective_function(x):+.4f}   gemessen: {wahrsch[x]:.1%}"
    )

# %% Konvergenzplot (nach Stein)
plt.figure(figsize=(6, 3.5))
plt.plot(verlauf, marker=".")
plt.xlabel("Iteration (COBYLA)")
plt.ylabel("⟨C⟩ – erwartete Kosten")
plt.title(f"Konvergenz der Parameteroptimierung (p = {p})")
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "images" / "konvergenz.png", dpi=200)

# %% Messverteilung
# zulässige zuerst, jeweils nach Kosten sortiert
x_sortiert = sorted(alle_x, key=lambda x: (sum(x) != B, objective_function(x)))
farben = [
    "tab:green" if x == x_opt else "tab:blue" if sum(x) == B else "tab:red"
    for x in x_sortiert
]
plt.figure(figsize=(10, 4))
plt.bar([name(x) for x in x_sortiert], [wahrsch[x] for x in x_sortiert],
        color=farben)
plt.axhline(1 / 2 ** n, color="gray", linestyle="--",
            label=f"Raten (1/{2 ** n})")
plt.xticks(rotation=60, ha="right")
plt.ylabel("Messwahrscheinlichkeit")
plt.title(f"Messverteilung nach Optimierung (p = {p}, α_min, {SHOTS} Shots)")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / "images" / "messverteilung.png",
            dpi=200)
