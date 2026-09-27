"""QAOA for mean-variance portfolio optimization (MVPO) with a budget constraint."""

import itertools
import json
import sys
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import qiskit
import qiskit_optimization
import scipy
from matplotlib import pyplot as plt
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    datafile: str = "portfolio_daten.npz"

    # MVPO / QUBO
    budget: int = 2  # B
    lam: float = 0.5  # risk/return trade-off λ

    # Circuit
    reps: int = 1  # p

    # Grid search (always at p = 1): β ∈ [0, π), γ ∈ [0, gamma_max]
    gamma_max: float = 16 * np.pi
    n_gamma: int = 800
    n_beta: int = 40

    # COBYLA; rhobeg ≈ grid spacing (γ: 16π/799 ≈ 0.063, β: π/40 ≈ 0.079)
    rhobeg: float = 0.1
    maxiter: int = 1000

    # Simulation
    shots: int = 10_000
    seed: int = 42


# %% Problem
@dataclass
class Portfolio:
    ticker: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    budget: int
    lam: float

    @property
    def n(self) -> int:
        return len(self.ticker)

    def cost(self, x) -> float:
        """Portfolio objective C(x) = (1-λ) xᵀΣx - λ μᵀx."""
        x = np.asarray(x)
        return float((1 - self.lam) * x @ self.sigma @ x - self.lam * self.mu @ x)

    @cached_property
    def all_x(self) -> list[tuple[int, ...]]:
        return list(itertools.product([0, 1], repeat=self.n))

    @cached_property
    def feasible(self) -> list[tuple[int, ...]]:
        """Portfolios satisfying the budget, sorted by cost (the best first)."""
        return sorted((x for x in self.all_x if sum(x) == self.budget),
                      key=self.cost)

    @cached_property
    def alpha(self) -> float:
        """Penalty factor (Hodson et al. 2019): max C - min C + 0.01."""
        values = [self.cost(x) for x in self.all_x]
        return max(values) - min(values) + 0.01

    def name(self, x) -> str:
        """Portfolio as a string of ticker symbols, e.g., 'GOOG+NVDA'."""
        return "+".join(t for t, xi in zip(self.ticker, x, strict=True) if xi) or "–"


def load_portfolio(cfg: Config) -> Portfolio:
    data = np.load(ROOT / "data" / cfg.datafile)
    return Portfolio(ticker=data["ticker"], mu=data["mu"], sigma=data["sigma"],
                     budget=cfg.budget, lam=cfg.lam)


# %% QUBO -> Ising cost Hamiltonian
def cost_hamiltonian(pf: Portfolio) -> tuple[SparsePauliOp, float]:
    qp = QuadraticProgram("portfolio")
    for t in pf.ticker:
        qp.binary_var(str(t))
    qp.minimize(linear=-pf.lam * pf.mu, quadratic=(1 - pf.lam) * pf.sigma)
    qp.linear_constraint(linear=[1] * pf.n, sense="==", rhs=pf.budget)
    qubo = QuadraticProgramToQubo(penalty=pf.alpha).convert(qp)
    h_c, offset = qubo.to_ising()
    check_hamiltonian(pf, h_c, offset)
    return h_c, offset


def check_hamiltonian(pf: Portfolio, h_c: SparsePauliOp, offset: float) -> None:
    """Diagonal of H_C + offset must equal C(x) + α(Σx - B)² for every x."""
    diag = np.real(np.diag(h_c.to_matrix())) + offset
    for k in range(2 ** pf.n):
        x = [(k >> i) & 1 for i in range(pf.n)]  # qubit i = bit i
        expected = pf.cost(x) + pf.alpha * (sum(x) - pf.budget) ** 2
        if not np.isclose(diag[k], expected):
            raise ValueError(
                f"H_C does not match the QUBO for x = {x}: {diag[k]} != {expected}"
            )


# %% QAOA
class QAOA:
    """Energy evaluation for a QAOA ansatz.

    Parameter vectors follow ansatz.parameters: (β_1..β_p, γ_1..γ_p).
    """

    def __init__(self, h_c: SparsePauliOp, offset: float, reps: int,
                 estimator: StatevectorEstimator):
        self.h_c = h_c
        self.offset = offset
        self.reps = reps
        self.ansatz = qaoa_ansatz(cost_operator=h_c, reps=reps)
        self.estimator = estimator
        self.history: list[float] = []  # <C> per COBYLA evaluation

    def energy(self, params) -> np.ndarray:
        """<ψ|H_C|ψ> + offset = expected cost C; params may be batched (..., 2p)."""
        pub = (self.ansatz, self.h_c, np.asarray(params))
        return self.estimator.run([pub]).result()[0].data.evs + self.offset

    def _objective(self, params) -> float:
        value = float(self.energy(params))
        self.history.append(value)
        return value

    def optimize(self, x0, cfg: Config):
        self.history.clear()
        return minimize(self._objective, x0=x0, method="COBYLA",
                        options={"maxiter": cfg.maxiter, "rhobeg": cfg.rhobeg})


def grid_search(qaoa_p1: QAOA, cfg: Config):
    """Rough search over the p = 1 landscape.

    β is π-periodic. γ is NOT periodic because the coefficients of H_C are
    not integers -> a large search window is required.
    (γ < 0 yields no new results due to the symmetry (γ, β) -> (-γ, -β).)
    """
    betas = np.linspace(0, np.pi, cfg.n_beta, endpoint=False)
    gammas = np.linspace(0, cfg.gamma_max, cfg.n_gamma)
    bb, gg = np.meshgrid(betas, gammas, indexing="ij")
    raster = qaoa_p1.energy(np.stack([bb, gg], axis=-1))  # one batched call
    ib, ig = np.unravel_index(raster.argmin(), raster.shape)
    return betas, gammas, raster, (betas[ib], gammas[ig])


def initial_point(beta0: float, gamma0: float, reps: int) -> np.ndarray:
    """Repeat the p = 1 optimum in every layer (= INTERP for p = 2, Zhou et al. 2020)."""
    return np.concatenate([np.full(reps, beta0), np.full(reps, gamma0)])


def sample(qaoa: QAOA, params, pf: Portfolio,
           sampler: StatevectorSampler, shots: int) -> dict:
    circuit = qaoa.ansatz.assign_parameters(params)
    circuit.measure_all()
    counts = sampler.run([circuit], shots=shots).result()[0].data.meas.get_counts()
    prob = dict.fromkeys(pf.all_x, 0.0)
    for bitstring, num in counts.items():
        # Qiskit reads from the right: last character = qubit 0 = ticker[0]
        prob[tuple(int(b) for b in reversed(bitstring))] = num / shots
    return prob


# %% Analysis
def analyze(pf: Portfolio, prob: dict) -> dict:
    x_opt = pf.feasible[0]  # brute force
    return {
        "optimum": pf.name(x_opt),
        "P_zul": sum(prob[x] for x in pf.feasible),
        "P_opt": prob[x_opt],
        "P_zul_raten": len(pf.feasible) / 2 ** pf.n,
        "P_opt_raten": 1 / 2 ** pf.n,
        "zulaessige_portfolios": {
            pf.name(x): {"C": pf.cost(x), "gemessen": prob[x]}
            for x in pf.feasible
        },
    }


def print_summary(pf: Portfolio, summary: dict) -> None:
    print(f"Optimum (Brute Force): {summary['optimum']}")
    print(f"P_zul = {summary['P_zul']:.1%}   "
          f"(Raten: {len(pf.feasible)}/{2 ** pf.n} = {summary['P_zul_raten']:.1%})")
    print(f"P_opt = {summary['P_opt']:.1%}   "
          f"(Raten: 1/{2 ** pf.n} = {summary['P_opt_raten']:.1%})")
    print("\nZulässige Portfolios, sortiert nach Kosten:")
    for label, row in summary["zulaessige_portfolios"].items():
        print(f"  {label:10s} C = {row['C']:+.4f}   gemessen: {row['gemessen']:.1%}")


def save_results(path: Path, cfg: Config, pf: Portfolio, result, n_evals: int,
                 summary: dict) -> None:
    """Source for all numbers on the slides."""
    reps = cfg.reps
    results = {
        "parameter": {**asdict(cfg), "n": pf.n, "alpha": pf.alpha},
        "optimierung": {"beta_opt": result.x[:reps].tolist(),
                        "gamma_opt": result.x[reps:].tolist(),
                        "erwartete_kosten": result.fun,
                        "auswertungen_cobyla": n_evals},
        "ergebnis": summary,
        "versionen": {"python": sys.version.split()[0],
                      "qiskit": qiskit.__version__,
                      "qiskit_optimization": qiskit_optimization.__version__,
                      "numpy": np.__version__, "scipy": scipy.__version__},
    }
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)


# %% Plots
def plot_convergence(history: list[float], reps: int, path: Path) -> None:
    """Convergence plot (according to Stein)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history, marker=".")
    ax.set_xlabel("Iteration (COBYLA)")
    ax.set_ylabel("⟨C⟩ – erwartete Kosten")
    ax.set_title(f"Konvergenz der Parameteroptimierung (p = {reps})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_landscape(betas, gammas, raster, start, path: Path) -> None:
    """Energy landscape of the p = 1 grid search (backup slide)."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    mesh = ax.pcolormesh(gammas, betas, raster, shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="⟨C⟩ – erwartete Kosten")
    ax.plot(start[1], start[0], marker="*", color="red", markersize=14,
            label="Minimum (Raster)")
    ax.axvline(2 * np.pi, color="white", linestyle="--",
               label="γ = 2π (übliches Suchfenster)")
    ax.set_xlabel("γ")
    ax.set_ylabel("β")
    ax.set_title("Energielandschaft ⟨C⟩(γ, β), p = 1")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_distribution(pf: Portfolio, prob: dict, cfg: Config, path: Path) -> None:
    """Measurement distribution: feasible portfolios first, sorted by cost."""
    x_opt = pf.feasible[0]
    infeasible = sorted(set(pf.all_x) - set(pf.feasible), key=pf.cost)
    x_sorted = pf.feasible + infeasible
    colors = ["tab:green" if x == x_opt
              else "tab:blue" if sum(x) == pf.budget
              else "tab:red" for x in x_sorted]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([pf.name(x) for x in x_sorted], [prob[x] for x in x_sorted],
           color=colors)
    ax.axhline(1 / 2 ** pf.n, color="gray", linestyle="--",
               label=f"Raten (1/{2 ** pf.n})")
    ax.tick_params(axis="x", labelrotation=60)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_ylabel("Messwahrscheinlichkeit")
    ax.set_title(f"Messverteilung nach Optimierung "
                 f"(p = {cfg.reps}, α_min, {cfg.shots} Shots)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# %% Main
def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    pf = load_portfolio(cfg)
    h_c, offset = cost_hamiltonian(pf)
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler(seed=cfg.seed)

    # 1. Rough search on the p = 1 landscape
    qaoa_p1 = QAOA(h_c, offset, reps=1, estimator=estimator)
    betas, gammas, raster, start = grid_search(qaoa_p1, cfg)
    print(f"Startpunkt aus Raster: β = {start[0]:.3f}, γ = {start[1]:.3f}")

    # 2. Classical optimizer: COBYLA
    qaoa = qaoa_p1 if cfg.reps == 1 else QAOA(h_c, offset, cfg.reps, estimator)
    result = qaoa.optimize(initial_point(*start, cfg.reps), cfg)
    betas_opt, gammas_opt = result.x[:cfg.reps], result.x[cfg.reps:]
    print(f"Optimiert: γ* = {np.round(gammas_opt, 4)}, β* = {np.round(betas_opt, 4)}, "
          f"<C> = {result.fun:.4f} nach {len(qaoa.history)} Auswertungen")

    # 3. Measuring with the optimal parameters
    prob = sample(qaoa, result.x, pf, sampler, cfg.shots)

    # 4. Analysis and output
    summary = analyze(pf, prob)
    print_summary(pf, summary)
    save_results(ROOT / "results" / "results.json", cfg, pf, result,
                 len(qaoa.history), summary)

    images = ROOT / "images"
    images.mkdir(exist_ok=True)
    plot_convergence(qaoa.history, cfg.reps, images / "konvergenz.png")
    plot_landscape(betas, gammas, raster, start, images / "energielandschaft.png")
    plot_distribution(pf, prob, cfg, images / "messverteilung.png")
    return summary


if __name__ == "__main__":
    main()
