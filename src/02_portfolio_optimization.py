"""QAOA for mean-variance portfolio optimization (MVPO) with a budget constraint."""

import itertools
import json
import re
import sys
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import qiskit
import qiskit_optimization
import scipy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction, Parameter
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent

# Presentation fonts. Figures are 12 in wide, i.e. about the content width of
# a 16:9 slide (13.33 in), so these point sizes are what the audience sees
# when a plot is placed at full slide width.
SLIDE_FIG_WIDTH = 12.0
PRESENTATION_RC = {
    "font.size": 16,
    "axes.titlesize": 22,
    "figure.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}
# Circuit plot: point sizes on the slide for gate names and gate angles.
# Smaller than the other plots: Qiskit's layout does not grow with the font,
# and with ~22 gate columns a column is only ~0.5 in wide at slide width.
CIRCUIT_GATE_PT = 13
CIRCUIT_PARAM_PT = 11


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
    maxiter: int = 2000

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

    Parameter vectors follow ansatz.parameters: (β_1…β_p, γ_1…γ_p).
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


def _ordered_cost_terms(h_c: SparsePauliOp) -> list[tuple[list[int], float]]:
    """Cost terms of H_C as (qubits, coeff), ordered for a narrow figure.

    The terms are all diagonal and commute, so they can be reordered freely:
    all Rz first, then the ZZ gates packed into as few columns as possible
    (e.g. ZZ(0,1) and ZZ(2,3) side by side).
    """
    terms = [(np.flatnonzero(pauli.z).tolist(), coeff)
             for pauli, coeff in zip(h_c.paulis, h_c.coeffs.real, strict=True)]
    singles = [t for t in terms if len(t[0]) == 1]
    # Greedy packing: a ZZ gate joins the first column whose gates' wire spans
    # it does not overlap (the vertical line of ZZ(i, j) covers wires i…j)
    columns: list[list] = []
    for t in sorted((t for t in terms if len(t[0]) == 2), key=lambda t: t[0]):
        i, j = t[0]
        col = next((c for c in columns
                    if all(j < a or b < i for (a, b), _ in c)), None)
        if col is None:
            columns.append(col := [])
        col.append(t)
    return singles + [t for col in columns for t in col]


def _append_cost_layer(qc: QuantumCircuit, terms: list, gamma: Parameter,
                       ks: str, symbolic: bool) -> None:
    """Append U_C(γ): Z_i -> Rz(2h_i·γ), Z_iZ_j -> Rzz(2J_ij·γ)."""
    sep = "" if qc.num_qubits < 10 else ","  # J_12, or J_{10,11} for many qubits
    for qubits, coeff in terms:
        if symbolic:
            idx = sep.join(str(q + 1) for q in qubits)
            name = "h" if len(qubits) == 1 else "J"
            angle = Parameter(rf"$2{name}_{{{idx}}}\gamma_{ks}$")
        else:
            angle = 2 * coeff * gamma
        if len(qubits) == 1:
            qc.rz(angle, qubits[0])
        else:
            qc.rzz(angle, *qubits)


def drawing_circuit(h_c: SparsePauliOp, reps: int,
                    symbolic: bool = True) -> QuantumCircuit:
    """Gate-level QAOA circuit with labeled layers, for display only.

    Same gates as qaoa_ansatz for H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j:
    Z_i -> Rz(2h_i·γ), Z_iZ_j -> Rzz(2J_ij·γ), mixer Rx(2β).
    symbolic=True labels the angles with h_i, J_ij (no numbers);
    symbolic=False uses the exact coefficients (for checking against qaoa_ansatz).
    Measurements are drawn as "M" boxes without classical bits, so the drawer
    places them in one column (real measurements are always staggered).
    The cost terms are reordered to make the figure narrower
    (see _ordered_cost_terms).
    """
    ordered = _ordered_cost_terms(h_c)

    # ASCII names; plot_circuit turns them into $x_{i}$ (Qiskit escapes "_")
    regs = [QuantumRegister(1, f"x{i + 1}") for i in range(h_c.num_qubits)]
    qc = QuantumCircuit(*regs)
    # Each barrier label marks the block that follows it;
    # the initial state is shown by the |0⟩ wire labels
    qc.h(range(qc.num_qubits))
    for k in range(1, reps + 1):
        # $...$ is rendered by matplotlib mathtext; braces only for k >= 10,
        # since Qiskit truncates barrier labels longer than 16 characters
        ks = str(k) if k < 10 else f"{{{k}}}"
        gamma = Parameter(rf"$\gamma_{ks}$")
        beta = Parameter(rf"$\beta_{ks}$")
        qc.barrier(label=rf"$U_C(\gamma_{ks})$")
        _append_cost_layer(qc, ordered, gamma, ks, symbolic)
        qc.barrier(label=rf"$U_M(\beta_{ks})$")
        mixer_angle = Parameter(rf"$2\beta_{ks}$") if symbolic else 2 * beta
        qc.rx(mixer_angle, range(qc.num_qubits))
    qc.barrier(label="Messung")
    for q in range(qc.num_qubits):
        qc.append(Instruction("M", 1, 0, [], label=r"$\mathcal{M}$"), [q])
    return qc


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
    fig, ax = plt.subplots(figsize=(SLIDE_FIG_WIDTH, 5))
    ax.plot(history, marker=".")
    ax.set_xlabel("Iteration (COBYLA)")
    ax.set_ylabel(r"$F_p$ – erwartete Kosten")
    ax.set_title(rf"Konvergenz der Parameteroptimierung $(p = {reps})$")
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def plot_landscape(betas, gammas, raster, start, path: Path) -> None:
    """Energy landscape of the p = 1 grid search (backup slide)."""
    fig, ax = plt.subplots(figsize=(SLIDE_FIG_WIDTH, 5))
    mesh = ax.pcolormesh(gammas, betas, raster, shading="nearest", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=r"$F_p$ – erwartete Kosten")
    ax.plot(start[1], start[0], marker="*", color="red", markersize=14,
            label="Minimum (Raster)")
    ax.axvline(2 * np.pi, color="white", linestyle="--",
               label=r"$\gamma = 2\pi$ (übliches Suchfenster)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"Energielandschaft $F_p(\boldsymbol{\gamma}, \boldsymbol{\beta})$, $p = 1$")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def plot_distribution(pf: Portfolio, prob: dict, cfg: Config, path: Path) -> None:
    """Measurement distribution: feasible portfolios first, sorted by cost."""
    x_opt = pf.feasible[0]
    infeasible = sorted(set(pf.all_x) - set(pf.feasible), key=pf.cost)
    x_sorted = pf.feasible + infeasible
    categories = {"tab:green": "Optimum",
                  "tab:blue": "zulässig",
                  "tab:red": "unzulässig"}
    colors = ["tab:green" if x == x_opt
              else "tab:blue" if sum(x) == pf.budget
    else "tab:red" for x in x_sorted]
    # Horizontal bars: the 16 portfolio names stay readable at presentation
    # font size (vertical bars would need rotated or overlapping labels)
    fig, ax = plt.subplots(figsize=(SLIDE_FIG_WIDTH, 6.5))
    ax.barh([pf.name(x) for x in x_sorted], [prob[x] for x in x_sorted],
            color=colors)
    ax.invert_yaxis()  # best portfolio at the top
    ax.axvline(1 / 2 ** pf.n, color="gray", linestyle="--",
               label=rf"Raten $(1/{2 ** pf.n})$")
    ax.set_xlabel("Messwahrscheinlichkeit")
    # Centred on the figure, not the axes (the long labels shift the axes
    # right). {{\min}}: literal braces; a bare {min} would insert Python's min()
    fig.suptitle(f"Messverteilung nach Optimierung "
                 rf"($p = {cfg.reps}$, $\alpha_{{\min}}$, {cfg.shots} Shots)")
    # Bar colours are not labelled artists, so their legend entries are
    # added as patches in front of the guessing line
    handles = [Patch(color=c, label=label) for c, label in categories.items()]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def _adjust_circuit_axes(ax, box_height: float = 0.9, width_pad: float = 1.1,
                         small_size: float = 0.7,
                         label_shift: float = 0.2) -> None:
    """Post-process Qiskit's circuit figure (not configurable in its drawer).

    - Rotation gates (Rz, Rx): `box_height` tall (Qiskit: fixed 0.65, wire
      spacing = 1) and `width_pad` × the widest of them wide, so the two-line
      labels fit; the label lines move apart.
    - H and M (one-line labels): smaller squares of side `small_size`.
    - All boxes stay centred where Qiskit placed them.
    - Barrier labels (U_C, U_M, Messung): moved `label_shift` further up.
    """

    def is_rotation(s: str) -> bool:
        return s.startswith(r"$\mathrm{R_")

    def is_small(s: str) -> bool:
        return s in ("H", r"$\mathcal{M}$")

    rotations, smalls = [], []  # (box, texts inside it)
    for box in [p for p in ax.patches if type(p) is Rectangle]:
        x0, y0 = box.get_xy()
        w, h = box.get_width(), box.get_height()
        inside = [t for t in ax.texts
                  if x0 <= t.get_position()[0] <= x0 + w
                  and y0 <= t.get_position()[1] <= y0 + h]
        labels = [t.get_text() for t in inside]
        if any(map(is_rotation, labels)):
            rotations.append((box, inside))
        elif any(map(is_small, labels)):
            smalls.append((box, inside))
    rot_width = width_pad * max((b.get_width() for b, _ in rotations), default=0)
    for group, (width, height) in ((rotations, (rot_width, box_height)),
                                   (smalls, (small_size, small_size))):
        for box, inside in group:
            x0, y0 = box.get_xy()
            xc = x0 + box.get_width() / 2
            yc = y0 + box.get_height() / 2
            factor = height / box.get_height()
            box.set_bounds(xc - width / 2, yc - height / 2, width, height)
            for t in inside:
                t.set_y(yc + (t.get_position()[1] - yc) * factor)
    for t in ax.texts:
        if t.get_text().startswith("$U_") or t.get_text() == "Messung":
            t.set_y(t.get_position()[1] + label_shift)


def plot_circuit(h_c: SparsePauliOp, reps: int, ticker, path: Path) -> None:
    """Gate-level QAOA circuit: |0⟩, H, cost layer, mixer, measurement."""
    qc = drawing_circuit(h_c, reps)
    style = {"name": "iqp", "displaycolor": {"M": ("#A0A0A0", "#000000")}}
    # Computer Modern (LaTeX font) for all $...$ math
    with plt.rc_context({"mathtext.fontset": "cm"}):
        # Qiskit sizes the figure from the gate layout (about 22 in for p = 1),
        # not from the font size. Scale the fonts by width / slide width so
        # they reach the CIRCUIT_*_PT sizes when shown at slide width.
        probe = qc.draw("mpl", initial_state=True, fold=-1, style=style)
        # width after bbox_inches="tight" (Qiskit adds white margins)
        k = probe.get_tightbbox().width / SLIDE_FIG_WIDTH
        plt.close(probe)
        style |= {"fontsize": CIRCUIT_GATE_PT * k,
                  "subfontsize": CIRCUIT_PARAM_PT * k}
        fig = qc.draw("mpl", initial_state=True, fold=-1, style=style)
        # Qiskit labels the wires "${x1}$ $|0\rangle$" -> "$x_{1}$ $∣0⟩$".
        # mathtext's \rangle is too big next to "|"; the Unicode glyphs
        # ∣ (U+2223) and ⟩ (U+27E9) have equal height in Computer Modern,
        # like \left|0\right\rangle in LaTeX
        for text in fig.axes[0].texts:
            label = re.sub(r"^\$\{x(\d+)}\$", r"$x_{\1}$", text.get_text())
            text.set_text(label.replace(r"$|0\rangle$", "$∣0⟩$"))
        _adjust_circuit_axes(fig.axes[0])
        mapping = ", ".join(rf"$x_{{{i}}}$={t}"
                            for i, t in enumerate(ticker, start=1))
        fig.suptitle(rf"QAOA-Schaltkreis – {mapping}",
                     fontsize=PRESENTATION_RC["figure.titlesize"] * k)
        fig.savefig(path, dpi=400, bbox_inches="tight")
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
          f"F_C = {result.fun:.4f} nach {len(qaoa.history)} Auswertungen")

    # 3. Measuring with the optimal parameters
    prob = sample(qaoa, result.x, pf, sampler, cfg.shots)

    # 4. Analysis and output
    summary = analyze(pf, prob)
    print_summary(pf, summary)
    save_results(ROOT / "results" / "results.json", cfg, pf, result,
                 len(qaoa.history), summary)

    images = ROOT / "images"
    images.mkdir(exist_ok=True)
    with plt.rc_context(PRESENTATION_RC):
        plot_convergence(qaoa.history, cfg.reps, images / "konvergenz.png")
        plot_landscape(betas, gammas, raster, start,
                       images / "energielandschaft.png")
        plot_distribution(pf, prob, cfg, images / "messverteilung.png")
        plot_circuit(h_c, cfg.reps, pf.ticker, images / "qaoa_schaltkreis.png")
    return summary


if __name__ == "__main__":
    main()
