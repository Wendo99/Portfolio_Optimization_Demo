"""Data acquisition – portfolio instance.

Stocks as in Baker et al. (2022): GOOG, AMZN, FB (now META), NVDA.
Period: 31.12.2024 close as base → full-year 2025 returns
Result: mu (expected returns) and sigma (covariance matrix),
        frozen in portfolio_daten.npz for all further steps.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    ticker: tuple[str, ...] = ("GOOG", "AMZN", "META", "NVDA")
    start: str = "2024-12-31"
    end: str = "2026-01-01"  # end is exclusive in yfinance
    trading_days_per_year: int = 252
    datafile: str = "portfolio_daten.npz"
    pricefile: str = "kurse_2025.csv"


def load_prices(cfg: Config) -> pd.DataFrame:
    """Dividend-/split-adjusted closing prices, columns in cfg.ticker order."""
    downloaded = yf.download(list(cfg.ticker), start=cfg.start, end=cfg.end,
                             auto_adjust=True, progress=False)
    if downloaded is None:
        raise RuntimeError("yfinance returned no price data")
    prices = downloaded["Close"]
    return prices[list(cfg.ticker)]


def annualized_moments(returns: pd.DataFrame,
                       cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Annualization (Hodson et al. 2019): does not change the optimal solution,
    since risk and return term are scaled by the same factor, but gives
    sensible magnitudes for the angles in the circuit.
    """
    mu = returns.mean().to_numpy() * cfg.trading_days_per_year
    sigma = returns.cov().to_numpy() * cfg.trading_days_per_year
    return mu, sigma


def slide_table(prices: pd.DataFrame, returns: pd.DataFrame,
                cfg: Config) -> pd.DataFrame:
    """Illustrative figures for slide 10."""
    return pd.DataFrame({
        "Rendite 2025": prices.iloc[-1] / prices.iloc[0] - 1,
        "Volatilität p.a.": returns.std() * np.sqrt(cfg.trading_days_per_year),
    })


def save(cfg: Config, prices: pd.DataFrame, mu: np.ndarray,
         sigma: np.ndarray) -> None:
    """Freeze the data – all further steps only read these files."""
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    np.savez(data_dir / cfg.datafile, ticker=list(cfg.ticker), mu=mu, sigma=sigma)
    prices.to_csv(data_dir / cfg.pricefile)


def main(cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    prices = load_prices(cfg)
    returns = prices.pct_change().dropna()
    print(f"{len(prices)} Handelstage geladen, {len(returns)} Tagesrenditen")

    mu, sigma = annualized_moments(returns, cfg)

    print(slide_table(prices, returns, cfg).map(lambda v: f"{v:.1%}"))
    print("\nKorrelationen:\n", returns.corr().round(2))

    save(cfg, prices, mu, sigma)
    print(f"\nGespeichert: {cfg.datafile}, {cfg.pricefile}")


if __name__ == "__main__":
    main()
