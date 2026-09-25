# %% [markdown]
# # Datenbeschaffung – Portfolio-Instanz
# Aktien wie bei Baker et al. (2022): GOOG, AMZN, FB (heute META), NVDA
# Zeitraum: Kalenderjahr 2025 (02.01. – 31.12.2025), letztes vollständiges Jahr
# Ergebnis: mu (erwartete Renditen) und sigma (Kovarianzmatrix),
#           eingefroren in portfolio_daten.npz für alle weiteren Schritte

# %%
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = ["GOOG", "AMZN", "META", "NVDA"]
START, ENDE = "2025-01-01", "2026-01-01"  # Ende ist bei yfinance exklusiv
HANDELSTAGE_PRO_JAHR = 252

# %% Kurse laden (dividenden-/splitbereinigte Schlusskurse)
kurse = yf.download(TICKER, start=START, end=ENDE,
                    auto_adjust=True, progress=False)["Close"][TICKER]
print(f"{len(kurse)} Handelstage geladen")

# %% Tägliche Renditen, mu und sigma
renditen = kurse.pct_change().dropna()  # 249 Tagesrenditen
mu_tag = renditen.mean().to_numpy()
sigma_tag = renditen.cov().to_numpy()

# Annualisierung (Hodson et al. 2019): ändert die optimale Lösung nicht,
# da Risiko- und Renditeterm mit demselben Faktor skaliert werden, sorgt aber
# für sinnvolle Größenordnungen der Winkel im Schaltkreis.
mu = mu_tag * HANDELSTAGE_PRO_JAHR
sigma = sigma_tag * HANDELSTAGE_PRO_JAHR

# %% Tabelle für Folie 10 (anschauliche Größen)
tabelle = pd.DataFrame({
  "Rendite 2025": kurse.iloc[-1] / kurse.iloc[0] - 1,
  "Volatilität p.a.": renditen.std() * np.sqrt(HANDELSTAGE_PRO_JAHR),
})
print(tabelle.map(lambda v: f"{v:.1%}"))
print("\nKorrelationen:\n", renditen.corr().round(2))

# %% Daten einfrieren – alle weiteren Schritte lesen nur noch diese Datei
np.savez("portfolio_daten.npz", ticker=TICKER, mu=mu, sigma=sigma)
kurse.to_csv("kurse_2025.csv")
print("\nGespeichert: portfolio_daten.npz, kurse_2025.csv")
