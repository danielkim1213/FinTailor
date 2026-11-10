"""Statistics for judging a strategy when many ideas were tried and the true edge, if any, is small."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm


def newey_west_t(x: np.ndarray | pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean, robust to autocorrelation in x."""
    v = np.asarray(pd.Series(x).dropna(), float)
    T = len(v)
    if T < 8:
        return float("nan")
    lags = int(4 * (T / 100) ** (2 / 9)) if lags is None else lags
    e = v - v.mean()
    s = e @ e / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / T
    return float(v.mean() / np.sqrt(s / T)) if s > 0 else float("nan")


def block_bootstrap(x: np.ndarray | pd.Series, block: int = 63, n: int = 5000, seed: int = 7, per_year: int = 252) -> dict:
    """Annualised mean with a confidence interval, resampling whole blocks so that streaks stay together."""
    v = np.asarray(pd.Series(x).dropna(), float)
    T = len(v)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, (n, k))
    cum = np.concatenate([[0.0], np.cumsum(v)])
    means = (cum[starts + block] - cum[starts]).sum(axis=1) / (k * block) * per_year
    return {"mean": float(v.mean() * per_year), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "p_positive": float((means > 0).mean())}


def sharpe(x: np.ndarray | pd.Series, per_year: int = 252) -> float:
    v = np.asarray(pd.Series(x).dropna(), float)
    return float(v.mean() / v.std(ddof=1) * np.sqrt(per_year))


def probabilistic_sharpe(x: np.ndarray | pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds benchmark_sr, allowing for skew and fat tails (Bailey & Lopez de Prado)."""
    v = np.asarray(pd.Series(x).dropna(), float)
    sr, T = v.mean() / v.std(ddof=1), len(v)
    g3 = ((v - v.mean()) ** 3).mean() / v.std(ddof=0) ** 3
    g4 = ((v - v.mean()) ** 4).mean() / v.std(ddof=0) ** 4
    return float(norm.cdf((sr - benchmark_sr) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def deflated_sharpe(x: np.ndarray | pd.Series, n_trials: int, sr_std_trials: float) -> float:
    """Probabilistic Sharpe against the best Sharpe that luck alone would give after n_trials attempts.
    sr_std_trials: standard deviation of the per-period Sharpe across the trials."""
    gamma = 0.5772156649
    expected_max = sr_std_trials * ((1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(x, expected_max)


def probability_of_overfitting(perf: pd.DataFrame, splits: int = 10) -> float:
    """Combinatorially symmetric cross-validation: how often the best in-sample candidate lands in the bottom half out of sample.
    perf: one column per candidate, rows are periods."""
    parts = np.array_split(np.arange(len(perf)), splits)
    x = perf.to_numpy(float)
    below, total = 0, 0
    for chosen in combinations(range(splits), splits // 2):
        inn = np.concatenate([parts[i] for i in chosen])
        out = np.concatenate([parts[i] for i in range(splits) if i not in chosen])
        best = np.nanargmax(np.nanmean(x[inn], axis=0))
        oos = np.nanmean(x[out], axis=0)
        below += (oos < oos[best]).mean() < 0.5  # the fraction beaten by the chosen one is under half
        total += 1
    return below / total
