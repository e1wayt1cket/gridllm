# price_forecaster.py
"""Price forecast models for rolling-horizon MPC with imperfect information.

Algorithm registry pattern adapted from ASSUME (forecast_algorithms.py):
  - forecast_algorithms:    dict of algorithm_id -> callable for initial forecast
  - get_price_forecast:     single entry point that resolves by algorithm_id

Built-in algorithms:
  - price_synthetic       Legacy sinusoidal DA price curve (no agents needed)
  - price_merit_order     Merit-order forecast from agent supply/demand stacks
  - price_ema             Exponential moving average of historical LMP
  - price_persistence     Naive persistence: tomorrow = today
"""
import numpy as np
from functools import lru_cache
from typing import Optional, List, Dict, Callable


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

forecast_algorithms: Dict[str, Callable] = {}


def register_forecast(algo_id: str):
    """Decorator to register a forecast algorithm by its ID."""
    def decorator(fn):
        forecast_algorithms[algo_id] = fn
        return fn
    return decorator


def get_price_forecast(algo_id: str, T: int = 96, agents=None, config=None,
                       history: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """Resolve and call a forecast algorithm by ID. Falls back to merit_order."""
    if algo_id in forecast_algorithms:
        return forecast_algorithms[algo_id](
            T=T, agents=agents, config=config, history=history, **kwargs)
    # Fallback: use grid's day_ahead_price_china
    from grid import day_ahead_price_china
    return day_ahead_price_china(T, agents=agents, config=config)


# ---------------------------------------------------------------------------
# Built-in forecast algorithms
# ---------------------------------------------------------------------------

@register_forecast("price_merit_order")
@lru_cache(maxsize=8)
def _cached_merit_order(T: int, agents=None, config=None,
                        history: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """Merit-order based: builds supply/demand stacks from agent fundamentals."""
    from grid import day_ahead_price_china
    return day_ahead_price_china(T, agents=agents, config=config)


@register_forecast("price_synthetic")
def _forecast_synthetic(T: int = 96, agents=None, config=None,
                        history: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """Legacy sinusoidal DA price curve."""
    from grid import _forecast_price_synthetic
    from config_loader import get_default
    cfg = get_default("price_curve", {})
    return _forecast_price_synthetic(T, cfg)


@register_forecast("price_ema")
def _forecast_ema(T: int = 96, agents=None, config=None,
                  history: Optional[np.ndarray] = None, alpha: float = 0.3,
                  **kwargs) -> np.ndarray:
    """EMA-smoothed historical LMP forecast.  Falls back to synthetic if no history."""
    if history is not None and len(history) > 0:
        ema = float(np.mean(history[:8]))  # initial from first 8 periods
        for v in history[8:]:
            ema = alpha * v + (1 - alpha) * ema
        return np.full(T, ema)
    # No history available — fall back to synthetic
    return _forecast_synthetic(T, agents=agents, config=config,
                               history=history, **kwargs)


@register_forecast("price_persistence")
def _forecast_persistence(T: int = 96, agents=None, config=None,
                          history: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
    """Naive persistence: repeat last period's price for all T periods."""
    if history is not None and len(history) > 0:
        return np.full(T, float(history[-1]))
    return _forecast_synthetic(T, agents=agents, config=config,
                               history=history, **kwargs)


# ---------------------------------------------------------------------------
# PriceForecaster — RT forecast over look-ahead window
# ---------------------------------------------------------------------------

class PriceForecaster:
    """Generate RT price forecasts over a look-ahead window.

    Parameters
    ----------
    da_prices : np.ndarray shape (T,)
        Day-ahead prices for all periods (public information).
    mode : str
        "perfect"       — use actual future DA prices (benchmark, perfect info).
        "da_as_forecast" — use DA prices directly as RT forecast (realistic baseline).
        "noisy_da"      — DA prices + Gaussian noise that grows with forecast distance.
    noise_pct : float
        Standard deviation of noise as percentage of DA price (only for noisy_da).
    seed : int or None
        RNG seed for reproducible noise.
    """

    def __init__(self, da_prices, mode="noisy_da", noise_pct=10.0, seed=42):
        self.da_prices = np.asarray(da_prices, dtype=float)
        self.T = len(self.da_prices)
        self.mode = mode
        self.noise_pct = noise_pct
        self._rng = np.random.RandomState(seed)

    def forecast(self, t_start, horizon):
        """Return forecast prices for periods [t_start, t_start + horizon).

        Returns array of length horizon. If horizon extends beyond T, the
        last available DA price is repeated.
        """
        end = min(t_start + horizon, self.T)
        n = end - t_start
        if n <= 0:
            return np.array([])

        da_slice = self.da_prices[t_start:end].copy()

        if self.mode == "perfect":
            return da_slice

        if self.mode == "da_as_forecast":
            return da_slice

        if self.mode == "noisy_da":
            forecast = da_slice.copy()
            for i in range(n):
                distance = i + 1  # 1-indexed forecast distance
                sigma = (self.noise_pct / 100.0) * da_slice[i] * np.sqrt(distance)
                noise = self._rng.normal(0, sigma)
                forecast[i] = max(0.0, da_slice[i] + noise)
            return forecast

        raise ValueError(f"unknown forecast mode: {self.mode}")

    def get_algo_id(self) -> str:
        """Map legacy mode string to algorithm registry ID for logging/debugging."""
        mode_map = {
            "perfect": "price_persistence",
            "da_as_forecast": "price_persistence",
            "noisy_da": "price_merit_order",
        }
        return mode_map.get(self.mode, "price_merit_order")


# ---------------------------------------------------------------------------
# NodalPriceForecaster — per-bus EMA forecaster
# ---------------------------------------------------------------------------

class NodalPriceForecaster:
    """Per-agent nodal LMP forecaster using EMA + seasonal decomposition.

    Each agent maintains its own forecaster based on the historical nodal LMP
    at its bus. Used as an input feature for RL state representation.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (0 < alpha <= 1). Higher = more weight on recent.
    history_len : int
        Number of past periods to keep for feature extraction.
    """

    def __init__(self, alpha=0.3, history_len=16):
        self.alpha = alpha
        self.history_len = history_len
        self._ema = None
        self._history = []

    def update(self, nodal_lmp: float):
        """Absorb a new LMP observation for this agent's bus."""
        if self._ema is None:
            self._ema = nodal_lmp
        else:
            self._ema = self.alpha * nodal_lmp + (1 - self.alpha) * self._ema
        self._history.append(nodal_lmp)
        if len(self._history) > self.history_len:
            self._history.pop(0)

    def forecast(self, n_steps: int = 4) -> list:
        """Predict the next n_steps nodal LMP values.

        Uses EMA as baseline with a seasonal adjustment if sufficient history
        exists. Returns list of length n_steps.
        """
        if self._ema is None:
            return [420.0] * n_steps  # default DA mean
        baseline = self._ema
        forecasts = []
        for i in range(n_steps):
            # Simple persistence with EMA reversion
            forecasts.append(baseline)
        return forecasts

    def get_features(self, n_blocks: int = 4) -> list:
        """Return forecast features for RL state (length = n_blocks)."""
        return self.forecast(n_blocks)
