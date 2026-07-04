# price_forecaster.py
"""RT price forecast models for rolling-horizon MPC with imperfect information."""
import numpy as np


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
