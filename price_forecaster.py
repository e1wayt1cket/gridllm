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

    def __init__(self, da_prices, mode="perfect", noise_pct=10.0, seed=42):
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
