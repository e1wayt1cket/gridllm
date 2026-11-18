"""Explicit inputs for the local day-ahead mechanism."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StorageUnit:
    name: str
    bus: int
    power_mw: float = .4
    energy_mwh: float = 1.6
    eta_charge: float = .92
    eta_discharge: float = .92
    soc_min: float = .1
    soc_max: float = .9
    soc_initial: float = .5
    degradation_cny_mwh: float = 100.


@dataclass(frozen=True)
class Generator:
    name: str
    bus: int
    available_mw: np.ndarray
    offer_cny_mwh: float = 0.
    real_cost_cny_mwh: float = 0.
    reactive_limit_mvar: float = 0.


@dataclass(frozen=True)
class StorageBid:
    charge_quantity_mw: np.ndarray
    discharge_quantity_mw: np.ndarray
    charge_price_cny_mwh: np.ndarray
    discharge_price_cny_mwh: np.ndarray


@dataclass(frozen=True)
class RadialNetwork:
    buses: int
    slack: int
    branches: tuple
    r_mw_base: np.ndarray
    x_mw_base: np.ndarray
    current_sq_limit: np.ndarray | None = None
    voltage_min: float = .93
    voltage_max: float = 1.07
    # The substation voltage the market holds; a tap changer sets it in practice.
    slack_vm_pu: float = 1.
    # Fixed shunt compensation carried by the feeder, not by a participant:
    # (bus, injected MVAr) pairs, positive when capacitive.
    shunts: tuple = ()


@dataclass(frozen=True)
class DayAheadInput:
    network: RadialNetwork
    active_load_mw: np.ndarray
    reactive_load_mvar: np.ndarray
    external_price_cny_mwh: np.ndarray
    storage: tuple
    generators: tuple
    bids: dict
    bid_price_lower: float
    bid_price_upper: float
    dt_hours: float = .25
    import_limit_mw: float | None = None
    export_limit_mw: float | None = 2.5
    terminal_soc_equal: bool = False

    @property
    def periods(self):
        return self.active_load_mw.shape[0]


def vector(value, periods, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (periods,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite curve with {periods} periods")
    return result


def validate_day(day):
    net = day.network
    load = np.asarray(day.active_load_mw)
    reactive = np.asarray(day.reactive_load_mvar)
    if (load.ndim != 2 or load.shape[0] == 0 or load.shape[1] != net.buses
            or reactive.shape != load.shape or not np.isfinite(load).all()
            or not np.isfinite(reactive).all() or (load < 0).any()):
        raise ValueError("Loads must be finite matching time-by-bus matrices with nonnegative P")
    if not np.isfinite(day.dt_hours) or day.dt_hours <= 0:
        raise ValueError("Period duration must be positive")
    if (not 0 <= net.slack < net.buses or not 0 < net.voltage_min < 1 < net.voltage_max
            or not net.voltage_min < net.slack_vm_pu <= net.voltage_max
            or len(net.branches) != net.buses - 1):
        raise ValueError("A connected radial network and a voltage band containing the slack are required")
    for shunt in net.shunts:
        if (not isinstance(shunt, tuple) or len(shunt) != 2
                or not isinstance(shunt[0], (int, np.integer)) or not 0 <= shunt[0] < net.buses
                or not np.isfinite(shunt[1])):
            raise ValueError("A shunt must name one bus inside the network with a finite MVAr")
    reached = {net.slack}
    pending = list(net.branches)
    while pending:
        progress = False
        for edge in pending.copy():
            start, end = edge
            if not 0 <= start < net.buses or not 0 <= end < net.buses:
                raise ValueError("Branch bus is outside the network")
            if start in reached:
                if end in reached:
                    raise ValueError("Branches must form a directed radial tree")
                reached.add(end)
                pending.remove(edge)
                progress = True
        if not progress:
            raise ValueError("Branches must be oriented away from the slack")
    for value, label in [(net.r_mw_base, "resistance"), (net.x_mw_base, "reactance")]:
        array = np.asarray(value)
        if array.shape != (len(net.branches),) or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"Invalid branch {label}")
    if net.current_sq_limit is not None:
        values = np.asarray(net.current_sq_limit)
        if values.shape != (len(net.branches),) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid explicit branch current limits")
    vector(day.external_price_cny_mwh, day.periods, "External price")
    if not np.isfinite([day.bid_price_lower, day.bid_price_upper]).all() or day.bid_price_lower > day.bid_price_upper:
        raise ValueError("An explicit finite ordered research bid price interval is required")
    for bound in [day.import_limit_mw, day.export_limit_mw]:
        if bound is not None and (not np.isfinite(bound) or bound < 0):
            raise ValueError("PCC limits must be nonnegative or None")
    names = [unit.name for unit in day.storage]
    if len(set(names)) != len(names) or set(day.bids) != set(names):
        raise ValueError("Every independent storage unit requires exactly one distinct bid")
    for unit in day.storage:
        params = [unit.power_mw, unit.energy_mwh, unit.eta_charge, unit.eta_discharge,
                  unit.soc_min, unit.soc_max, unit.soc_initial, unit.degradation_cny_mwh]
        if (not np.isfinite(params).all() or not 0 <= unit.bus < net.buses
                or unit.power_mw <= 0 or unit.energy_mwh <= 0
                or not 0 < unit.eta_charge <= 1 or not 0 < unit.eta_discharge <= 1
                or not 0 <= unit.soc_min <= unit.soc_initial <= unit.soc_max <= 1
                or unit.degradation_cny_mwh < 0):
            raise ValueError("Invalid independent storage device")
        bid = day.bids[unit.name]
        for label in ["charge_quantity_mw", "discharge_quantity_mw"]:
            values = vector(getattr(bid, label), day.periods, label)
            if (values < 0).any() or (values > unit.power_mw).any():
                raise ValueError("Bid quantity is outside the device power rating")
        for label in ["charge_price_cny_mwh", "discharge_price_cny_mwh"]:
            values = vector(getattr(bid, label), day.periods, label)
            if (values < day.bid_price_lower).any() or (values > day.bid_price_upper).any():
                raise ValueError("Bid price is outside the declared research interval")
    generator_names = set()
    for generator in day.generators:
        if generator.name in generator_names or generator.name in names:
            raise ValueError("Participant names must be unique")
        generator_names.add(generator.name)
        available = vector(generator.available_mw, day.periods, "Generation availability")
        if (not 0 <= generator.bus < net.buses or (available < 0).any()
                or not np.isfinite([generator.offer_cny_mwh, generator.real_cost_cny_mwh,
                                    generator.reactive_limit_mvar]).all()
                or generator.reactive_limit_mvar < 0):
            raise ValueError("Invalid generation input")
