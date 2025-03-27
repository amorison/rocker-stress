"""Follow the model developed by Zhou et al. (2010)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final, override

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TimeSeries:
    times: NDArray[np.float64]
    values: NDArray[np.float64]

    def with_values(self, values: NDArray[np.float64]) -> TimeSeries:
        return TimeSeries(
            times=self.times,
            values=values,
        )

    def neg(self) -> TimeSeries:
        return self.with_values(-self.values)

    def derivate(self) -> TimeSeries:
        return self.with_values(np.gradient(self.values, self.times))


class RockingPattern(ABC):
    @abstractmethod
    def angle(self, ntimes: int) -> TimeSeries:
        """Flip angle time series."""


@dataclass(frozen=True)
class SineRocking(RockingPattern):
    angle_max: float
    period: float

    def angle(self, ntimes: int) -> TimeSeries:
        times = np.linspace(0.0, self.period, ntimes, dtype=np.float64)
        values = self.angle_max * np.sin(2 * np.pi * times / self.period)
        return TimeSeries(times=times, values=values)


@dataclass(frozen=True)
class ConstantRocking(RockingPattern):
    forward_angular_vel: float
    backward_angular_vel: float
    angle_max: float

    def angle(self, ntimes: int) -> TimeSeries:
        n_fwd = int(
            self.backward_angular_vel
            / (self.backward_angular_vel + self.forward_angular_vel)
            * ntimes
        )
        n_back = ntimes - n_fwd
        assert n_fwd > 0 and n_back > 0
        times = np.zeros(ntimes)
        values = np.zeros_like(times)

        dt_fwd = self.angle_max / self.forward_angular_vel
        dt_back = self.angle_max / self.backward_angular_vel

        times[: n_fwd + 1] = np.linspace(0.0, dt_fwd, n_fwd + 1)
        values[: n_fwd + 1] = np.linspace(0.0, self.angle_max, n_fwd + 1)
        times[-n_back:] = np.linspace(dt_fwd, dt_fwd + dt_back, n_back)
        values[-n_back:] = np.linspace(self.angle_max, 0.0, n_back)
        return TimeSeries(times=times, values=values)


class Dish(ABC):
    @property
    @abstractmethod
    def critical_angle(self) -> float:
        """Maximum allowed angle."""

    @property
    @abstractmethod
    def fluid_height(self) -> float:
        """Fluid height."""

    @property
    @abstractmethod
    def cross_length(self) -> float:
        """Length along the direction perpendicular to rocking."""

    @abstractmethod
    def fluid_volume(self, angle: TimeSeries) -> TimeSeries:
        """Fluid volume on the right of the center of the dish."""

    @final
    def fluid_flux(self, angle: TimeSeries) -> TimeSeries:
        if np.any(np.absolute(angle.values) > self.critical_angle):
            raise RuntimeError(f"angle beyond critical value {self.critical_angle}")
        return self.fluid_volume(angle).derivate().neg()


@dataclass(frozen=True)
class RectDish(Dish):
    length: float
    width: float
    height: float

    @property
    def fluid_height(self) -> float:
        return self.height

    @property
    def cross_length(self) -> float:
        return self.width

    @property
    def critical_angle(self) -> float:
        return np.arctan(2 * self.height / self.length)

    def fluid_volume(self, angle: TimeSeries) -> TimeSeries:
        vol = (
            self.width
            * self.length
            / 2
            * (self.height - self.length / 4 * np.tan(angle.values))
        )
        return angle.with_values(vol)


@dataclass(frozen=True)
class CylDish(Dish):
    radius: float
    vol: float

    @property
    def fluid_height(self) -> float:
        return self.vol / (np.pi * self.radius**2)

    @property
    def cross_length(self) -> float:
        return self.radius

    @property
    def critical_angle(self) -> float:
        return np.arctan(self.fluid_height / self.radius)

    def fluid_volume(self, angle: TimeSeries) -> TimeSeries:
        vol = self.radius**2 * (
            np.pi * self.fluid_height / 2 - 2 * self.radius / 3 * np.tan(angle.values)
        )
        return angle.with_values(vol)


@dataclass(frozen=True)
class LubricationLister92:
    dynamic_viscosity: float
    height: float
    width: float

    def shear_stress_bottom(self, flux: TimeSeries) -> TimeSeries:
        # in this model, the velocity profile follows:
        # u(z) = 3 q / (2 b h**3) z (2h - z) = a z (2h - z)
        # Hence, du/dz = 2 a (h - z)
        # and at z = 0, du/dz = 2 a h = 3 q / (b h**2)
        du_dz = 3 / (self.width * self.height**2) * flux.values
        return flux.with_values(self.dynamic_viscosity * du_dz)
