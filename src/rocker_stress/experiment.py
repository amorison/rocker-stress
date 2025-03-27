from __future__ import annotations

from dataclasses import dataclass

import matplotlib.axes as mpla
from lazympl.plot import Plot

from .zhou2010 import Dish, LubricationLister92, RockingPattern, TimeSeries


@dataclass(frozen=True)
class Experiment:
    rocking: RockingPattern
    dish: Dish
    viscosity: float

    def shear_stress(self, ntimes: int) -> TimeSeries:
        lub = LubricationLister92(
            dynamic_viscosity=self.viscosity,
            height=self.dish.fluid_height,
            width=self.dish.cross_length,
        )
        flip = self.rocking.angle(ntimes)
        flux = self.dish.fluid_flux(flip)
        return lub.shear_stress_bottom(flux)


@dataclass(frozen=True)
class ShearStressPlot(Plot):
    experiment: Experiment
    ntimes: int

    def draw_on(self, ax: mpla.Axes) -> None:
        stress = self.experiment.shear_stress(ntimes=self.ntimes)
        ax.plot(stress.times, stress.values)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("stress (Pa)")
