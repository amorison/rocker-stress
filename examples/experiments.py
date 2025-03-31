from __future__ import annotations

import numpy as np
from lazympl.figure import SinglePlotFigure

from rocker_stress.experiment import Experiment, ShearStressPlot
from rocker_stress.zhou2010 import (
    ConstantRocking,
    CylDish,
    Dish,
    RockingPattern,
    SineRocking,
)


def main(
    name: str,
    rocking: RockingPattern,
    dish: Dish,
    viscosity: float,
) -> None:
    exp = Experiment(
        rocking=rocking,
        dish=dish,
        viscosity=viscosity,
    )
    SinglePlotFigure(
        plot=ShearStressPlot(
            experiment=exp,
            ntimes=100,
        )
    ).save_to(f"stress_{name}.pdf")


if __name__ == "__main__":
    rockings: dict[str, RockingPattern] = {
        "sine": SineRocking(angle_max=np.radians(5), period=10),
        "constant": ConstantRocking(
            forward_angular_vel=np.radians(5),
            backward_angular_vel=np.radians(1),
            angle_max=np.radians(15),
        ),
    }
    dishes: dict[str, Dish] = {
        "cyl": CylDish(radius=2.1e-2 / 2, vol=1e-6),
    }
    for rname, rpattern in rockings.items():
        for dname, dish in dishes.items():
            main(
                name=f"{rname}_{dname}",
                rocking=rpattern,
                dish=dish,
                viscosity=0.9e-3,
            )
