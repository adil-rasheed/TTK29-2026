"""Course benchmark problems, shared across every module.

Three physics-based reference models with reference solvers:

    vessel         -- 3-DOF surface-vessel maneuvering ODE (Fossen)
    turbine        -- variable-speed variable-pitch wind-turbine rotor ODE
    heat           -- 2-D heat-diffusion PDE (explicit FTCS)
    cable          -- radial heat conduction in a power-cable cross-section
    navier_stokes  -- 2-D incompressible Navier-Stokes (cavity + vortex merger)

Later modules (data-driven, PINN, DeepONet, FNO, CoSTA, equation discovery,
ROM) import these so that every method is tested on the *same* problems with
the *same* ground truth.

Example
-------
>>> from pyutils.benchmarks import vessel
>>> t, Z = vessel.simulate(vessel.turning_circle(), t_end=40.0)
"""
from . import vessel, turbine, heat, cable, navier_stokes

__all__ = ["vessel", "turbine", "heat", "cable", "navier_stokes"]
