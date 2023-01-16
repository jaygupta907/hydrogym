import numpy as np
from common import Re, flow
from firedrake.petsc import PETSc
from ufl import real

import hydrogym as hgym

assert (
    PETSc.ScalarType == np.complex128
), "Complex PETSc configuration required for stability analysis"

# First we have to ramp up the Reynolds number to get the steady state
Re_init = [500, 1000, 2000, 4000, Re]

for (i, R) in enumerate(Re_init):
    flow.Re.assign(real(R))
    hgym.print(f"Steady solve at Re={Re_init[i+1]}")
    qB = flow.solve_steady(solver_parameters={"snes_monitor": None})


def least_stable(flow, Re):
    flow.Re.assign(real(Re))
    qB = flow.solve_steady(solver_parameters={"snes_monitor": None})
    A, M = flow.linearize(qB)

    evals, es = hgym.linalg.eig(A, M, num_eigenvalues=20, sigma=7.5j)
    max_idx = np.argmax(np.real(evals))

    hgym.print(f"Re={Re}:\t\tlargest: {evals[max_idx]}")
    return evals[max_idx]


# Bisection search
#  Sipp & Lebedev: Re=4140, omega=7.5
#  Bengana et al:  Re=4126
#  Meliga:         Re=4114
#  Hydrogym:       Re=4134, omega=7.49
Re_lo = 4000
Re_hi = 4200
sigma = np.real(least_stable(flow, 0.5 * (Re_hi + Re_lo)))
while abs(sigma) > 1e-6:
    Re_mid = 0.5 * (Re_hi + Re_lo)
    hgym.print((Re_lo, Re_mid, Re_hi))
    sigma = np.real(least_stable(flow, Re_mid))
    if sigma > 0:
        Re_hi = Re_mid
    else:
        Re_lo = Re_mid
