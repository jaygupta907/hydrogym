import firedrake as fd
import firedrake_adjoint as fda
from ufl import sin

import hydrogym as hg


def test_import_medium():
    flow = hg.flow.Cavity(mesh="medium")
    return flow


def test_import_fine():
    flow = hg.flow.Cavity(mesh="fine")
    return flow


def test_steady(tol=1e-3):
    flow = hg.flow.Cavity(Re=500, mesh="medium")
    flow.solve_steady()

    (y,) = flow.get_observations()
    assert abs(y - 2.2122) < tol  # Re = 500


def test_actuation():
    flow = hg.flow.Cavity(Re=500, mesh="medium")
    flow.set_control(1.0)
    flow.solve_steady()


def test_step():
    flow = hg.flow.Cavity(Re=500, mesh="medium")
    dt = 1e-4

    solver = hg.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow = solver.step(iter)


def test_integrate():
    flow = hg.flow.Cavity(Re=500, mesh="medium")
    dt = 1e-4

    hg.integrate(flow, t_span=(0, 10 * dt), dt=dt, method="IPCS")


def test_control():
    flow = hg.flow.Cavity(Re=500, mesh="medium")
    dt = 1e-4

    solver = hg.ts.IPCS(flow, dt=dt)

    num_steps = 10
    for iter in range(num_steps):
        flow.get_observations()
        flow = solver.step(iter, control=0.1 * sin(solver.t))


def test_env():
    env_config = {"Re": 500, "mesh": "medium"}
    env = hg.env.CavityEnv(env_config)

    for _ in range(10):
        y, reward, done, info = env.step(0.1 * sin(env.solver.t))


def test_grad():
    flow = hg.flow.Cavity(Re=500, mesh="medium")

    c = fda.AdjFloat(0.0)
    flow.set_control(c)

    flow.solve_steady()
    (y,) = flow.get_observations()

    fda.compute_gradient(y, fda.Control(c))


def test_sensitivity(dt=1e-2, num_steps=10):
    from ufl import dx, inner

    flow = hg.flow.Cavity(Re=500, mesh="medium")

    # Store a copy of the initial condition to distinguish it from the time-varying solution
    q0 = flow.q.copy(deepcopy=True)
    flow.q.assign(
        q0, annotate=True
    )  # Note the annotation flag so that the assignment is tracked

    # Time step forward as usual
    flow = hg.ts.integrate(flow, t_span=(0, num_steps * dt), dt=dt, method="IPCS_diff")

    # Define a cost functional... here we're just using the energy inner product
    J = 0.5 * fd.assemble(inner(flow.u, flow.u) * dx)

    # Compute the gradient with respect to the initial condition
    #   The option for Riesz representation here specifies that we should end up back in the primal space
    fda.compute_gradient(J, fda.Control(q0), options={"riesz_representation": "L2"})


def test_env_grad():
    env_config = {"Re": 500, "differentiable": True, "mesh": "medium"}
    env = hg.env.CavityEnv(env_config)
    y = env.reset()
    omega = fda.AdjFloat(1.0)
    A = 0.1
    J = fda.AdjFloat(0.0)
    for _ in range(10):
        y, reward, done, info = env.step(A * sin(omega * env.solver.t))
        J = J - reward
    dJdm = fda.compute_gradient(J, fda.Control(omega))
    print(dJdm)


if __name__ == "__main__":
    test_import_medium()
