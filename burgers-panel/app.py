import numpy as np
import panel as pn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clawpack import pyclaw
from clawpack import riemann

pn.extension()

# -----------------------------
# Exact Riemann Solution
# -----------------------------
def exact_solution(x, t, uL, uR):
    if t == 0:
        return np.where(x < 0, uL, uR)

    xi = x / t
    u = np.zeros_like(x)

    if uL < uR:  # Rarefaction
        u[xi < uL] = uL
        u[xi > uR] = uR
        mask = (xi >= uL) & (xi <= uR)
        u[mask] = xi[mask]

    else:  # Shock
        s = 0.5 * (uL + uR)
        u[x < s * t] = uL
        u[x >= s * t] = uR

    return u


# -----------------------------
# PyClaw Solver
# -----------------------------
def solve_burgers(uL, uR, tfinal=0.5, nx=400, nframes=50):

    solver = pyclaw.ClawSolver1D(riemann.burgers_1D)
    solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap

    x = pyclaw.Dimension(-1.0, 1.0, nx, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, 1)

    xc = state.grid.x.centers
    state.q[0, :] = np.where(xc < 0, uL, uR)

    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.num_output_times = nframes
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver

    claw.run()

    return xc, claw.frames

solution_cache = {}

def compute_solution(uL, uR):
    global solution_cache
    solution_cache['x'], solution_cache['frames'] = solve_burgers(uL, uR)

# -----------------------------
# Interactive Plot
# -----------------------------
def plot_solution(uL, uR, frame_index):

    if 'uL' not in solution_cache or solution_cache.get('uL') != uL or solution_cache.get('uR') != uR:
        solution_cache['uL'] = uL
        solution_cache['uR'] = uR
        compute_solution(uL, uR)

        # Update time slider max dynamically
        time_slider.end = len(solution_cache['frames']) - 1
        time_slider.value = 0

    x = solution_cache['x']
    frames = solution_cache['frames']

    frame = frames[frame_index]
    u_num = frame.q[0, :]
    t = frame.t

    u_exact = exact_solution(x, t, uL, uR)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, u_num, label="Numerical")
    ax.plot(x, u_exact, '--', label="Exact")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Burgers Riemann | t = {t:.3f}")
    ax.legend()
    ax.grid(True)

    plt.close(fig)   # Prevent mem accumulation

    return fig

# -----------------------------
# Panel Widgets
# -----------------------------
uL_slider = pn.widgets.FloatInput(name="u_L", value=1.0)
uR_slider = pn.widgets.FloatInput(name="u_R", value=2.0)

time_slider = pn.widgets.IntSlider(name="Time Index", start=0, end=50, step=1, value=0)

interactive_plot = pn.bind(plot_solution, uL_slider, uR_slider, time_slider)

layout = pn.Column(
    "# 1D Burgers Riemann Problem",
    "Adjust u_L and u_R. Use time slider to see evolution.",
    uL_slider,
    uR_slider,
    time_slider,
    interactive_plot
)

layout.servable()