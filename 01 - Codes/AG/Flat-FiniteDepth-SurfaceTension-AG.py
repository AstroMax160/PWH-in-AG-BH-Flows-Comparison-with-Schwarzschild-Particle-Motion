"""
Simulation of finite-depth gravity-capillary perturbations on an inward cylindrical
background flow, WITH surface tension.

Model variables used:
    eta = free-surface perturbation
    phi = surface value of the perturbation velocity potential
    H   = mean fluid depth (bath depth since flat bottom)
    sigma = surface tension [N/m]
    rho = fluid density [kg/m^3]
    gamma = artificial damping layer near boundaries

The finite-depth DtN operator G_H is used. Its derivation is in the notes/report

Diagnostics:
    A diagnostic wavenumber k0 is chosen for the horizon radius and for the trapped turbulence
    requirement. 

    r_t/r_H < 1 ; if this is not met turbulence will be outside the 
    horizon so simulation is not valid. It should also not be too close to 1

Important to note:
    Since phi is used for the perturbation velocity potential, theta is used
    for the angle of cylindrical coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. Input Parameters
# ============================================================

# Physical parameters
g = 9.81                    # gravitational acceleration [m/s^2]
H = 2.0e-3                 # mean fluid depth [m]
nu = 20.0e-6                # kinematic viscosity [m^2/s]
rho = 950.0                 # fluid density [kg/m^3]
sigma = 20.6e-3             # surface tension [N/m]
Re_c = 450.0                # critical Reynolds number for turbulence

# Diagnostic wavenumber
lambda0 = 4.0e-2            # diagnostic wavelength [m]
k0 = 2.0 * np.pi / lambda0  # diagnostic wavenumber [1/m]
c_k0 = np.sqrt((g / k0 + sigma * k0 / rho) * np.tanh(k0 * H))  # finite-depth phase speed [m/s]

# Horizon setup
r_H = 10.0e-2               # desired diagnostic horizon radius for k0 [m]
A = r_H * c_k0              # sink strength parameter, since r_H = A/c_k0

# Shallow speed used for CFL bound
c_shallow = np.sqrt(g * H)
c_max = max(c_shallow, c_k0)

# Inner radius that is excluded (because of the drain)
r_in = 0.1 * r_H            # excluded radius [m]

# Simulation domain
L = 5 * r_H                 # domain is from -L to L in x and y [m]
N = 251                     # grid points in each direction

# Initial perturbation
eta0 = 3.0e-4               # initial amplitude [m]
w = 4e-2                    # gaussian width [m]
x0 = 2 * r_H                # center x-position [m]
y0 = 1.5 * r_H              # center y-position [m]

# Time integration
t_end = 1                # final simulation time [s]
CFL = 0.25                  # CFL factor

# Damping layer settings
outer_damp_width = 0.45 * r_H       # width of outer damping layer [m]
inner_damp_width = 0.20 * r_H       # width of inner damping layer [m]
gamma_max = 10.0                    # maximum damping strength [1/s]

# Plotting / animation
plot_every = 12              # store one frame every this many RK4 steps
interval_ms = 30             # animation speed in milliseconds between frames

# ===============================================================================
# 2. Grid
# ==============================================================================


x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)

# Define active region (outside excluded radius)
active = r >= r_in

# ============================================================
# 3. Background Flow
# ============================================================

# Inward cylindrical flow outside r_in.
# Inside r_in, set velocity to zero because that region is excluded.
u_B = np.zeros_like(X)
v_B = np.zeros_like(Y)

u_B[active] = -A * X[active] / r[active]**2
v_B[active] = -A * Y[active] / r[active]**2

speed_B = np.sqrt(u_B**2 + v_B**2)


# ============================================================
# 4. Turbulence Diagnostic
# ============================================================

r_t = A * H / (nu * Re_c)
rt_over_rH = r_t / r_H

print(f"Turbulence diagnostic: r_t/r_H(k0) = {rt_over_rH:.3f}")


# ============================================================
# 5. Damping Layer
# ============================================================

def smooth_ramp(s):
    """
    Smooth ramp from 0 to 1 for s in [0,1].
    Values below 0 become 0, values above 1 become 1.

    This avoids abrupt damping jumps.
    """
    s = np.clip(s, 0.0, 1.0)
    return s**2 * (3.0 - 2.0 * s)


gamma = np.zeros_like(r)

# Outer damping near the square box edges.
# d_edge is distance to nearest box edge.
d_edge = L - np.maximum(np.abs(X), np.abs(Y))
outer_s = (outer_damp_width - d_edge) / outer_damp_width
gamma += gamma_max * smooth_ramp(outer_s)

# Inner damping near the excluded central radius.
inner_s = (r_in + inner_damp_width - r) / inner_damp_width
gamma += gamma_max * smooth_ramp(inner_s)

# Inside excluded region make damping strong 
gamma[~active] = gamma_max


# ============================================================
# 6. Initial Conditions
# ============================================================

#eta = eta0 * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2.0 * w**2)) * np.cos( k0 * (X - x0)) 
s = np.sqrt((X - x0)**2 + (Y - y0)**2)

eta = eta0 * np.exp(-s**2 / (2.0 * w**2)) * np.cos(k0 * s)
phi = np.zeros_like(eta)

# Excluded center starts with no perturbation
eta[~active] = 0.0
phi[~active] = 0.0


# ==========================================================================
# 7. Finite-Difference Operators & Finite-Ddepth DtN Operator
# ==========================================================================

def ddx_upwind(F, u, dx):
    """
    First derivative dF/dx using first-order upwinding (This was recommended for advection)
    """
    backward = (F - np.roll(F, shift=1, axis=1)) / dx
    forward = (np.roll(F, shift=-1, axis=1) - F) / dx

    # Boundary corrections to avoid periodic wrapping at x-boundaries
    backward[:, 0] = (F[:, 1] - F[:, 0]) / dx
    forward[:, -1] = (F[:, -1] - F[:, -2]) / dx

    return np.where(u >= 0.0, backward, forward)


def ddy_upwind(F, v, dy):
    """
    First derivative dF/dy using first-order upwinding.
    """
    backward = (F - np.roll(F, shift=1, axis=0)) / dy
    forward = (np.roll(F, shift=-1, axis=0) - F) / dy

    # Boundary corrections to avoid periodic wrapping at y-boundaries
    backward[0, :] = (F[1, :] - F[0, :]) / dy
    forward[-1, :] = (F[-1, :] - F[-2]) / dy

    return np.where(v >= 0.0, backward, forward)


def laplacian(F, dx):
    """
    2D Laplacian using centered finite differences.
    """
    lap = (
        np.roll(F, shift=1, axis=1)
        + np.roll(F, shift=-1, axis=1)
        + np.roll(F, shift=1, axis=0)
        + np.roll(F, shift=-1, axis=0)
        - 4.0 * F
    ) / dx**2

    # Boundary: copy adjacent interior values to avoid wrap artifacts
    lap[:, 0] = lap[:, 1]
    lap[:, -1] = lap[:, -2]
    lap[0, :] = lap[1, :]
    lap[-1, :] = lap[-2, :]

    return lap


# Wavenumber arrays for the FFT-based DtN operator.

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
KX, KY = np.meshgrid(kx, ky)
k_abs = np.sqrt(KX**2 + KY**2)

# Fourier symbol of the DtN operator for this setup
DtN_symbol = k_abs * np.tanh(k_abs * H)
DtN_symbol[0, 0] = 0.0


def dtn_finite_depth(F):
    """
    Apply the DtN operator

    In Fourier space:
        Fourier[G_H(F)] = k tanh(k H) Fourier[F]

    where k = sqrt(kx^2 + ky^2).
    """
    F_hat = np.fft.fft2(F)
    GH_F = np.fft.ifft2(DtN_symbol * F_hat)
    return np.real(GH_F)


# ============================================================
# 8. RHS of the PDE
# ============================================================

def rhs(eta_current, phi_current):
    """
    Compute eta_t and phi_t from the finite-depth PDE system.

    eta_t = -u_B eta_x - v_B eta_y + G_H(phi) - gamma eta
    phi_t = -u_B phi_x - v_B phi_y - g eta + (sigma/rho) Laplacian(eta) - gamma phi

    G_H is the DtN operator with Fourier symbol k tanh(kH).
    """

    # Enforce excluded region before taking derivatives and before applying DtN.
    eta_work = eta_current.copy()
    phi_work = phi_current.copy()
    eta_work[~active] = 0.0
    phi_work[~active] = 0.0

    eta_x = ddx_upwind(eta_work, u_B, dx)
    eta_y = ddy_upwind(eta_work, v_B, dy)

    phi_x = ddx_upwind(phi_work, u_B, dx)
    phi_y = ddy_upwind(phi_work, v_B, dy)

    GH_phi = dtn_finite_depth(phi_work)
    lap_eta = laplacian(eta_work, dx)

    eta_t = -u_B * eta_x - v_B * eta_y + GH_phi - gamma * eta_work
    phi_t = (
        -u_B * phi_x
        -v_B * phi_y
        -g * eta_work
        + (sigma / rho) * lap_eta
        - gamma * phi_work
    )

    # Force excluded region to remain inactive
    eta_t[~active] = 0.0
    phi_t[~active] = 0.0

    return eta_t, phi_t


# ============================================================
# 9. RK4 Method (Time stepping the PDE)
# ============================================================

def rk4_step(eta_current, phi_current, dt):
    """
    Fourth-order Runge-Kutta time step.

    This advances:
        eta_t = F_eta(eta, phi)
        phi_t = F_phi(eta, phi)
    """

    k1_eta, k1_phi = rhs(eta_current, phi_current)

    k2_eta, k2_phi = rhs(
        eta_current + 0.5 * dt * k1_eta,
        phi_current + 0.5 * dt * k1_phi,
    )

    k3_eta, k3_phi = rhs(
        eta_current + 0.5 * dt * k2_eta,
        phi_current + 0.5 * dt * k2_phi,
    )

    k4_eta, k4_phi = rhs(
        eta_current + dt * k3_eta,
        phi_current + dt * k3_phi,
    )

    eta_next = eta_current + (dt / 6.0) * (
        k1_eta + 2.0 * k2_eta + 2.0 * k3_eta + k4_eta
    )

    phi_next = phi_current + (dt / 6.0) * (
        k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi
    )

    # Keep excluded center inactive
    eta_next[~active] = 0.0
    phi_next[~active] = 0.0

    return eta_next, phi_next


# ============================================================
# 10. Time step from CFL condition
# ============================================================

max_signal_speed = np.max(speed_B[active]) + c_max
dt = CFL * dx / max_signal_speed

n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps  # adjust so final time lands exactly on t_end

print(f"Diagnostic horizon radius: r_H(k0) = {r_H:.5e} m")
print(f"Diagnostic wavenumber: k0 = {k0:.5e} 1/m")
print(f"Finite-depth with sigma phase speed: c_k0 = {c_k0:.5e} m/s")
print(f"Shallow phase speed: sqrt(gH) = {c_shallow:.5e} m/s")
print(f"Density: rho = {rho:.5e} kg/m^3")
print(f"Surface tension: sigma = {sigma:.5e} N/m")
print(f"Time step: dt = {dt:.5e} s")
print(f"Number of RK4 steps: {n_steps}")


# ============================================================
# 11. Run Simulation and Store Frames
# ============================================================

frames = []
times = []

for n in range(n_steps + 1):
    if n % plot_every == 0:
        frames.append(eta.copy())
        times.append(n * dt)

    eta, phi = rk4_step(eta, phi, dt)


# ============================================================
# 12. Animation
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

extent = [-L, L, -L, L]

# Choose a symmetric color scale based on initial amplitude.
# You can adjust this if later waves become much weaker/stronger.
vmax = eta0
vmin = -eta0

im = ax.imshow(
    frames[0],
    extent=extent,
    origin="lower",
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    interpolation="bilinear",
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"$\eta(x,y,t)$ [m]")

# Overlay diagnostic circles
theta = np.linspace(0.0, 2.0 * np.pi, 600)

# Diagnostic horizon circle for the selected wavenumber k0
ax.plot(
    r_H * np.cos(theta),
    r_H * np.sin(theta),
    "k-",
    linewidth=1.8,
    label=rf"Diagnostic horizon radius $r_H$(k0)",
)

# Turbulence diagnostic circle
ax.plot(
    r_t * np.cos(theta),
    r_t * np.sin(theta),
    "k--",
    linewidth=1.4,
    label=r"Turbulence radius $r_t$",
)

# Inner excluded radius
ax.plot(
    r_in * np.cos(theta),
    r_in * np.sin(theta),
    "k:",
    linewidth=1.4,
    label=r"Excluded radius $r_{\rm in}$",
)


# Damping-region markers

# Inner damping acts roughly between r_in and r_in + inner_damp_width.
r_inner_damp_outer = r_in + inner_damp_width

ax.plot(
    r_inner_damp_outer * np.cos(theta),
    r_inner_damp_outer * np.sin(theta),
    color="gray",
    linestyle="-.",
    linewidth=1.2,
    label="Damping region markers",
)

# Outer damping acts near the square-box boundary.
# This square marks where the outer damping region begins.
L_undamped = L - outer_damp_width

outer_damp_square_x = [
    -L_undamped, L_undamped, L_undamped, -L_undamped, -L_undamped
]
outer_damp_square_y = [
    -L_undamped, -L_undamped, L_undamped, L_undamped, -L_undamped
]

ax.plot(
    outer_damp_square_x,
    outer_damp_square_y,
    color="gray",
    linestyle="-.",
    linewidth=1.2,
)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_aspect("equal")
ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.7), frameon=True)

title = ax.set_title("")


def update(frame_index):
    im.set_data(frames[frame_index])
    title.set_text(f"Surface perturbation, t = {times[frame_index]:.3f} s")
    return im, title


ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=interval_ms,
    blit=False,
)

# plt.tight_layout()
plt.show()
