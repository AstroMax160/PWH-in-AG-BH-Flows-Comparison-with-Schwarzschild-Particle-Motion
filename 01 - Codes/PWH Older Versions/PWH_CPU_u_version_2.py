"""
CPU simulation of a single walking droplet using a Milewski-style PDE wave model.
This version has Gaussian pressure patch, eliminated unnecessary animation code
and made everything non-dimensional when solving
"""

import numpy as np
from pathlib import Path


# ============================================================
# 1. Input Parameters
# ============================================================

# Physical constants
rho = 949.0                  # liquid density [kg/m^3]
sigma = 20.6e-3              # surface tension [N/m]
nu = 2.0e-5                  # liquid kinematic viscosity [m^2/s]
mu_air = 1.8e-5              # air dynamic viscosity [kg/(m s)]
g = 9.8                      # gravitational acceleration [m/s^2]

# Bath vibration / Faraday parameters.
f0 = 80.0                    # driving frequency [Hz]
omega0 = 2.0 * np.pi * f0    # driving angular frequency [rad/s]
Gamma = 3.4                  # vibration forcing used in the wave equation [-]
GammaF = 4.22                # Faraday threshold [-]

# Faraday scale from Milewski Fig. 1.
kF = 12.52e2                 # Faraday wavenumber [1/m]
length_scale = 2.0 * np.pi / kF   # Faraday wavelength [m]
time_scale = 1.0 / 40.0           # Faraday period [s]

# Droplet parameters.
R0 = 0.38e-3                 # droplet radius [m]
m = (4.0 / 3.0) * np.pi * rho * R0**3  # droplet mass [kg]
R0_nd = R0 / length_scale    # nondimensional droplet radius [-]

# Initial droplet state, specified dimensionally for readability and then converted.
x_drop0_phys = 0.0           # initial x-position [m]
y_drop0_phys = 0.0           # initial y-position [m]
u_drop0_phys = 4.0e-2        # initial horizontal x-velocity [m/s]
v_drop0_phys = 0.0           # initial horizontal y-velocity [m/s]
z_drop0_phys = 1.2 * R0      # initial droplet-base height [m]
w_drop0_phys = -2.0e-2       # initial vertical velocity [m/s]

x_drop0 = x_drop0_phys / length_scale
y_drop0 = y_drop0_phys / length_scale
u_drop0 = u_drop0_phys * time_scale / length_scale
v_drop0 = v_drop0_phys * time_scale / length_scale
z_drop0 = z_drop0_phys / length_scale
w_drop0 = w_drop0_phys * time_scale / length_scale

# Logarithmic spring / skidding constants
c1 = 0.7
c2 = 8.0
c3 = 0.7
c4 = 0.13

# Nondimensional wave parameters
epsilon_nd = 0.02            # reciprocal Reynolds number [-]
Bo_nd = 0.20                 # Bond number [-]
G_nd = 1.23                  # gravity coefficient [-]

# Useful nondimensional droplet coefficients.
air_drag_nd = 6.0 * np.pi * R0 * mu_air * time_scale / m
impact_drag_prefactor_nd = c4 * np.sqrt(rho * R0 / sigma) * length_scale / time_scale
pressure_force_scale_nd = m / (sigma * time_scale**2)

# Simulation domain in nondimensional units.
# L is the half-width of the square domain measured in Faraday wavelengths.
L = 8.0                     # domain half-width in terms of lambdaF[-]
N = 256                      # grid points in each direction

# Time integration in nondimensional units.
t_end_phys = 0.5             # final simulation time [s]
t_end = t_end_phys / time_scale
steps_per_faraday_period = 300
impact_dt_max_phys = 7.5e-5
impact_dt_max = impact_dt_max_phys / time_scale

# Initial wave field, specified dimensionally for readability and then converted.
eta0_phys = 0.0
phi0_phys = 0.0
eta0 = eta0_phys / length_scale
phi0 = phi0_phys * time_scale / length_scale**2

# Contact settings in nondimensional units.
contact_radius_cap = R0_nd / 3.0
min_contact_radius = 0.05 * R0_nd
contact_tolerance = 0.01 * R0_nd

# Gaussian pressure patch settings.
pressure_width_cells = 2.0

# Sponge layer settings. The Fourier box is periodic, but this damps outgoing waves near the edges.
use_sponge = True
sponge_width = 0.22 * L       # width measured inward from the box edge [-]
sponge_strength_nd = 1.5      # maximum sponge damping rate in nondimensional time [-]

# Output settings
frames_per_faraday_period = 12
save_npz = True


# ============================================================
# 2. Grid and Fourier Symbols
# ============================================================

box_size = 2.0 * L
x = np.linspace(-L, L, N, endpoint=False)
y = np.linspace(-L, L, N, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y)


def smooth_ramp(s):
    """
    Smooth ramp from 0 to 1 for s in [0,1].
    Values below 0 become 0, values above 1 become 1.
    """
    s = np.clip(s, 0.0, 1.0)
    return s**2 * (3.0 - 2.0 * s)


# Sponge damping profile.
d_edge = L - np.maximum(np.abs(X), np.abs(Y))
sponge_s = (sponge_width - d_edge) / sponge_width
gamma_sponge = sponge_strength_nd * smooth_ramp(sponge_s) if use_sponge else np.zeros_like(X)

# Fourier wavenumbers in nondimensional coordinates.
kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
KX, KY = np.meshgrid(kx, ky)
k_abs = np.sqrt(KX**2 + KY**2)
k2 = k_abs**2

# Infinite-depth DtN and Laplacian symbols.
DtN_symbol = k_abs.copy()
DtN_symbol[0, 0] = 0.0
Lap_symbol = -k2

# Homogeneous gravity-capillary frequency.
omega_wave = np.sqrt(np.maximum(k_abs * (G_nd + Bo_nd * k_abs**2), 0.0))
omega_wave[0, 0] = 1.0

k_over_omega = np.zeros_like(k_abs)
nonzero = k_abs > 0.0
k_over_omega[nonzero] = k_abs[nonzero] / omega_wave[nonzero]

linear_operator = 1j * omega_wave + 2.0 * epsilon_nd * k2
linear_operator[0, 0] = 0.0

# Index arrays for negative Fourier modes: (-kx,-ky).
neg_i = (-np.arange(N)) % N
neg_j = (-np.arange(N)) % N


# ============================================================
# 3. Helper Functions
# ============================================================

def G_eff(t):
    """
    Nondimensional effective gravity used in the droplet ODEs.

    This convention matches the wave equation form:
        G(t) = G + Gamma cos(4 pi t)
    where t is nondimensional Faraday time.
    """
    return G_nd + Gamma * np.cos(4.0 * np.pi * t)


def real_ifft(F_hat):
    """
    Real inverse FFT for fields that should be real up to roundoff.
    """
    return np.real(np.fft.ifft2(F_hat))


def fft(F):
    """
    Forward 2D FFT wrapper.
    """
    return np.fft.fft2(F)


def periodic_delta(a, b):
    """
    Shortest periodic displacement a - b on the nondimensional domain.
    """
    return (a - b + 0.5 * box_size) % box_size - 0.5 * box_size


def wrap_position(pos):
    """
    Wrap a scalar position into [-L, L).
    """
    return (pos + L) % box_size - L


def bilinear_periodic(F, xp, yp):
    """
    Bilinear interpolation of a grid field F at the periodic position (xp, yp).
    """
    sx = (wrap_position(xp) + L) / dx
    sy = (wrap_position(yp) + L) / dy

    i0 = int(np.floor(sx)) % N
    j0 = int(np.floor(sy)) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N

    tx = sx - np.floor(sx)
    ty = sy - np.floor(sy)

    return (
        (1.0 - tx) * (1.0 - ty) * F[j0, i0]
        + tx * (1.0 - ty) * F[j0, i1]
        + (1.0 - tx) * ty * F[j1, i0]
        + tx * ty * F[j1, i1]
    )


def u_from_eta_phi(eta_hat, phi_hat):
    """
    Build the integrating-factor variable:
        u_hat = eta_hat + i (k/omega) phi_hat.
    """
    u_hat = eta_hat + 1j * k_over_omega * phi_hat
    u_hat[0, 0] = 0.0
    return u_hat


def eta_phi_from_u(u_hat):
    """
    Recover eta_hat and phi_hat from u_hat using Hermitian/skew-Hermitian parts.
    """
    u_neg_conj = np.conj(u_hat[np.ix_(neg_j, neg_i)])

    eta_hat = 0.5 * (u_hat + u_neg_conj)
    phi_hat = np.zeros_like(u_hat)

    diff = u_hat - u_neg_conj
    phi_hat[nonzero] = diff[nonzero] / (2j * k_over_omega[nonzero])
    phi_hat[0, 0] = 0.0

    return eta_hat, phi_hat


def wave_fields_from_u(u_hat):
    """
    Return eta, phi, and slopes in nondimensional variables.
    """
    eta_hat, phi_hat = eta_phi_from_u(u_hat)
    eta = real_ifft(eta_hat)
    phi = real_ifft(phi_hat)
    eta_x = real_ifft(1j * KX * eta_hat)
    eta_y = real_ifft(1j * KY * eta_hat)
    return eta, phi, eta_x, eta_y


def eta_t_hat_from_u_no_pressure(u_hat):
    """
    Compute eta_t_hat from the no-current-impact wave equation.
    """
    eta_hat, phi_hat = eta_phi_from_u(u_hat)
    eta_t_hat = DtN_symbol * phi_hat + 2.0 * epsilon_nd * Lap_symbol * eta_hat

    if use_sponge:
        eta = real_ifft(eta_hat)
        eta_t_hat += fft(-gamma_sponge * eta)

    return eta_t_hat


def forcing_u_hat(u_hat, t, P_hat=None):
    """
    Non-homogeneous part of the integrating-factor wave equation.

    u_t = -(i omega + 2 epsilon k^2) u
          - i (k/omega) [Gamma cos(4 pi t) eta_hat + Bo P_hat]
    """
    if P_hat is None:
        P_hat = 0.0

    eta_hat, phi_hat = eta_phi_from_u(u_hat)

    forcing = -1j * k_over_omega * (
        Gamma * np.cos(4.0 * np.pi * t) * eta_hat + Bo_nd * P_hat
    )

    if use_sponge:
        eta = real_ifft(eta_hat)
        phi = real_ifft(phi_hat)
        eta_sponge_hat = fft(-gamma_sponge * eta)
        phi_sponge_hat = fft(-gamma_sponge * phi)
        forcing += eta_sponge_hat + 1j * k_over_omega * phi_sponge_hat

    forcing[0, 0] = 0.0
    return forcing


def pressure_patch_hat(F_acc, xp, yp, indentation):
    """
    Construct a normalized Gaussian pressure patch and return its FFT.

    F_acc = F_phys * time_scale^2 / (m length_scale).

    The pressure field P_D is scaled by sigma/length_scale and normalized so that:
        integral P_D dA_nd = F_phys / (sigma length_scale).
    """
    if F_acc <= 0.0 or indentation <= 0.0:
        return np.zeros((N, N), dtype=np.complex128)

    Rc2 = min(2.0 * R0_nd * indentation, contact_radius_cap**2)
    Rc = max(np.sqrt(max(Rc2, 0.0)), min_contact_radius)

    gaussian_width = max(Rc, pressure_width_cells * dx)

    dX = periodic_delta(X, xp)
    dY = periodic_delta(Y, yp)
    r2 = dX**2 + dY**2

    gaussian = np.exp(-0.5 * r2 / gaussian_width**2)
    gaussian_integral = np.sum(gaussian) * dx * dy

    if gaussian_integral <= 0.0:
        return np.zeros((N, N), dtype=np.complex128)

    force_integral_nd = pressure_force_scale_nd * F_acc
    P = force_integral_nd * gaussian / gaussian_integral
    return fft(P)


def surface_quantities_at_drop(u_hat, xp, yp):
    """
    Evaluate eta, grad(eta), and eta_t at the droplet position.
    """
    eta, phi, eta_x, eta_y = wave_fields_from_u(u_hat)

    eta_t_hat = eta_t_hat_from_u_no_pressure(u_hat)
    eta_t = real_ifft(eta_t_hat)

    eta_p = bilinear_periodic(eta, xp, yp)
    eta_x_p = bilinear_periodic(eta_x, xp, yp)
    eta_y_p = bilinear_periodic(eta_y, xp, yp)
    eta_t_p = bilinear_periodic(eta_t, xp, yp)

    return eta_p, eta_x_p, eta_y_p, eta_t_p


def logarithmic_vertical_acceleration(zp, wp, eta_bar_p, eta_bar_t_material, t):
    """
    Compute nondimensional z_ddot during contact from the logarithmic spring model.
    """
    s = zp - eta_bar_p
    indentation = max(-s, 1.0e-12 * R0_nd)
    s_dot = wp - eta_bar_t_material

    Q = np.log(max(c1 * R0_nd / indentation, 1.0001))
    Q = max(Q, 1.0e-3)

    A_eff = 1.0 + c3 / Q**2

    B_phys = ((4.0 / 3.0) * np.pi * nu * rho * R0 * c2) / Q
    C_phys = (2.0 * np.pi * sigma) / Q

    B_nd = B_phys * time_scale / m
    C_nd = C_phys * time_scale**2 / m

    zdd = (-G_eff(t) - B_nd * s_dot - C_nd * s) / A_eff
    return zdd


def normal_force_acceleration(zdd, t):
    """
    Return F_phys * time_scale^2 / (m length_scale), thresholded to be nonnegative.
    """
    return max(zdd + G_eff(t), 0.0)


def add_state(state, deriv, scale):
    """
    Add scale * deriv to a mixed tuple containing arrays and scalars.
    """
    return tuple(s + scale * d for s, d in zip(state, deriv))


def rk4_generic_local(state, dt, rhs_function):
    """
    Fourth-order Runge-Kutta step for a local-in-time RHS.
    """
    k1 = rhs_function(state, 0.0)
    k2 = rhs_function(add_state(state, k1, 0.5 * dt), 0.5 * dt)
    k3 = rhs_function(add_state(state, k2, 0.5 * dt), 0.5 * dt)
    k4 = rhs_function(add_state(state, k3, dt), dt)

    return tuple(
        s + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        for s, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4)
    )


def in_contact_condition(u_hat_reference, zp, wp, xp, yp, up, vp):
    """
    Check contact state relative to a reference wave field.
    """
    eta_p, eta_x_p, eta_y_p, eta_t_p = surface_quantities_at_drop(u_hat_reference, xp, yp)
    eta_t_material = eta_t_p + up * eta_x_p + vp * eta_y_p
    rel_height = zp - eta_p
    rel_velocity = wp - eta_t_material
    return rel_height, rel_velocity


# ============================================================
# 4. Output Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent[1]
RESULTS_DIR = PROJECT_DIR / "06_Results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 5. Integrating-Factor RK4 Steps
# ============================================================

def rk4_if_flight_step(state, t0, dt):
    """
    One RK4 step in the flight phase using a local integrating factor for the wave field.

    state = (u_hat, zp, wp, xp, yp, up, vp)
    """
    u_hat, zp, wp, xp, yp, up, vp = state
    q0 = u_hat.copy()

    def rhs(local_state, tau):
        q_hat, zp_l, wp_l, xp_l, yp_l, up_l, vp_l = local_state
        t = t0 + tau

        exp_pos = np.exp(linear_operator * tau)
        exp_neg = np.exp(-linear_operator * tau)
        u_stage = q_hat * exp_neg

        q_t = forcing_u_hat(u_stage, t, P_hat=None) * exp_pos

        z_t = wp_l
        w_t = -G_eff(t)

        x_t = up_l
        y_t = vp_l
        u_t = -air_drag_nd * up_l
        v_t = -air_drag_nd * vp_l

        return (q_t, z_t, w_t, x_t, y_t, u_t, v_t)

    q_next, zp, wp, xp, yp, up, vp = rk4_generic_local(
        (q0, zp, wp, xp, yp, up, vp), dt, rhs
    )

    u_next = q_next * np.exp(-linear_operator * dt)
    u_next[0, 0] = 0.0
    return (u_next, zp, wp, wrap_position(xp), wrap_position(yp), up, vp)


def rk4_if_impact_step(state, t0, dt):
    """
    One RK4 step in the impact phase using a local integrating factor for both wave fields.

    state = (u_hat, zp, wp, xp, yp, up, vp, u_bar_hat)
    """
    u_hat, zp, wp, xp, yp, up, vp, u_bar_hat = state

    q0 = u_hat.copy()
    qbar0 = u_bar_hat.copy()

    def rhs(local_state, tau):
        q_hat, zp_l, wp_l, xp_l, yp_l, up_l, vp_l, qbar_hat = local_state
        t = t0 + tau

        exp_pos = np.exp(linear_operator * tau)
        exp_neg = np.exp(-linear_operator * tau)
        u_stage = q_hat * exp_neg
        u_bar_stage = qbar_hat * exp_neg

        eta_bar_p, eta_bar_x_p, eta_bar_y_p, eta_bar_t_p = surface_quantities_at_drop(
            u_bar_stage, xp_l, yp_l
        )
        eta_bar_t_material = eta_bar_t_p + up_l * eta_bar_x_p + vp_l * eta_bar_y_p

        indentation = eta_bar_p - zp_l
        zdd = logarithmic_vertical_acceleration(zp_l, wp_l, eta_bar_p, eta_bar_t_material, t)
        F_acc = normal_force_acceleration(zdd, t)

        impact_drag_nd = impact_drag_prefactor_nd * F_acc
        total_drag_nd = impact_drag_nd + air_drag_nd

        xdd = -F_acc * eta_bar_x_p - total_drag_nd * up_l
        ydd = -F_acc * eta_bar_y_p - total_drag_nd * vp_l

        P_hat = pressure_patch_hat(F_acc, xp_l, yp_l, indentation)
        q_t = forcing_u_hat(u_stage, t, P_hat=P_hat) * exp_pos
        qbar_t = forcing_u_hat(u_bar_stage, t, P_hat=None) * exp_pos

        return (q_t, wp_l, zdd, up_l, vp_l, xdd, ydd, qbar_t)

    q_next, zp, wp, xp, yp, up, vp, qbar_next = rk4_generic_local(
        (q0, zp, wp, xp, yp, up, vp, qbar0), dt, rhs
    )

    u_next = q_next * np.exp(-linear_operator * dt)
    u_bar_next = qbar_next * np.exp(-linear_operator * dt)
    u_next[0, 0] = 0.0
    u_bar_next[0, 0] = 0.0

    return (u_next, zp, wp, wrap_position(xp), wrap_position(yp), up, vp, u_bar_next)


# ============================================================
# 6. Initial Conditions
# ============================================================

eta = eta0 * np.ones((N, N), dtype=float)
phi = phi0 * np.ones((N, N), dtype=float)

eta_hat = fft(eta)
phi_hat = fft(phi)
u_hat = u_from_eta_phi(eta_hat, phi_hat)

z_drop = z_drop0
w_drop = w_drop0
x_drop = x_drop0
y_drop = y_drop0
u_drop_x = u_drop0
u_drop_y = v_drop0

impact_active = False
u_bar_hat = None


# ============================================================
# 7. Time Step Estimate
# ============================================================

dt_forcing = 1.0 / steps_per_faraday_period
dt = min(dt_forcing, impact_dt_max)

n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

plot_every = max(1, int((1.0 / frames_per_faraday_period) / dt))

omega_max = np.max(np.sqrt(np.maximum(k_abs * (G_nd + Bo_nd * k_abs**2), 0.0)))
dt_direct_wave_estimate = 0.18 / max(omega_max, 1.0)

print(f"Milewski wave parameters: epsilon = {epsilon_nd:.5e}, Bo = {Bo_nd:.5e}, G = {G_nd:.5e}")
print(f"Milewski impact constants: c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}")
print(f"Gamma/GammaF = {Gamma / GammaF:.5f}")
print(f"Faraday length scale: lambdaF = {length_scale:.5e} m")
print(f"Faraday wavenumber: kF = {kF:.5e} 1/m")
print(f"Faraday period: time_scale = {time_scale:.5e} s")
print(f"Domain half-width: L = {L:.5e} wavelengths = {L * length_scale:.5e} m")
print(f"Grid: N = {N}, dx = {dx:.5e} wavelengths = {dx * length_scale:.5e} m")
print(f"Direct explicit wave dt estimate = {dt_direct_wave_estimate:.5e} Faraday periods")
print(f"Integrating-factor dt = {dt:.5e} Faraday periods = {dt * time_scale:.5e} s")
print(f"Number of RK4 steps: {n_steps}")
print(f"Store one saved frame every {plot_every} RK4 steps")
print(f"R0/lambdaF = {R0_nd:.5e}")
print(f"Gaussian pressure minimum width = {pressure_width_cells:.2f} dx")
print(f"Contact tolerance = {contact_tolerance:.5e} wavelengths")


# ============================================================
# 8. Run Simulation and Store Frames
# ============================================================

frames = []
times = []
drop_x_history = []
drop_y_history = []
drop_z_history = []
drop_speed_history = []
contact_history = []

for n in range(n_steps + 1):
    t = n * dt

    if n % plot_every == 0:
        eta_hat_current, _ = eta_phi_from_u(u_hat)
        frames.append(length_scale * real_ifft(eta_hat_current))
        times.append(t * time_scale)

    drop_x_history.append(length_scale * x_drop)
    drop_y_history.append(length_scale * y_drop)
    drop_z_history.append(length_scale * z_drop)
    drop_speed_history.append((length_scale / time_scale) * np.sqrt(u_drop_x**2 + u_drop_y**2))
    contact_history.append(1 if impact_active else 0)

    if n == n_steps:
        break

    if not impact_active:
        state = (u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y)
        state_next = rk4_if_flight_step(state, t, dt)
        u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y = state_next

        rel_height, rel_velocity = in_contact_condition(
            u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y
        )

        if rel_height <= contact_tolerance and rel_velocity < 0.0:
            impact_active = True
            u_bar_hat = u_hat.copy()

    else:
        state = (u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y, u_bar_hat)
        state_next = rk4_if_impact_step(state, t, dt)
        u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y, u_bar_hat = state_next

        rel_height, rel_velocity = in_contact_condition(
            u_bar_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y
        )

        if rel_height >= contact_tolerance and rel_velocity > 0.0:
            impact_active = False
            u_bar_hat = None


# Convert histories to arrays.
drop_x_history = np.asarray(drop_x_history)
drop_y_history = np.asarray(drop_y_history)
drop_z_history = np.asarray(drop_z_history)
drop_speed_history = np.asarray(drop_speed_history)
contact_history = np.asarray(contact_history)
time_history = time_scale * np.linspace(0.0, t_end, n_steps + 1)

if save_npz:
    output_path = RESULTS_DIR / "PWH_CPU_u_version_2.npz"

    np.savez(
        output_path,
        x=length_scale * x,
        y=length_scale * y,
        x_nd=x,
        y_nd=y,
        times=np.asarray(times),
        frames=np.asarray(frames),
        time_history=time_history,
        drop_x_history=drop_x_history,
        drop_y_history=drop_y_history,
        drop_z_history=drop_z_history,
        drop_speed_history=drop_speed_history,
        contact_history=contact_history,
        lambdaF=length_scale,
        kF=kF,
        dt=dt * time_scale,
        dt_nd=dt,
        epsilon_nd=epsilon_nd,
        Bo_nd=Bo_nd,
        G_nd=G_nd,
        Gamma=Gamma,
        GammaF=GammaF,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        L_nd=L,
        N=N,
        use_sponge=use_sponge,
        sponge_width=sponge_width,
        sponge_strength_nd=sponge_strength_nd,
        pressure_width_cells=pressure_width_cells,
        contact_tolerance=contact_tolerance,
    )
    print(f"Saved output to {output_path}")

print("Simulation complete.")
