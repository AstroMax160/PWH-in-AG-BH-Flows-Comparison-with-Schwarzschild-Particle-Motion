"""
GPU simulation of a single walking droplet using a Milewski-style full PDE wave model
with an inward radial background flow. Heavy grid/FFT calculations use CuPy.

Do not save wave frames to reduce the weight of the output, as it can be 15GB or more
for bigger domains.
"""

import numpy as np
import cupy as cp
from pathlib import Path


# ============================================================
# 1. User Input Parameters
# ============================================================

# ------------------------------------------------------------
# Output settings
# ------------------------------------------------------------

filename = "PWH_AG_GPU_EscapePhase.npz"

# Saving location toggle.
# If True, save results to storage_dir. If False, save results in the same folder as this Python file.
save_to_storage = True
storage_dir = Path("/data/storage18/mprediger/Runs")  # Path to save folder (only works while on post-1)
save_npz = True


# False saves the droplet histories and metadata without storing full 2D wave frames.
save_wave_frames = True


# ------------------------------------------------------------
# Horizon / background-flow setup
# ------------------------------------------------------------

# The horizon radius is chosen and A is later calculated acoordingly
r_H_phys = 5.0e-2              # chosen group horizon radius [m]

# Inner drain / sponge region.
# The ideal radial flow is only physically interpreted outside this region.
r_drain_fraction = 0.10              # drain radius as a fraction of r_H [-]
inner_sponge_width_fraction = 0.20   # inner sponge width as a fraction of r_H [-]

# Spectral filtering / dealiasing.
use_spectral_filter = True
filter_strength = 36.0
filter_order = 12


# ------------------------------------------------------------
# Initial droplet state
# ------------------------------------------------------------

R0 = 0.3e-3                    # droplet radius [m]

x_drop0_phys = 5.5e-2          # initial x-position [m]
y_drop0_phys = 0          # initial y-position [m]
u_drop0_phys = 0.99*0.1901          # initial x-velocity [m/s]
v_drop0_phys = 0          # initial y-velocity [m/s]
z_drop0_phys = 1.2 * R0        # initial droplet-base height [m]
w_drop0_phys = 0.0e-2         # initial vertical velocity [m/s]


# ------------------------------------------------------------
# Simulation domain and resolution
# ------------------------------------------------------------

# L_phys is the half-width of the square domain in metres.
L_phys = 10e-2               # half-width of square domain [m]
N = 800                        # grid points in each direction; ideal dx = 2.5e-4 m

# Outer sponge layer.
use_sponge = True
outer_sponge_width_fraction = 1.0e-2 / L_phys  # outer sponge width as a fraction of L (divide by L for m)
sponge_strength_nd = 1.5             # maximum sponge damping rate in nondimensional time [-]

# ------------------------------------------------------------
# Time integration / output settings
# ------------------------------------------------------------

t_end_phys = 0.8               # final simulation time [s]
steps_per_bath_period = 160    # RK4 steps per bath shaking period
impact_dt_max_phys = 1.0e-4    # maximum timestep for resolving impacts [s]
frames_per_bath_period = 4    # used only when save_wave_frames = True
CFL_wave = 0.18                # conservative RK4 stability factor for direct wave solve
CFL_adv = 0.25                 # conservative CFL factor for background-flow advection


# ------------------------------------------------------------
# Contact / pressure settings
# ------------------------------------------------------------

contact_radius_cap_factor = 1.0 / 3.0   # maximum contact radius: R0/3
min_contact_radius_factor = 0.05        # minimum contact radius: 0.05 R0
contact_tolerance_factor = 0.01         # contact tolerance: 0.01 R0
pressure_width_cells = 2                # minimum Gaussian pressure width in grid cells


# ------------------------------------------------------------
# Bath vibration / forcing parameters
# ------------------------------------------------------------

omega0 = 160.0 * np.pi         # bath angular frequency [rad/s]
Gamma = 4.9137661117           # dimensionless peak forcing, Gamma = gamma_peak / g [-]
GammaF = 5.1723853807          # Faraday threshold in the same convention [-]


# ------------------------------------------------------------
# Logarithmic spring constants
# ------------------------------------------------------------

c1 = 0.7
c2 = 8.0
c3 = 0.7
c4 = 0.13


# ------------------------------------------------------------
# Initial wave field
# ------------------------------------------------------------

eta0_phys = 0.0                # initial free-surface elevation [m]
phi0_phys = 0.0                # initial surface potential [m^2/s]


# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------

rho = 949.0                    # liquid density [kg/m^3]
sigma = 20.6e-3                # surface tension [N/m]
nu = 20.0e-6                   # liquid kinematic viscosity [m^2/s]
mu_air = 1.84e-5               # air dynamic viscosity [kg/(m s)]
g = 9.81                       # gravitational acceleration [m/s^2]


# ------------------------------------------------------------
# Setup diagnostics
# ------------------------------------------------------------

# Faraday wavenumber used for group horizon.
kF = 1322.35443365             # Faraday wavenumber [1/m]

# Critical Reynolds number used for the trapped-instability diagnostic.
Re_crit = 450.0                # critical Reynolds number [-]

# Bath depth used only for setup/stability diagnostics.
# The PWH wave model itself remains infinite-depth.
h_phys = 8.0e-3                # bath depth [m]


# ============================================================
# 2. Derived Quantities and Nondimensionalisation
# ============================================================

# Dimensional scales used for nondimensionalisation.
length_scale = np.sqrt(sigma / (rho * g))       # capillary length [m]
time_scale = 2.0 * np.pi / omega0              # bath shaking period [s]
mass_scale = rho * length_scale**3             # mass scale [kg]

# Useful dimensional diagnostics.
f0 = omega0 / (2.0 * np.pi)                    # bath frequency [Hz]
gamma_peak = Gamma * g                         # dimensional peak acceleration [m/s^2]
T_faraday = 2.0 * time_scale                    # subharmonic bath period [s]
lambdaF = 2.0 * np.pi / kF                     # Faraday wavelength [m]

# Gravity-capillary speeds at the Faraday wavenumber.
omega_kF = np.sqrt(g * kF + (sigma / rho) * kF**3)
c_p_kF = omega_kF / kF
c_g_kF = (g + 3.0 * sigma * kF**2 / rho) / (2.0 * omega_kF)

# Horizon and flow diagnostics.
A_phys = r_H_phys * c_g_kF
r_drain_phys = r_drain_fraction * r_H_phys
inner_sponge_width_phys = inner_sponge_width_fraction * r_H_phys
r_inner_sponge_outer_phys = r_drain_phys + inner_sponge_width_phys
outer_sponge_width_phys = outer_sponge_width_fraction * L_phys
outer_sponge_inner_half_width_phys = L_phys - outer_sponge_width_phys

slow_variation_tolerance = 0.10
slow_variation_ratio = lambdaF / r_H_phys

r_Re = A_phys * h_phys / (nu * Re_crit)
Re_ratio = r_Re / r_H_phys
reynolds_ratio = Re_ratio

deep_water_threshold = np.pi
deep_water_value = kF * h_phys

# Droplet derived quantities.
m = (4.0 / 3.0) * np.pi * rho * R0**3
R0_nd = R0 / length_scale
m_nd = m / mass_scale

# Dimensionless numbers calculated from the chosen scales.
Omega = omega0 * np.sqrt(rho * R0**3 / sigma)
Bo_nd = sigma * time_scale**2 / (rho * length_scale**3)
epsilon_nd = nu * time_scale / length_scale**2
G_nd = g * time_scale**2 / length_scale
Gamma_acc_nd = gamma_peak * time_scale**2 / length_scale
A_nd = A_phys * time_scale / length_scale**2

# Nondimensional initial droplet state.
x_drop0 = x_drop0_phys / length_scale
y_drop0 = y_drop0_phys / length_scale
u_drop0 = u_drop0_phys * time_scale / length_scale
v_drop0 = v_drop0_phys * time_scale / length_scale
z_drop0 = z_drop0_phys / length_scale
w_drop0 = w_drop0_phys * time_scale / length_scale

# Nondimensional initial wave field.
eta0 = eta0_phys / length_scale
phi0 = phi0_phys * time_scale / length_scale**2

# Nondimensional contact settings.
contact_radius_cap = contact_radius_cap_factor * R0_nd
min_contact_radius = min_contact_radius_factor * R0_nd
contact_tolerance = contact_tolerance_factor * R0_nd

# Nondimensional domain, sponge, and time settings.
L = L_phys / length_scale
r_H = r_H_phys / length_scale
r_drain = r_drain_phys / length_scale
inner_sponge_width = inner_sponge_width_phys / length_scale
r_inner_sponge_outer = r_inner_sponge_outer_phys / length_scale
outer_sponge_width = outer_sponge_width_phys / length_scale
outer_sponge_inner_half_width = outer_sponge_inner_half_width_phys / length_scale

t_end = t_end_phys / time_scale
impact_dt_max = impact_dt_max_phys / time_scale

# Nondimensional droplet coefficients.
air_drag_nd = 6.0 * np.pi * R0 * mu_air * time_scale / m
impact_drag_prefactor_nd = c4 * np.sqrt(rho * R0 / sigma) * length_scale / time_scale
pressure_force_scale_nd = m / (sigma * time_scale**2)


# ============================================================
# 3. Grid, Background Flow and Fourier Symbols
# ============================================================

box_size = 2.0 * L
x = cp.linspace(-L, L, N, endpoint=False)
y = cp.linspace(-L, L, N, endpoint=False)
dx = float(cp.asnumpy(x[1] - x[0]))
dy = float(cp.asnumpy(y[1] - y[0]))

X, Y = cp.meshgrid(x, y)
r = cp.sqrt(X**2 + Y**2)


def smooth_ramp(s):
    """
    Smooth ramp from 0 to 1 for s in [0,1].
    """
    s = cp.clip(s, 0.0, 1.0)
    return s**2 * (3.0 - 2.0 * s)


# Sponge damping profile.
gamma_sponge = cp.zeros_like(X)

if use_sponge:
    d_edge = L - cp.maximum(cp.abs(X), cp.abs(Y))
    outer_s = (outer_sponge_width - d_edge) / outer_sponge_width
    gamma_sponge += sponge_strength_nd * smooth_ramp(outer_s)

    inner_s = (r_inner_sponge_outer - r) / inner_sponge_width
    gamma_sponge += sponge_strength_nd * smooth_ramp(inner_s)
    gamma_sponge[r <= r_drain] = sponge_strength_nd

# Smooth radial background flow.
activation = smooth_ramp((r - r_drain) / inner_sponge_width)
r2_safe = cp.maximum(r**2, r_drain**2)
vx_B = -activation * A_nd * X / r2_safe
vy_B = -activation * A_nd * Y / r2_safe
speed_B = cp.sqrt(vx_B**2 + vy_B**2)

# Fourier wavenumbers in nondimensional coordinates.
kx = 2.0 * np.pi * cp.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * cp.fft.fftfreq(N, d=dy)
KX, KY = cp.meshgrid(kx, ky)
k_abs = cp.sqrt(KX**2 + KY**2)
k2 = k_abs**2

# Infinite-depth DtN and Laplacian symbols.
DtN_symbol = k_abs.copy()
DtN_symbol[0, 0] = 0.0
Lap_symbol = -k2

# Homogeneous gravity-capillary frequency diagnostic.
omega_wave = cp.sqrt(cp.maximum(k_abs * (G_nd + Bo_nd * k_abs**2), 0.0))
omega_wave[0, 0] = 0.0

# Smooth spectral filter.
if use_spectral_filter:
    kx_max = cp.max(cp.abs(kx))
    ky_max = cp.max(cp.abs(ky))
    k_norm = cp.sqrt((KX / kx_max)**2 + (KY / ky_max)**2)
    k_norm = cp.clip(k_norm, 0.0, 1.0)
    spectral_filter = cp.exp(-filter_strength * k_norm**filter_order)
    spectral_filter[0, 0] = 1.0
else:
    spectral_filter = cp.ones_like(k_abs)


# ============================================================
# 4. Helper Functions
# ============================================================


def G_eff(t):
    """
    Nondimensional effective acceleration in the vibrating bath frame.
    """
    return G_nd + Gamma_acc_nd * np.cos(2.0 * np.pi * t)


def real_ifft(F_hat):
    """
    Real inverse FFT for fields that should be real up to roundoff.
    """
    return cp.real(cp.fft.ifft2(F_hat))


def fft(F):
    """
    Forward 2D FFT wrapper.
    """
    return cp.fft.fft2(F)


def apply_filter(F_hat):
    """
    Apply the high-wavenumber filter.
    """
    return spectral_filter * F_hat if use_spectral_filter else F_hat


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

    value = (
        (1.0 - tx) * (1.0 - ty) * F[j0, i0]
        + tx * (1.0 - ty) * F[j0, i1]
        + (1.0 - tx) * ty * F[j1, i0]
        + tx * ty * F[j1, i1]
    )
    return float(cp.asnumpy(value))


def background_velocity_at_drop(xp, yp):
    """
    Interpolate the background velocity at the droplet position.
    """
    return bilinear_periodic(vx_B, xp, yp), bilinear_periodic(vy_B, xp, yp)


def wave_fields_from_hat(eta_hat, phi_hat):
    """
    Return eta, phi, and slopes in nondimensional variables.
    """
    eta = real_ifft(eta_hat)
    phi = real_ifft(phi_hat)
    eta_x = real_ifft(1j * KX * eta_hat)
    eta_y = real_ifft(1j * KY * eta_hat)
    return eta, phi, eta_x, eta_y


def advection_hat(q_hat):
    """
    Pseudo-spectral background-flow advection term.
    """
    q_x = real_ifft(1j * KX * q_hat)
    q_y = real_ifft(1j * KY * q_hat)
    adv = vx_B * q_x + vy_B * q_y
    return apply_filter(fft(adv))


def wave_rhs_hat(eta_hat, phi_hat, t, P_hat=None):
    """
    Fourier-space RHS of the nondimensional advected eta/phi wave PDE.
    """
    if P_hat is None:
        P_hat = 0.0

    eta_t_hat = (
        DtN_symbol * phi_hat
        + 2.0 * epsilon_nd * Lap_symbol * eta_hat
        - advection_hat(eta_hat)
    )

    phi_t_hat = (
        -G_eff(t) * eta_hat
        + Bo_nd * Lap_symbol * eta_hat
        + 2.0 * epsilon_nd * Lap_symbol * phi_hat
        - Bo_nd * P_hat
        - advection_hat(phi_hat)
    )

    if use_sponge:
        eta = real_ifft(eta_hat)
        phi = real_ifft(phi_hat)
        eta_t_hat += fft(-gamma_sponge * eta)
        phi_t_hat += fft(-gamma_sponge * phi)

    eta_t_hat[0, 0] = 0.0
    phi_t_hat[0, 0] = 0.0
    return eta_t_hat, phi_t_hat


def pressure_patch_hat(F_acc, xp, yp, indentation):
    """
    Construct a normalized Gaussian pressure patch and return its FFT.
    """
    if F_acc <= 0.0 or indentation <= 0.0:
        return cp.zeros((N, N), dtype=cp.complex128)

    Rc2 = min(2.0 * R0_nd * indentation, contact_radius_cap**2)
    Rc = max(np.sqrt(max(Rc2, 0.0)), min_contact_radius)

    gaussian_width = max(Rc, pressure_width_cells * dx)

    dX = periodic_delta(X, xp)
    dY = periodic_delta(Y, yp)
    r2_patch = dX**2 + dY**2

    gaussian = cp.exp(-0.5 * r2_patch / gaussian_width**2)
    gaussian_integral = float(cp.asnumpy(cp.sum(gaussian))) * dx * dy

    if gaussian_integral <= 0.0:
        return cp.zeros((N, N), dtype=cp.complex128)

    force_integral_nd = pressure_force_scale_nd * F_acc
    P = force_integral_nd * gaussian / gaussian_integral
    return fft(P)


def surface_quantities_at_drop(eta_hat, phi_hat, t, xp, yp):
    """
    Evaluate eta, grad(eta), and eta_t at the droplet position.
    """
    eta, phi, eta_x, eta_y = wave_fields_from_hat(eta_hat, phi_hat)

    eta_t_hat, _ = wave_rhs_hat(eta_hat, phi_hat, t, P_hat=None)
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


def rk4_generic(state, t0, dt, rhs_function):
    """
    Fourth-order Runge-Kutta step for a local-in-time RHS.
    """
    k1 = rhs_function(state, t0)
    k2 = rhs_function(add_state(state, k1, 0.5 * dt), t0 + 0.5 * dt)
    k3 = rhs_function(add_state(state, k2, 0.5 * dt), t0 + 0.5 * dt)
    k4 = rhs_function(add_state(state, k3, dt), t0 + dt)

    return tuple(
        s + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        for s, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4)
    )


def in_contact_condition(eta_hat_reference, phi_hat_reference, t, zp, wp, xp, yp, up, vp):
    """
    Check contact state relative to a reference wave field.
    """
    eta_p, eta_x_p, eta_y_p, eta_t_p = surface_quantities_at_drop(
        eta_hat_reference, phi_hat_reference, t, xp, yp
    )
    eta_t_material = eta_t_p + up * eta_x_p + vp * eta_y_p
    rel_height = zp - eta_p
    rel_velocity = wp - eta_t_material
    return rel_height, rel_velocity


# ============================================================
# 5. Output Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if save_to_storage:
    RESULTS_DIR = storage_dir
else:
    RESULTS_DIR = SCRIPT_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. Direct RK4 Steps for the Full Equations
# ============================================================


def rhs_flight(state, t):
    """
    RHS while the droplet is in flight.
    """
    eta_hat, phi_hat, zp, wp, xp, yp, up, vp = state

    eta_t_hat, phi_t_hat = wave_rhs_hat(eta_hat, phi_hat, t, P_hat=None)

    z_t = wp
    w_t = -G_eff(t)

    x_t = up
    y_t = vp
    u_t = -air_drag_nd * up
    v_t = -air_drag_nd * vp

    return (eta_t_hat, phi_t_hat, z_t, w_t, x_t, y_t, u_t, v_t)


def rhs_impact(state, t):
    """
    RHS while the droplet is in impact.
    """
    (
        eta_hat,
        phi_hat,
        zp,
        wp,
        xp,
        yp,
        up,
        vp,
        eta_bar_hat,
        phi_bar_hat,
    ) = state

    eta_bar_p, eta_bar_x_p, eta_bar_y_p, eta_bar_t_p = surface_quantities_at_drop(
        eta_bar_hat, phi_bar_hat, t, xp, yp
    )
    eta_bar_t_material = eta_bar_t_p + up * eta_bar_x_p + vp * eta_bar_y_p

    indentation = eta_bar_p - zp
    zdd = logarithmic_vertical_acceleration(zp, wp, eta_bar_p, eta_bar_t_material, t)
    F_acc = normal_force_acceleration(zdd, t)

    vBx_p, vBy_p = background_velocity_at_drop(xp, yp)
    impact_drag_nd = impact_drag_prefactor_nd * F_acc

    xdd = -F_acc * eta_bar_x_p - air_drag_nd * up - impact_drag_nd * (up - vBx_p)
    ydd = -F_acc * eta_bar_y_p - air_drag_nd * vp - impact_drag_nd * (vp - vBy_p)

    P_hat = pressure_patch_hat(F_acc, xp, yp, indentation)
    eta_t_hat, phi_t_hat = wave_rhs_hat(eta_hat, phi_hat, t, P_hat=P_hat)

    eta_bar_t_hat, phi_bar_t_hat = wave_rhs_hat(eta_bar_hat, phi_bar_hat, t, P_hat=None)

    return (
        eta_t_hat,
        phi_t_hat,
        wp,
        zdd,
        up,
        vp,
        xdd,
        ydd,
        eta_bar_t_hat,
        phi_bar_t_hat,
    )


def filter_wave_state(eta_hat, phi_hat):
    """
    Filter wave fields after an RK4 step.
    """
    eta_hat = apply_filter(eta_hat)
    phi_hat = apply_filter(phi_hat)
    eta_hat[0, 0] = 0.0
    phi_hat[0, 0] = 0.0
    return eta_hat, phi_hat


def rk4_full_flight_step(state, t0, dt):
    """
    One direct RK4 step in the flight phase.
    """
    eta_hat, phi_hat, zp, wp, xp, yp, up, vp = rk4_generic(state, t0, dt, rhs_flight)
    eta_hat, phi_hat = filter_wave_state(eta_hat, phi_hat)
    return (eta_hat, phi_hat, zp, wp, wrap_position(xp), wrap_position(yp), up, vp)


def rk4_full_impact_step(state, t0, dt):
    """
    One direct RK4 step in the impact phase.
    """
    (
        eta_hat,
        phi_hat,
        zp,
        wp,
        xp,
        yp,
        up,
        vp,
        eta_bar_hat,
        phi_bar_hat,
    ) = rk4_generic(state, t0, dt, rhs_impact)

    eta_hat, phi_hat = filter_wave_state(eta_hat, phi_hat)
    eta_bar_hat, phi_bar_hat = filter_wave_state(eta_bar_hat, phi_bar_hat)

    return (
        eta_hat,
        phi_hat,
        zp,
        wp,
        wrap_position(xp),
        wrap_position(yp),
        up,
        vp,
        eta_bar_hat,
        phi_bar_hat,
    )


# ============================================================
# 7. Initial Conditions
# ============================================================

eta = eta0 * cp.ones((N, N), dtype=float)
phi = phi0 * cp.ones((N, N), dtype=float)

eta_hat = fft(eta)
phi_hat = fft(phi)

z_drop = z_drop0
w_drop = w_drop0
x_drop = x_drop0
y_drop = y_drop0
u_drop_x = u_drop0
u_drop_y = v_drop0

impact_active = False
eta_bar_hat = None
phi_bar_hat = None


# ============================================================
# 8. Time Step Estimate
# ============================================================

dt_forcing = 1.0 / steps_per_bath_period
dt_wave = CFL_wave / max(float(cp.asnumpy(cp.max(omega_wave))), 1.0)
max_adv_speed = max(float(cp.asnumpy(cp.max(speed_B))), 1.0e-12)
dt_adv = CFL_adv * dx / max_adv_speed
dt = min(dt_forcing, impact_dt_max, dt_wave, dt_adv)

n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

if save_wave_frames:
    if frames_per_bath_period <= 0:
        raise ValueError(
            "frames_per_bath_period must be positive when save_wave_frames is True."
        )
    plot_every = max(1, int((1.0 / frames_per_bath_period) / dt))
else:
    plot_every = None

dt_phys = dt * time_scale
dx_phys = dx * length_scale


def pass_fail(condition):
    """
    Return a compact PASS/FAIL label for terminal diagnostics.
    """
    return "[PASS]" if condition else "[FAIL]"


slow_variation_ok = slow_variation_ratio <= slow_variation_tolerance
reynolds_ok = Re_ratio < 1.0
deep_water_ok = deep_water_value > deep_water_threshold
domain_side_length_phys = 2.0 * L_phys

print("GPU backend", flush=True)
print("-----------", flush=True)
print(f"CuPy version: {cp.__version__}", flush=True)
print(f"CUDA runtime: {cp.cuda.runtime.runtimeGetVersion()}", flush=True)
print(f"GPU device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}", flush=True)
print("", flush=True)

print("Simulation parameters", flush=True)
print("---------------------", flush=True)
print("", flush=True)

print("Wave / impact parameters", flush=True)
print(
    f"epsilon = {epsilon_nd:.5e}, "
    f"Bo = {Bo_nd:.5e}, "
    f"G = {G_nd:.5e}, "
    f"Omega = {Omega:.5e}",
    flush=True,
)
print(f"Impact constants: c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}", flush=True)
print(f"Gamma/GammaF = {Gamma / GammaF:.5f}", flush=True)
print(f"Gamma = {Gamma:.5e}, GammaF = {GammaF:.5e}", flush=True)
print("", flush=True)

print("Walker horizon setup", flush=True)
print(f"Faraday wavenumber: kF = {kF:.5e} 1/m", flush=True)
print(f"Chosen group horizon radius: r_H = {r_H_phys:.5e} m", flush=True)
print("", flush=True)

print("Setup requirements", flush=True)
print(
    f"Slow-variation ratio: lambdaF/r_H = {slow_variation_ratio:.5e} "
    f"< {slow_variation_tolerance:.5e}  {pass_fail(slow_variation_ok)}",
    flush=True,
)
print(f"Critical Reynolds radius: r_Re = {r_Re:.5e} m", flush=True)
print(
    f"Reynolds ratio: r_Re/r_H = {Re_ratio:.5e} "
    f"< 1.00000e+00  {pass_fail(reynolds_ok)}",
    flush=True,
)
print(f"Excluded drain radius: r_drain = {r_drain_phys:.5e} m", flush=True)
print(
    f"Deep-water diagnostic: kF*h = {deep_water_value:.5e}  "
    f"{pass_fail(deep_water_ok)}",
    flush=True,
)
print("", flush=True)

print("Numerical setup", flush=True)
print(f"Domain side length: {domain_side_length_phys:.5e} m", flush=True)
print(f"Grid: N = {N}, dx = {dx_phys:.5e} m", flush=True)
print(f"Number of RK4 steps: {n_steps}", flush=True)
print(f"Time step: dt = {dt_phys:.5e} s", flush=True)
if save_wave_frames:
    print(
        f"Wave-frame saving: enabled "
        f"(one frame every {plot_every} RK4 steps)",
        flush=True,
    )
else:
    print(
        "Wave-frame saving: disabled "
        "(droplet histories and metadata will still be saved)",
        flush=True,
    )
print("", flush=True)


# ============================================================
# 9. Run Simulation and Store Histories / Optional Frames
# ============================================================

frames = []
times = []
drop_x_history = []
drop_y_history = []
drop_z_history = []
drop_u_history = []
drop_v_history = []
drop_w_history = []
drop_speed_history = []
contact_history = []

progress_every = max(1, n_steps // 10)
last_progress_percent = -10

for n in range(n_steps + 1):
    t = n * dt

    if n % progress_every == 0 or n == n_steps:
        progress_percent = int(round(100.0 * n / n_steps))
        if progress_percent >= last_progress_percent + 10 or n == n_steps:
            print(
                f"Progress: {100.0 * n / n_steps:5.1f}% | "
                f"t = {t * time_scale:.4f} s / {t_end_phys:.4f} s",
                flush=True,
            )
            last_progress_percent = progress_percent

    if save_wave_frames and n % plot_every == 0:
        frames.append(cp.asnumpy(length_scale * real_ifft(eta_hat)))
        times.append(t * time_scale)

    drop_x_history.append(length_scale * x_drop)
    drop_y_history.append(length_scale * y_drop)
    drop_z_history.append(length_scale * z_drop)
    drop_u_history.append((length_scale / time_scale) * u_drop_x)
    drop_v_history.append((length_scale / time_scale) * u_drop_y)
    drop_w_history.append((length_scale / time_scale) * w_drop)
    drop_speed_history.append((length_scale / time_scale) * np.sqrt(u_drop_x**2 + u_drop_y**2))
    contact_history.append(1 if impact_active else 0)

    if n == n_steps:
        break

    if not impact_active:
        state = (eta_hat, phi_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y)
        state_next = rk4_full_flight_step(state, t, dt)
        eta_hat, phi_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y = state_next

        rel_height, rel_velocity = in_contact_condition(
            eta_hat, phi_hat, t + dt, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y
        )

        if rel_height <= contact_tolerance and rel_velocity < 0.0:
            impact_active = True
            eta_bar_hat = eta_hat.copy()
            phi_bar_hat = phi_hat.copy()

    else:
        state = (
            eta_hat,
            phi_hat,
            z_drop,
            w_drop,
            x_drop,
            y_drop,
            u_drop_x,
            u_drop_y,
            eta_bar_hat,
            phi_bar_hat,
        )
        state_next = rk4_full_impact_step(state, t, dt)
        (
            eta_hat,
            phi_hat,
            z_drop,
            w_drop,
            x_drop,
            y_drop,
            u_drop_x,
            u_drop_y,
            eta_bar_hat,
            phi_bar_hat,
        ) = state_next

        rel_height, rel_velocity = in_contact_condition(
            eta_bar_hat, phi_bar_hat, t + dt, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y
        )

        if rel_height >= contact_tolerance and rel_velocity > 0.0:
            impact_active = False
            eta_bar_hat = None
            phi_bar_hat = None


cp.cuda.Stream.null.synchronize()

# Convert histories to arrays.
drop_x_history = np.asarray(drop_x_history)
drop_y_history = np.asarray(drop_y_history)
drop_z_history = np.asarray(drop_z_history)
drop_u_history = np.asarray(drop_u_history)
drop_v_history = np.asarray(drop_v_history)
drop_w_history = np.asarray(drop_w_history)
drop_speed_history = np.asarray(drop_speed_history)
contact_history = np.asarray(contact_history)
time_history = time_scale * np.linspace(0.0, t_end, n_steps + 1)

# Keep empty arrays when wave-frame saving is disabled. This preserves the output
# keys without allocating or writing any full 2D wave fields.
if save_wave_frames:
    saved_times = np.asarray(times)
    saved_frames = np.asarray(frames)
else:
    saved_times = np.empty(0, dtype=float)
    saved_frames = np.empty((0, 0, 0), dtype=np.float32)


# ============================================================
# 10. Save Output
# ============================================================

if save_npz:
    output_path = RESULTS_DIR / filename

    np.savez(
        output_path,

        # Grid / wave output
        x=cp.asnumpy(length_scale * x),
        y=cp.asnumpy(length_scale * y),
        times=saved_times,
        frames=saved_frames,

        # Full droplet history
        time_history=time_history,
        drop_x_history=drop_x_history,
        drop_y_history=drop_y_history,
        drop_z_history=drop_z_history,
        drop_u_history=drop_u_history,
        drop_v_history=drop_v_history,
        drop_w_history=drop_w_history,
        drop_speed_history=drop_speed_history,
        contact_history=contact_history,

        # Horizon / background flow
        r_H=r_H_phys,
        A=A_phys,
        kF=kF,
        lambdaF=lambdaF,
        c_g_kF=c_g_kF,
        c_p_kF=c_p_kF,

        # Setup requirement diagnostics
        slow_variation_ratio=slow_variation_ratio,
        slow_variation_tolerance=slow_variation_tolerance,
        r_Re=r_Re,
        Re_ratio=Re_ratio,
        Re_crit=Re_crit,
        h_phys=h_phys,
        deep_water_value=deep_water_value,
        deep_water_threshold=deep_water_threshold,

        # Sponge / drain geometry
        r_drain=r_drain_phys,
        inner_sponge_width=inner_sponge_width_phys,
        r_inner_sponge_outer=r_inner_sponge_outer_phys,
        L_phys=L_phys,
        outer_sponge_width=outer_sponge_width_phys,
        outer_sponge_inner_half_width=outer_sponge_inner_half_width_phys,

        # Physical/model parameters
        rho=rho,
        sigma=sigma,
        nu=nu,
        mu_air=mu_air,
        g=g,
        omega0=omega0,
        Gamma=Gamma,
        GammaF=GammaF,
        R0=R0,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,

        # Numerical metadata
        N=N,
        dx=dx_phys,
        dt=dt_phys,
        n_steps=n_steps,
        save_wave_frames=save_wave_frames,
        plot_every=(-1 if plot_every is None else plot_every),
        t_end_phys=t_end_phys,
        frames_per_bath_period=frames_per_bath_period,
        steps_per_bath_period=steps_per_bath_period,
        pressure_width_cells=pressure_width_cells,
        use_sponge=use_sponge,
        use_spectral_filter=use_spectral_filter,
        CFL_wave=CFL_wave,
        CFL_adv=CFL_adv,
    )

    print(f"Saved output to {output_path}", flush=True)

print("Simulation complete.", flush=True)
