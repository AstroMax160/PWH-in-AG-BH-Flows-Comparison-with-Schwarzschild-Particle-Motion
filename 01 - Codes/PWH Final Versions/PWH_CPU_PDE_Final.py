"""
CPU simulation of a single walking droplet using a Milewski-style full PDE wave model.

This version solves directly for eta_hat and phi_hat rather than using the combined
u-variable.  It is structured to match the polished u-version of the code as closely
as possible, so that the full-equations solver can later be extended with a
background flow.

The code uses nondimensional variables during the solve, a Gaussian pressure patch,
the calibrated physical parameters from the polished u-version, and the same output
structure.
"""

import numpy as np
from pathlib import Path


# ============================================================
# 1. User Input Parameters
# ============================================================

filename = "PWH_CPU_PDE_NoInitSpeed_Sponge10.npz"
# Saving location toggle.
# If True, save results to storage_dir. If False, save results in the same folder as this Python file.
save_to_storage = True
storage_dir = Path("/data/storage18/mprediger/Runs") # Path to save folder (only works while on post-1)

# Physical constants
rho = 949.0                  # liquid density [kg/m^3]
sigma = 20.6e-3              # surface tension [N/m]
nu = 20.0e-6                 # liquid kinematic viscosity [m^2/s]
mu_air = 1.84e-5             # air dynamic viscosity [kg/(m s)]
g = 9.81                     # gravitational acceleration [m/s^2]

# Bath vibration / forcing parameters
omega0 = 160.0 * np.pi       # bath angular frequency [rad/s]
Gamma = 4.9137661117          # dimensionless peak forcing, Gamma = gamma_peak / g [-]
GammaF = 5.1723853807           # Faraday threshold in the same convention [-]

# Droplet parameters
R0 = 0.3e-3                  # droplet radius [m]

# Logarithmic spring / skidding constants
c1 = 0.7
c2 = 8.0
c3 = 0.7
c4 = 0.13

# Initial droplet state
x_drop0_phys = 1.0e-2           # initial x-position [m]
y_drop0_phys = 1.0e-2           # initial y-position [m]
u_drop0_phys = 0.0e-2        # initial x-velocity [m/s]
v_drop0_phys = 0.0e-2        # initial y-velocity [m/s]
z_drop0_phys = 1.2 * R0      # initial droplet-base height [m]
w_drop0_phys = 0 #-2.0e-2       # initial vertical velocity [m/s]

# Initial wave field
eta0_phys = 0.0              # initial free-surface elevation [m]
phi0_phys = 0.0              # initial surface potential [m^2/s]

# Simulation domain
# L_phys is the half-width of the square domain in metres.
L_phys = 3.2e-2              # half-width of square domain [m]
N = 256                      # grid points in each direction

# Time integration / output settings
# The final timestep is the minimum of:
#   1) the requested bath-period timestep,
#   2) the impact-resolution cap,
#   3) the fastest resolved wave stability estimate.
t_end_phys = 1            # final simulation time [s]
steps_per_bath_period = 160  # RK4 steps per bath shaking period
impact_dt_max_phys = 1.0e-4  # maximum timestep for resolving impacts [s]
frames_per_bath_period = 6  # saved frames per bath shaking period
CFL_wave = 0.18              # conservative RK4 stability factor for direct wave solve
save_npz = True

# Contact / pressure settings
contact_radius_cap_factor = 1.0 / 3.0   # maximum contact radius: R0/3
min_contact_radius_factor = 0.05        # minimum contact radius: 0.05 R0
contact_tolerance_factor = 0.01         # contact tolerance: 0.01 R0
pressure_width_cells = 2                # minimum Gaussian pressure width in grid cells

# Sponge layer settings. The Fourier box is periodic, but this damps outgoing waves near the edges.
use_sponge = True
sponge_width_fraction = 0.22     # sponge width as a fraction of L
sponge_strength_nd = 10         # maximum sponge damping rate in nondimensional time [-]


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
kF = 1322.35443365                             # Faraday wavenumber [1/m]
lambdaF = 2.0 * np.pi / kF                     # Faraday wavelength diagnostic [m]

# Droplet derived quantities.
m = (4.0 / 3.0) * np.pi * rho * R0**3           # droplet mass [kg]
R0_nd = R0 / length_scale                       # nondimensional droplet radius [-]
m_nd = m / mass_scale                           # nondimensional droplet mass [-]

# Dimensionless numbers calculated from the chosen scales.
Omega = omega0 * np.sqrt(rho * R0**3 / sigma)   # vibration number [-]
Bo_nd = sigma * time_scale**2 / (rho * length_scale**3)
epsilon_nd = nu * time_scale / length_scale**2
G_nd = g * time_scale**2 / length_scale
Gamma_acc_nd = gamma_peak * time_scale**2 / length_scale  # = Gamma * G_nd

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
sponge_width = sponge_width_fraction * L
t_end = t_end_phys / time_scale
impact_dt_max = impact_dt_max_phys / time_scale

# Nondimensional droplet coefficients.
air_drag_nd = 6.0 * np.pi * R0 * mu_air * time_scale / m
impact_drag_prefactor_nd = c4 * np.sqrt(rho * R0 / sigma) * length_scale / time_scale
pressure_force_scale_nd = m / (sigma * time_scale**2)


# ============================================================
# 3. Grid and Fourier Symbols
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

# Homogeneous gravity-capillary frequency diagnostic for direct explicit RK4.
omega_wave = np.sqrt(np.maximum(k_abs * (G_nd + Bo_nd * k_abs**2), 0.0))
omega_wave[0, 0] = 0.0


# ============================================================
# 4. Helper Functions
# ============================================================

def G_eff(t):
    """
    Nondimensional effective acceleration in the vibrating bath frame.

    Time is nondimensionalised by the bath shaking period, so the imposed bath
    forcing has phase cos(2*pi*t).

    Gamma is the user-set ratio gamma_peak/g.  Therefore the nondimensional
    acceleration amplitude is Gamma_acc_nd = Gamma * G_nd.
    """
    return G_nd + Gamma_acc_nd * np.cos(2.0 * np.pi * t)


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


def wave_fields_from_hat(eta_hat, phi_hat):
    """
    Return eta, phi, and slopes in nondimensional variables.
    """
    eta = real_ifft(eta_hat)
    phi = real_ifft(phi_hat)
    eta_x = real_ifft(1j * KX * eta_hat)
    eta_y = real_ifft(1j * KY * eta_hat)
    return eta, phi, eta_x, eta_y


def wave_rhs_hat(eta_hat, phi_hat, t, P_hat=None):
    """
    Fourier-space RHS of the nondimensional full eta/phi wave PDE.

    eta_t = DtN(phi) + 2 epsilon Delta eta
    phi_t = -G_eff(t) eta + Bo Delta eta + 2 epsilon Delta phi - Bo P

    P is the nondimensional pressure field scaled by sigma / length_scale.
    """
    if P_hat is None:
        P_hat = 0.0

    eta_t_hat = DtN_symbol * phi_hat + 2.0 * epsilon_nd * Lap_symbol * eta_hat
    phi_t_hat = (
        -G_eff(t) * eta_hat
        + Bo_nd * Lap_symbol * eta_hat
        + 2.0 * epsilon_nd * Lap_symbol * phi_hat
        - Bo_nd * P_hat
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

# ============================================================
# Output Paths
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

    state = (eta_hat, phi_hat, zp, wp, xp, yp, up, vp)
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

    state = (eta_hat, phi_hat, zp, wp, xp, yp, up, vp, eta_bar_hat, phi_bar_hat)
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

    impact_drag_nd = impact_drag_prefactor_nd * F_acc
    total_drag_nd = impact_drag_nd + air_drag_nd

    xdd = -F_acc * eta_bar_x_p - total_drag_nd * up
    ydd = -F_acc * eta_bar_y_p - total_drag_nd * vp

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


def rk4_full_flight_step(state, t0, dt):
    """
    One direct RK4 step in the flight phase for the full eta/phi equations.
    """
    eta_hat, phi_hat, zp, wp, xp, yp, up, vp = rk4_generic(state, t0, dt, rhs_flight)
    eta_hat[0, 0] = 0.0
    phi_hat[0, 0] = 0.0
    return (eta_hat, phi_hat, zp, wp, wrap_position(xp), wrap_position(yp), up, vp)


def rk4_full_impact_step(state, t0, dt):
    """
    One direct RK4 step in the impact phase for the full eta/phi equations.
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

    eta_hat[0, 0] = 0.0
    phi_hat[0, 0] = 0.0
    eta_bar_hat[0, 0] = 0.0
    phi_bar_hat[0, 0] = 0.0

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

eta = eta0 * np.ones((N, N), dtype=float)
phi = phi0 * np.ones((N, N), dtype=float)

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
dt_wave = CFL_wave / max(np.max(omega_wave), 1.0)
dt = min(dt_forcing, impact_dt_max, dt_wave)

n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

plot_every = max(1, int((1.0 / frames_per_bath_period) / dt))

dt_direct_wave_estimate = dt_wave

print("Simulation parameters", flush=True)
print("---------------------", flush=True)
print(f"Wave parameters: epsilon = {epsilon_nd:.5e}, Bo = {Bo_nd:.5e}, G = {G_nd:.5e}, Omega = {Omega:.5e}", flush=True)
print(f"Impact constants: c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}", flush=True)
print(f"Gamma/GammaF = {Gamma / GammaF:.5f}", flush=True)
print(f"Gamma = {Gamma:.5e}, gamma_peak = {gamma_peak:.5e} m/s^2", flush=True)
print(f"Length scale: l_c = {length_scale:.5e} m", flush=True)
print(f"Time scale: T0 = {time_scale:.5e} s", flush=True)
print(f"Mass scale: M0 = {mass_scale:.5e} kg", flush=True)
print(f"Faraday wavenumber: kF = {kF:.5e} 1/m", flush=True)
print(f"Domain half-width: L = {L_phys:.5e} m = {L:.5e} nondimensional units", flush=True)
print(f"Grid: N = {N}, dx = {dx * length_scale:.5e} m = {dx:.5e} nondimensional units", flush=True)
print(f"Number of RK4 steps: {n_steps}", flush=True)
print(f"Direct full-equations dt = {dt * time_scale:.5e} s = {dt:.5e} bath periods", flush=True)
print(f"Requested forcing dt = {dt_forcing * time_scale:.5e} s = {dt_forcing:.5e} bath periods", flush=True)
print(f"Direct explicit wave dt estimate = {dt_direct_wave_estimate * time_scale:.5e} s = {dt_direct_wave_estimate:.5e} bath periods", flush=True)
print(f"Saved-frame stride: one frame every {plot_every} RK4 steps", flush=True)


# ============================================================
# 9. Run Simulation and Store Frames
# ============================================================

frames = []
times = []
drop_x_history = []
drop_y_history = []
drop_z_history = []
drop_speed_history = []
contact_history = []

# Print progress every 10% of the run.
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

    if n % plot_every == 0:
        frames.append(length_scale * real_ifft(eta_hat))
        times.append(t * time_scale)

    drop_x_history.append(length_scale * x_drop)
    drop_y_history.append(length_scale * y_drop)
    drop_z_history.append(length_scale * z_drop)
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


# Convert histories to arrays.
drop_x_history = np.asarray(drop_x_history)
drop_y_history = np.asarray(drop_y_history)
drop_z_history = np.asarray(drop_z_history)
drop_speed_history = np.asarray(drop_speed_history)
contact_history = np.asarray(contact_history)
time_history = time_scale * np.linspace(0.0, t_end, n_steps + 1)

if save_npz:
    output_path = RESULTS_DIR / filename

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
        length_scale=length_scale,
        kF=kF,
        dt=dt * time_scale,
        dt_nd=dt,
        epsilon_nd=epsilon_nd,
        Bo_nd=Bo_nd,
        G_nd=G_nd,
        Gamma=Gamma,
        GammaF=GammaF,
        gamma_peak=gamma_peak,
        Gamma_acc_nd=Gamma_acc_nd,
        Omega=Omega,
        time_scale=time_scale,
        mass_scale=mass_scale,
        R0=R0,
        R0_nd=R0_nd,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        L_nd=L,
        L_phys=L_phys,
        N=N,
        use_sponge=use_sponge,
        sponge_width=sponge_width,
        sponge_strength_nd=sponge_strength_nd,
        pressure_width_cells=pressure_width_cells,
        contact_tolerance=contact_tolerance,
        CFL_wave=CFL_wave,
    )
    print(f"Saved output to {output_path}", flush=True)

print("Simulation complete.", flush=True)
