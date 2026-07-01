"""
CPU simulation of a single walking droplet using a PDE wave model using the 
u variable, which reduces computation but does not work with background flow.

"""

import numpy as np
from pathlib import Path


# ============================================================
# 1. Input Parameters
# ============================================================

# Physical parameters for silicone oil / air
rho = 949.0                 # liquid density [kg/m^3]
sigma = 20.6e-3             # surface tension [N/m]
nu = 2.0e-5                 # liquid kinematic viscosity [m^2/s]
mu_air = 1.8e-5             # air dynamic viscosity [kg/(m s)]
g = 9.8                     # gravitational acceleration [m/s^2]

# Bath vibration
f0 = 80.0                   # driving frequency [Hz]
omega0 = 2.0 * np.pi * f0   # driving angular frequency [rad/s]
Gamma = 3.8                 # dimensionless vibrational forcing

# Droplet parameters
R0 = 0.38e-3                # droplet radius [m], Fig. 3 uses Omega=0.8 at 80 Hz
m = (4.0 / 3.0) * np.pi * rho * R0**3  # droplet mass [kg]

# Initial droplet state
x_drop0 = 0.0                  # initial x-position [m]; start at the centre of the box
y_drop0 = 0.0                  # initial y-position [m]
u_drop0 = 4.0e-2               # initial horizontal x-velocity [m/s]
v_drop0 = 0.0                  # initial horizontal y-velocity [m/s]
z_drop0 = 1.2 * R0             # vertical position of droplet base [m]
w_drop0 = -2.0e-2              # vertical velocity of droplet base [m/s]

# Constants from PDE EoM
c1 = 0.7                    
c2 = 8.0                  
c3 = 0.7                    
c4 = 0.13                   

# Nondimensional wave parameters
T_scale = 1.0 / 40.0        # subharmonic Faraday period used for nondimensional time [s]
epsilon_nd = 0.02           # reciprocal Reynolds number in the wave model [-]
Bo_nd = 0.20                # Bond number in the wave model [-]
G_nd = 1.23                 # gravity coefficient in the wave model [-]

# Faraday wavenumber (from lowermost unstable wavenumber when plotting Gamma-k)
kF=12.52e2                   # Faraday wavenumber [1/m]
lambdaF = 2.0 * np.pi / kF
TF = T_scale                  # Faraday period [s]

# Simulation domain: periodic square box 
L_factor = 14
L = L_factor * lambdaF         # half-width of domain [m]
N = 256                        # grid points in each direction; increase later if stable

# Time integration
t_end = 0.5                   # final simulation time [s]
steps_per_faraday_period = 300 # temporal resolution of bath/walker dynamics
impact_dt_max = 7.5e-5         # hard cap for impact resolution [s]

# Initial wave field
eta0 = 0.0                     # start from a flat bath [m]
phi0 = 0.0                     # start from zero surface potential [m^2/s]

# Contact pressure settings
contact_radius_cap = R0 / 3.0  # maximum contact radius [m]
min_contact_radius = 0.05 * R0 # prevents division by zero in early contact [m]

# Sponge layer settings.
# The Fourier box remains mathematically periodic, but the sponge damps outgoing waves near
# the edges so that periodic wrap-around is strongly reduced.
use_sponge = True
sponge_width = 0.22 * L        # width of damping layer measured inward from the box edge [m]
sponge_strength_nd = 1.5       # maximum damping rate in nondimensional time units [-]

# Output / plotting
frames_per_faraday_period = 12 # animation frames per Faraday period
interval_ms = 30               # animation speed in milliseconds between frames
save_npz = True                # save diagnostic arrays at the end
make_animation = False         # cluster-safe: no plotting/animation in simulation script

# Animation display settings.
# This plots signed eta. Negative waves are not converted into positive waves.
plot_abs_eta = False           # True is only for magnitude-only debugging
surface_cmap = "Blues"         # blue-shade colormap; signed eta is preserved in the colorbar
color_percentile = 99.5        # ignore only the strongest outliers when setting color scale
color_scale_multiplier = 1.50  # larger value gives a less saturated, less intense colour scale
min_vmax = 1.0e-9              # minimum color scale [m]


# ============================================================
# 2. Grid and Fourier Symbols
# ============================================================

# Use endpoint=False so that the grid is truly periodic.
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
# gamma_sponge_nd = 0 in the central region and rises smoothly near the square-box edges.
# It is dimensionless because the wave equations use nondimensional time.
d_edge = L - np.maximum(np.abs(X), np.abs(Y))
sponge_s = (sponge_width - d_edge) / sponge_width
gamma_sponge_nd = sponge_strength_nd * smooth_ramp(sponge_s) if use_sponge else np.zeros_like(X)

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
KX, KY = np.meshgrid(kx, ky)
k_abs = np.sqrt(KX**2 + KY**2)
k2 = k_abs**2

# Nondimensional spectral symbols.  The physical grid is in metres, so multiply physical
# wavenumbers by lambdaF before using the nondimensional Milewski equations.
KX_nd = lambdaF * KX
KY_nd = lambdaF * KY
k_abs_nd = lambdaF * k_abs
k2_nd = k_abs_nd**2

# Infinite-depth DtN in nondimensional form: Fourier[Phi_z] = k Fourier[phi].
DtN_symbol = k_abs_nd.copy()
DtN_symbol[0, 0] = 0.0

# Spectral Laplacian symbol in nondimensional coordinates.
Lap_symbol = -k2_nd

# Nondimensional gravity-capillary frequency for the homogeneous wave part.
omega_wave = np.sqrt(np.maximum(k_abs_nd * (G_nd + Bo_nd * k_abs_nd**2), 0.0))
omega_wave[0, 0] = 1.0  # temporary nonzero value to avoid division by zero; k/omega is reset below

k_over_omega = np.zeros_like(k_abs_nd)
nonzero = k_abs_nd > 0.0
k_over_omega[nonzero] = k_abs_nd[nonzero] / omega_wave[nonzero]

# Homogeneous operator converted from nondimensional time to physical time.
linear_operator = (1j * omega_wave + 2.0 * epsilon_nd * k2_nd) / T_scale
linear_operator[0, 0] = 0.0

# Index arrays for the negative Fourier mode: (-kx,-ky)
neg_i = (-np.arange(N)) % N
neg_j = (-np.arange(N)) % N


# ============================================================
# 3. Helper Functions
# ============================================================

def g_eff(t):
    """
    Effective gravity in the vibrating bath frame for the dimensional droplet ODEs.
    """
    return g * (1.0 - Gamma * np.cos(omega0 * t))


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
    Shortest periodic displacement a - b on the domain of size box_size.
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
    Build the integrating-factor variable u_hat from eta_hat and phi_hat.

        u_hat = eta_hat + i (k/omega) phi_hat

    The zero mode is set to zero because it does not affect surface slopes or Phi_z.
    """
    u_hat = eta_hat + 1j * k_over_omega * phi_hat
    u_hat[0, 0] = 0.0
    return u_hat


def eta_phi_from_u(u_hat):
    """
    Recover eta_hat and phi_hat from u_hat using Hermitian/skew-Hermitian parts.

    Since eta and phi are real fields:
        eta_hat(-k) = conj(eta_hat(k))
        phi_hat(-k) = conj(phi_hat(k))

    Hence:
        eta_hat = 0.5 * [u(k) + conj(u(-k))]
        phi_hat = omega/(2 i k) * [u(k) - conj(u(-k))]
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
    Return physical eta, physical surface potential, and physical slopes from u_hat.

    u_hat stores nondimensional eta and phi.  The conversions are:
        eta_phys = lambdaF * eta_nd
        phi_phys = lambdaF^2 / T_scale * phi_nd
        grad_phys(eta_phys) = grad_nd(eta_nd)
    """
    eta_hat, phi_hat = eta_phi_from_u(u_hat)
    eta_nd = real_ifft(eta_hat)
    phi_nd = real_ifft(phi_hat)

    eta = lambdaF * eta_nd
    phi = (lambdaF**2 / T_scale) * phi_nd
    eta_x = real_ifft(1j * KX_nd * eta_hat)
    eta_y = real_ifft(1j * KY_nd * eta_hat)
    return eta, phi, eta_x, eta_y


def eta_t_hat_from_u_no_pressure(u_hat, t):
    """
    Compute physical eta_t_hat from the no-current-impact wave equation.

    The stored variables are nondimensional, so eta_t_nd is converted to physical
    eta_t_phys by multiplying by lambdaF / T_scale.
    """
    eta_hat, phi_hat = eta_phi_from_u(u_hat)
    eta_t_hat_nd = DtN_symbol * phi_hat + 2.0 * epsilon_nd * Lap_symbol * eta_hat

    if use_sponge:
        eta_nd = real_ifft(eta_hat)
        eta_t_hat_nd += fft(-gamma_sponge_nd * eta_nd)

    return (lambdaF / T_scale) * eta_t_hat_nd


def forcing_u_hat(u_hat, t, P_hat=None):
    """
    Non-homogeneous part of the Milewski integrating-factor wave equation.

    In nondimensional time:
        u_t = -(i omega + 2 epsilon k^2) u
              - i (k/omega) [Gamma cos(4 pi t) eta_hat + Bo P_hat]

    The function returns the RHS with respect to physical time, hence the factor 1/T_scale.
    P_hat is the FFT of the nondimensional pressure P_D = lambdaF P_phys / sigma.
    """
    if P_hat is None:
        P_hat = 0.0

    eta_hat, phi_hat = eta_phi_from_u(u_hat)
    t_nd = t / T_scale

    # Parametric bath forcing and droplet pressure forcing.
    forcing_nd = -1j * k_over_omega * (
        Gamma * np.cos(4.0 * np.pi * t_nd) * eta_hat + Bo_nd * P_hat
    )

    # Sponge damping is spatially localized, so it is applied in physical space and then
    # transformed back to Fourier space.  It is treated as part of the non-homogeneous
    # forcing in the integrating-factor method.
    if use_sponge:
        eta_nd = real_ifft(eta_hat)
        phi_nd = real_ifft(phi_hat)
        eta_sponge_hat = fft(-gamma_sponge_nd * eta_nd)
        phi_sponge_hat = fft(-gamma_sponge_nd * phi_nd)
        forcing_nd += eta_sponge_hat + 1j * k_over_omega * phi_sponge_hat

    forcing_nd[0, 0] = 0.0
    return forcing_nd / T_scale


def pressure_patch_hat(F_normal, xp, yp, indentation):
    """
    Construct the pressure patch from the normal force and return its FFT.

    P_D = F_normal / (pi R_c^2) inside the contact disk.
    """
    if F_normal <= 0.0 or indentation <= 0.0:
        return np.zeros((N, N), dtype=np.complex128)

    Rc2 = min(2.0 * R0 * indentation, contact_radius_cap**2)
    Rc = max(np.sqrt(max(Rc2, 0.0)), min_contact_radius)

    dX = periodic_delta(X, xp)
    dY = periodic_delta(Y, yp)
    mask = dX**2 + dY**2 <= Rc**2

    # Physical pressure in Pa, converted to Milewski nondimensional pressure.
    P_phys = F_normal / (np.pi * Rc**2)
    P_nd = (lambdaF / sigma) * P_phys

    P = np.zeros((N, N), dtype=float)
    P[mask] = P_nd
    return fft(P)


def surface_quantities_at_drop(u_hat, t, xp, yp):
    """
    Evaluate eta, grad(eta), and eta_t at the droplet position.

    eta_t is computed from the no-current-impact wave equation with P_D = 0.
    The material derivative used in the vertical impact equation is
        D eta / Dt = eta_t + Xdot . grad(eta)
    The Xdot contribution is added later because this function does not know the droplet velocity.
    """
    eta, phi, eta_x, eta_y = wave_fields_from_u(u_hat)

    eta_t_hat = eta_t_hat_from_u_no_pressure(u_hat, t)
    eta_t = real_ifft(eta_t_hat)

    eta_p = bilinear_periodic(eta, xp, yp)
    eta_x_p = bilinear_periodic(eta_x, xp, yp)
    eta_y_p = bilinear_periodic(eta_y, xp, yp)
    eta_t_p = bilinear_periodic(eta_t, xp, yp)

    return eta_p, eta_x_p, eta_y_p, eta_t_p


def logarithmic_vertical_acceleration(zp, wp, eta_bar_p, eta_bar_t_material, t):
    """
    Compute z_ddot during contact from the logarithmic spring model.

    The relative displacement is
        s = zp - eta_bar_p.
    During contact s < 0. The indentation is delta = eta_bar_p - zp > 0.

    Model structure:
        A m z_ddot + B s_dot + C s = -m g_eff(t)

    where
        A = 1 + c3/Q^2
        B = (4/3 pi nu rho R0 c2)/Q
        C = (2 pi sigma)/Q
        Q = ln(c1 R0 / |s|)
    """
    s = zp - eta_bar_p
    indentation = max(-s, 1.0e-12 * R0)
    s_dot = wp - eta_bar_t_material

    # Keep Q positive and finite. If indentation approaches c1*R0, the logarithmic model is outside
    # its small-deformation comfort zone. The clipping prevents a numerical singularity.
    Q = np.log(max(c1 * R0 / indentation, 1.0001))
    Q = max(Q, 1.0e-3)

    A_eff = 1.0 + c3 / Q**2
    B_eff = ((4.0 / 3.0) * np.pi * nu * rho * R0 * c2) / Q
    C_eff = (2.0 * np.pi * sigma) / Q

    zdd = (-m * g_eff(t) - B_eff * s_dot - C_eff * s) / (A_eff * m)
    return zdd


def normal_force_from_zdd(zdd, t):
    """
    Normal reaction force acting on the droplet.
    """
    return max(m * zdd + m * g_eff(t), 0.0)


def add_state(state, deriv, scale):
    """
    Add scale * deriv to a mixed tuple containing arrays and scalars.
    """
    return tuple(s + scale * d for s, d in zip(state, deriv))


def rk4_generic_local(state, dt, rhs_function):
    """
    Fourth-order Runge-Kutta step for a local-in-time RHS.

    The RHS receives the local time tau in [0,dt], not the global time.
    This is used for the integrating-factor wave update.
    """
    k1 = rhs_function(state, 0.0)
    k2 = rhs_function(add_state(state, k1, 0.5 * dt), 0.5 * dt)
    k3 = rhs_function(add_state(state, k2, 0.5 * dt), 0.5 * dt)
    k4 = rhs_function(add_state(state, k3, dt), dt)

    return tuple(
        s + (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        for s, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4)
    )


def in_contact_condition(u_hat_reference, t, zp, wp, xp, yp, up, vp):
    """
    Check contact state relative to a reference wave field.

    Impact starts when zp <= eta and relative velocity is downward.
    Impact ends when zp >= eta_bar and relative velocity is upward.
    """
    eta_p, eta_x_p, eta_y_p, eta_t_p = surface_quantities_at_drop(
        u_hat_reference, t, xp, yp
    )
    eta_t_material = eta_t_p + up * eta_x_p + vp * eta_y_p
    rel_height = zp - eta_p
    rel_velocity = wp - eta_t_material
    return rel_height, rel_velocity

# ============================================================
# Output paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent[1]
RESULTS_DIR = PROJECT_DIR / "06_Results"

RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 4. Integrating-Factor RK4 Steps
# ============================================================

def rk4_if_flight_step(state, t0, dt):
    """
    One RK4 step in the flight phase using a local integrating factor for the wave field.

    state = (u_hat, zp, wp, xp, yp, up, vp)
    """
    u_hat, zp, wp, xp, yp, up, vp = state

    # Interaction variable q = u * exp(linear_operator * tau), with tau measured from t0.
    q0 = u_hat.copy()

    def rhs(local_state, tau):
        q_hat, zp_l, wp_l, xp_l, yp_l, up_l, vp_l = local_state
        t = t0 + tau

        exp_pos = np.exp(linear_operator * tau)
        exp_neg = np.exp(-linear_operator * tau)
        u_stage = q_hat * exp_neg

        q_t = forcing_u_hat(u_stage, t, P_hat=None) * exp_pos

        z_t = wp_l
        w_t = -g_eff(t)

        x_t = up_l
        y_t = vp_l
        u_t = -(6.0 * np.pi * R0 * mu_air / m) * up_l
        v_t = -(6.0 * np.pi * R0 * mu_air / m) * vp_l

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

        # Hypothetical wave field quantities at the droplet.
        eta_bar_p, eta_bar_x_p, eta_bar_y_p, eta_bar_t_p = surface_quantities_at_drop(
            u_bar_stage, t, xp_l, yp_l
        )

        eta_bar_t_material = eta_bar_t_p + up_l * eta_bar_x_p + vp_l * eta_bar_y_p

        # Vertical impact dynamics gives the normal force.
        indentation = eta_bar_p - zp_l
        zdd = logarithmic_vertical_acceleration(zp_l, wp_l, eta_bar_p, eta_bar_t_material, t)
        F_normal = normal_force_from_zdd(zdd, t)

        # Horizontal dynamics uses the same normal force and the hypothetical slope.
        impact_drag = c4 * np.sqrt(rho * R0 / sigma) * F_normal
        air_drag = 6.0 * np.pi * R0 * mu_air
        total_drag = impact_drag + air_drag

        xdd = (-F_normal * eta_bar_x_p - total_drag * up_l) / m
        ydd = (-F_normal * eta_bar_y_p - total_drag * vp_l) / m

        # Real wave field is forced by the current impact pressure.
        P_hat = pressure_patch_hat(F_normal, xp_l, yp_l, indentation)
        q_t = forcing_u_hat(u_stage, t, P_hat=P_hat) * exp_pos

        # Hypothetical wave field evolves without the current impact pressure.
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
# 5. Initial Conditions
# ============================================================

# Stored wave variables are nondimensional.
eta_nd = (eta0 / lambdaF) * np.ones((N, N), dtype=float)
phi_nd = (phi0 * T_scale / lambdaF**2) * np.ones((N, N), dtype=float)

eta_hat = fft(eta_nd)
phi_hat = fft(phi_nd)
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
# 6. Time Step Estimate
# ============================================================

# The homogeneous wave oscillation is handled analytically, so dt is set by impact and forcing resolution.
dt_forcing = TF / steps_per_faraday_period
dt = min(dt_forcing, impact_dt_max)

n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

plot_every = max(1, int((TF / frames_per_faraday_period) / dt))

# Direct explicit wave timestep printed only for comparison.
omega_wave_print_nd = np.sqrt(np.maximum(k_abs_nd * (G_nd + Bo_nd * k_abs_nd**2), 0.0))
omega_max = np.max(omega_wave_print_nd) / T_scale
dt_direct_wave_estimate = 0.18 / max(omega_max, 1.0)

print(f"Milewski wave parameters: epsilon = {epsilon_nd:.5e}, Bo = {Bo_nd:.5e}, G = {G_nd:.5e}")
print(f"Milewski impact constants: c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}")
print(f"Wave length scale / lambdaF diagnostic: lambda = {lambdaF:.5e} m")
print(f"Faraday wavenumber diagnostic: kF = {kF:.5e} 1/m")
print(f"Faraday period / T_scale: TF = {TF:.5e} s")
print(f"Domain half-width: L = {L:.5e} m, box size = {box_size:.5e} m")
print(f"Grid: N = {N}, dx = {dx:.5e} m")
print(f"Direct explicit wave dt estimate would be {dt_direct_wave_estimate:.5e} s")
print(f"Integrating-factor dt = {dt:.5e} s")
print(f"Number of RK4 steps: {n_steps}")
print(f"Store one animation frame every {plot_every} RK4 steps")
print(f"Droplet mass: m = {m:.5e} kg")
print(f"Sponge: use_sponge = {use_sponge}, width = {sponge_width:.5e} m, max gamma_nd = {sponge_strength_nd:.3f}")


# ============================================================
# 7. Run Simulation and Store Frames
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
        frames.append(lambdaF * real_ifft(eta_hat_current))
        times.append(t)

    drop_x_history.append(x_drop)
    drop_y_history.append(y_drop)
    drop_z_history.append(z_drop)
    drop_speed_history.append(np.sqrt(u_drop_x**2 + u_drop_y**2))
    contact_history.append(1 if impact_active else 0)

    if n == n_steps:
        break

    if not impact_active:
        # Flight update.
        state = (u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y)
        state_next = rk4_if_flight_step(state, t, dt)
        u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y = state_next

        # Check if impact begins, using the real wave field.
        rel_height, rel_velocity = in_contact_condition(
            u_hat, t + dt, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y
        )

        if rel_height <= 0.0 and rel_velocity < 0.0:
            impact_active = True
            u_bar_hat = u_hat.copy()

    else:
        # Impact update. The real and hypothetical wave fields are advanced together.
        state = (u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y, u_bar_hat)
        state_next = rk4_if_impact_step(state, t, dt)
        u_hat, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y, u_bar_hat = state_next

        # Check if impact ends, using the hypothetical wave field.
        rel_height, rel_velocity = in_contact_condition(
            u_bar_hat, t + dt, z_drop, w_drop, x_drop, y_drop, u_drop_x, u_drop_y
        )

        if rel_height >= 0.0 and rel_velocity > 0.0:
            impact_active = False
            u_bar_hat = None


# Convert histories to arrays.
drop_x_history = np.asarray(drop_x_history)
drop_y_history = np.asarray(drop_y_history)
drop_z_history = np.asarray(drop_z_history)
drop_speed_history = np.asarray(drop_speed_history)
contact_history = np.asarray(contact_history)
time_history = np.linspace(0.0, t_end, n_steps + 1)

if save_npz:
    output_path = RESULTS_DIR / "PWH_CPU_u_version.npz"
    
    np.savez(
        output_path,
        x=x,
        y=y,
        times=np.asarray(times),
        frames=np.asarray(frames),
        time_history=time_history,
        drop_x_history=drop_x_history,
        drop_y_history=drop_y_history,
        drop_z_history=drop_z_history,
        drop_speed_history=drop_speed_history,
        contact_history=contact_history,
        lambdaF=lambdaF,
        kF=kF,
        dt=dt,
        epsilon_nd=epsilon_nd,
        Bo_nd=Bo_nd,
        G_nd=G_nd,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        use_sponge=use_sponge,
        sponge_width=sponge_width,
        sponge_strength_nd=sponge_strength_nd,
    )
    print("Saved output")


# ============================================================
# 8. Cluster-safe termination
# ============================================================

# Use RunFiles.py to view diagnostics and animate the saved wave field.

print("Simulation complete.")
