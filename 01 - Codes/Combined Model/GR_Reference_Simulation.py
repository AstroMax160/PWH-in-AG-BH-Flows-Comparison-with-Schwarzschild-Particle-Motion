"""
Nondimensional Schwarzschild reference-particle simulation.

This script integrates the nondimensional GR trajectory equations derived in
Section 6.1 of the report and saves the complete trajectory to an NPZ file.


The GR equations are solved directly in nondimensional form, with

    Schwarzschild horizon radius = 1
    characteristic velocity      = 1

The initial conditions are specified only in Cartesian form to match the
PWH-AG simulation convention. 
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# USER INPUT
# =============================================================================


RUN_NAME = "GR_CircOrbit_Phase"

# ---------------------------------------------------------------------------
# Paste the PWH_TO_GR converter output directly into this block.
# The variable names are intentionally identical to the converter output.
# ---------------------------------------------------------------------------

X0_TILDE             = 2.69230769231
Y0_TILDE             = 0
VX0_TILDE            = 0
VY0_TILDE            = 0.647027880063
TAU_END_TILDE        = 5.84923076923
STOP_RADIUS_TILDE    = 0.2

# ---------------------------------------------------------------------------

OUTPUT_DT_TILDE = 1.0e-3

# Integration tolerances
RTOL = 1.0e-10
ATOL = 1.0e-12
MAX_INTERNAL_STEP = 1.0e-2

# ---------------------------------------------------------------------------
# Stopping conditions
# ---------------------------------------------------------------------------
# The proper-time equations remain regular at the horizon r_tilde = 1.
# Leave STOP_AT_HORIZON = False when horizon crossing should be simulated.
STOP_AT_HORIZON = False


# ---------------------------------------------------------------------------
# Output settings
# ---------------------------------------------------------------------------


# When True, overwrite a file with the same RUN_NAME.
# When False, append a timestamp if that file already exists.
OVERWRITE_EXISTING = True

# Explicit output directory. Leave as None to use the project path:
# 06_Results/Combined Model/Reference GR Model Runs/
CUSTOM_OUTPUT_DIRECTORY: str | None = None


# =============================================================================
# COORDINATE CONVERSIONS
# =============================================================================

def cartesian_to_polar_state(
    x: float,
    y: float,
    vx: float,
    vy: float,
) -> tuple[float, float, float, float]:
    """Convert Cartesian position and velocity to polar components."""
    r = float(np.hypot(x, y))

    if r <= 0.0:
        raise ValueError("The initial position must satisfy r_tilde > 0.")

    phi = float(np.arctan2(y, x))
    vr = float((x * vx + y * vy) / r)
    vphi = float((-y * vx + x * vy) / r)

    return r, phi, vr, vphi


def polar_to_cartesian_state(
    r: np.ndarray,
    phi: np.ndarray,
    vr: np.ndarray,
    vphi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert polar position and velocity arrays to Cartesian arrays."""
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    x = r * cos_phi
    y = r * sin_phi
    vx = vr * cos_phi - vphi * sin_phi
    vy = vr * sin_phi + vphi * cos_phi

    return x, y, vx, vy


# =============================================================================
# NONDIMENSIONAL SCHWARZSCHILD MODEL
# =============================================================================

def gr_rhs(
    tau_tilde: float,
    state: np.ndarray,
    h_tilde: float,
) -> np.ndarray:
    """
    Nondimensional Schwarzschild geodesic equations.

    State:
        state[0] = r_tilde
        state[1] = u_r_tilde = d r_tilde / d tau_tilde
        state[2] = phi

    Equations:
        dr/dtau     = u_r
        du_r/dtau   = -1/(2 r^2) + h^2/r^3 - 3 h^2/(2 r^4)
        dphi/dtau   = h/r^2
    """
    del tau_tilde

    r_tilde, ur_tilde, phi = state
    del phi

    if r_tilde <= 0.0:
        raise RuntimeError("The GR integration reached r_tilde <= 0.")

    dr_dtau = ur_tilde
    dur_dtau = (
        -1.0 / (2.0 * r_tilde**2)
        + h_tilde**2 / r_tilde**3
        - 3.0 * h_tilde**2 / (2.0 * r_tilde**4)
    )
    dphi_dtau = h_tilde / r_tilde**2

    return np.array([dr_dtau, dur_dtau, dphi_dtau], dtype=float)


def make_inward_radius_event(radius: float) -> Callable:
    """Create a terminal event triggered while crossing a radius inward."""

    def event(
        tau_tilde: float,
        state: np.ndarray,
        h_tilde: float,
    ) -> float:
        del tau_tilde, h_tilde
        return float(state[0] - radius)

    event.terminal = True
    event.direction = -1.0
    return event


def compute_energy_parameter_squared(
    r0_tilde: float,
    ur0_tilde: float,
    h_tilde: float,
) -> float:
    """
    Compute k_tilde^2 from the nondimensional radial constraint:

        u_r^2 = k^2 - (1 - 1/r)(1 + h^2/r^2).
    """
    return float(
        ur0_tilde**2
        + (1.0 - 1.0 / r0_tilde)
        * (1.0 + h_tilde**2 / r0_tilde**2)
    )


def compute_radial_constraint_residual(
    r_tilde: np.ndarray,
    ur_tilde: np.ndarray,
    h_tilde: float,
    k_squared: float,
) -> np.ndarray:
    """Return the numerical residual of the first-integral constraint."""
    return (
        ur_tilde**2
        - k_squared
        + (1.0 - 1.0 / r_tilde)
        * (1.0 + h_tilde**2 / r_tilde**2)
    )


# =============================================================================
# OUTPUT PATH
# =============================================================================

def determine_project_root() -> Path:
    """
    Infer the repository root from the intended script location.

    For:
        <root>/03_Codes/Combined Model/GR_Reference_Simulation.py

    this returns:
        <root>
    """
    script_path = Path(__file__).resolve()

    try:
        return script_path.parents[2]
    except IndexError as exc:
        raise RuntimeError(
            "Could not infer the project root from the script location."
        ) from exc


def determine_output_directory() -> Path:
    """Return and create the configured results directory."""
    if CUSTOM_OUTPUT_DIRECTORY is not None:
        output_directory = Path(CUSTOM_OUTPUT_DIRECTORY).expanduser().resolve()
    else:
        project_root = determine_project_root()
        output_directory = (
            project_root
            / "06_Results"
            / "Combined Model"
            / "Reference GR Model Runs"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def determine_output_file(output_directory: Path) -> Path:
    """Construct a safe NPZ output filename."""
    base_file = output_directory / f"{RUN_NAME}.npz"

    if OVERWRITE_EXISTING or not base_file.exists():
        return base_file

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_directory / f"{RUN_NAME}_{timestamp}.npz"


# =============================================================================
# SIMULATION
# =============================================================================

def build_initial_state() -> tuple[float, float, float, float]:
    """
    Convert the Cartesian nondimensional initial conditions to the polar
    variables required by the Schwarzschild ODE system.

    Returns
    -------
    r0_tilde, phi0, ur0_tilde, vphi0_tilde
    """
    return cartesian_to_polar_state(
        X0_TILDE,
        Y0_TILDE,
        VX0_TILDE,
        VY0_TILDE,
    )


def run_simulation() -> Path:
    """Integrate the GR trajectory and save all comparison quantities."""
    if TAU_END_TILDE <= 0.0:
        raise ValueError("TAU_END_TILDE must be positive.")

    if OUTPUT_DT_TILDE <= 0.0:
        raise ValueError("OUTPUT_DT_TILDE must be positive.")

    if STOP_RADIUS_TILDE <= 0.0:
        raise ValueError("STOP_RADIUS_TILDE must be positive.")

    r0_tilde, phi0, ur0_tilde, vphi0_tilde = build_initial_state()

    if r0_tilde <= STOP_RADIUS_TILDE:
        raise ValueError(
            "The initial radius must be larger than STOP_RADIUS_TILDE."
        )

    # Conserved nondimensional angular momentum.
    h_tilde = r0_tilde * vphi0_tilde

    # Energy-like constant, retained as a diagnostic.
    k_squared = compute_energy_parameter_squared(
        r0_tilde,
        ur0_tilde,
        h_tilde,
    )

    if k_squared < 0.0:
        raise ValueError(
            "The selected initial conditions give k_tilde^2 < 0 and do not "
            "define a real timelike Schwarzschild trajectory."
        )

    k_tilde = float(np.sqrt(k_squared))

    initial_state = np.array(
        [r0_tilde, ur0_tilde, phi0],
        dtype=float,
    )

    # Include the requested final time exactly.
    tau_eval = np.arange(
        0.0,
        TAU_END_TILDE,
        OUTPUT_DT_TILDE,
        dtype=float,
    )
    if tau_eval.size == 0 or tau_eval[-1] < TAU_END_TILDE:
        tau_eval = np.append(tau_eval, TAU_END_TILDE)

    events = [make_inward_radius_event(STOP_RADIUS_TILDE)]
    event_names = ["inner_stop_radius"]

    if STOP_AT_HORIZON and r0_tilde > 1.0:
        events.append(make_inward_radius_event(1.0))
        event_names.append("horizon")

    solution = solve_ivp(
        fun=gr_rhs,
        t_span=(0.0, TAU_END_TILDE),
        y0=initial_state,
        method="DOP853",
        t_eval=tau_eval,
        args=(h_tilde,),
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_INTERNAL_STEP,
        events=events,
        dense_output=True,
    )

    if not solution.success:
        raise RuntimeError(
            f"GR integration failed: {solution.message}"
        )

    tau_tilde = solution.t
    r_tilde = solution.y[0]
    ur_tilde = solution.y[1]
    phi = solution.y[2]

    # Tangential velocity and angular rate.
    vphi_tilde = h_tilde / r_tilde
    omega_phi_tilde = h_tilde / r_tilde**2

    x_tilde, y_tilde, vx_tilde, vy_tilde = polar_to_cartesian_state(
        r_tilde,
        phi,
        ur_tilde,
        vphi_tilde,
    )

    speed_tilde = np.hypot(vx_tilde, vy_tilde)
    radial_acceleration_tilde = (
        -1.0 / (2.0 * r_tilde**2)
        + h_tilde**2 / r_tilde**3
        - 3.0 * h_tilde**2 / (2.0 * r_tilde**4)
    )

    constraint_residual = compute_radial_constraint_residual(
        r_tilde,
        ur_tilde,
        h_tilde,
        k_squared,
    )

    inside_horizon = r_tilde < 1.0

    # Determine horizon crossing from the saved trajectory.
    crossing_indices = np.flatnonzero(
        (r_tilde[:-1] >= 1.0) & (r_tilde[1:] < 1.0)
    )

    if crossing_indices.size > 0:
        i = int(crossing_indices[0])

        # Linear interpolation gives a sufficiently precise output diagnostic.
        fraction = (
            (1.0 - r_tilde[i])
            / (r_tilde[i + 1] - r_tilde[i])
        )
        horizon_crossing_tau_tilde = float(
            tau_tilde[i]
            + fraction * (tau_tilde[i + 1] - tau_tilde[i])
        )
        horizon_crossed = True
    else:
        horizon_crossing_tau_tilde = np.nan
        horizon_crossed = False

    terminal_event = "none"
    terminal_event_tau_tilde = np.nan

    for name, event_times in zip(event_names, solution.t_events):
        if event_times.size > 0:
            terminal_event = name
            terminal_event_tau_tilde = float(event_times[0])
            break

    metadata = {
        "model": "Nondimensional Schwarzschild timelike geodesic",
        "coordinate_parameter": "proper time",
        "horizon_radius_tilde": 1.0,
        "initial_condition_mode": "CARTESIAN",
        "solver": "scipy.integrate.solve_ivp",
        "method": "DOP853",
        "rtol": RTOL,
        "atol": ATOL,
        "max_internal_step": MAX_INTERNAL_STEP,
        "output_dt_tilde": OUTPUT_DT_TILDE,
        "requested_tau_end_tilde": TAU_END_TILDE,
        "stop_at_horizon": STOP_AT_HORIZON,
        "stop_radius_tilde": STOP_RADIUS_TILDE,
        "terminal_event": terminal_event,
        "terminal_event_tau_tilde": terminal_event_tau_tilde,
        "horizon_crossed": horizon_crossed,
        "horizon_crossing_tau_tilde": horizon_crossing_tau_tilde,
    }

    output_directory = determine_output_directory()
    output_file = determine_output_file(output_directory)

    # Several aliases are deliberately saved to make later comparison code
    # tolerant of different naming conventions.
    np.savez_compressed(
        output_file,

        # Time
        t=tau_tilde,
        time=tau_tilde,
        tau=tau_tilde,
        tau_tilde=tau_tilde,

        # Cartesian trajectory
        x=x_tilde,
        y=y_tilde,
        x_tilde=x_tilde,
        y_tilde=y_tilde,

        # Polar trajectory
        r=r_tilde,
        phi=phi,
        r_tilde=r_tilde,

        # Cartesian velocity
        vx=vx_tilde,
        vy=vy_tilde,
        vx_tilde=vx_tilde,
        vy_tilde=vy_tilde,

        # Polar velocity
        vr=ur_tilde,
        ur=ur_tilde,
        vphi=vphi_tilde,
        vr_tilde=ur_tilde,
        ur_tilde=ur_tilde,
        vphi_tilde=vphi_tilde,
        phi_dot=omega_phi_tilde,

        # Speed and acceleration
        speed=speed_tilde,
        speed_tilde=speed_tilde,
        radial_acceleration=radial_acceleration_tilde,

        # Constants and initial conditions
        h=np.array(h_tilde),
        h_tilde=np.array(h_tilde),
        k=np.array(k_tilde),
        k_squared=np.array(k_squared),
        r0_tilde=np.array(r0_tilde),
        phi0=np.array(phi0),
        ur0_tilde=np.array(ur0_tilde),
        vphi0_tilde=np.array(vphi0_tilde),

        # Horizon diagnostics
        horizon_radius=np.array(1.0),
        horizon_radius_tilde=np.array(1.0),
        inside_horizon=inside_horizon,
        horizon_crossed=np.array(horizon_crossed),
        horizon_crossing_tau_tilde=np.array(
            horizon_crossing_tau_tilde
        ),

        # Numerical diagnostics
        radial_constraint_residual=constraint_residual,
        max_abs_constraint_residual=np.array(
            np.max(np.abs(constraint_residual))
        ),
        solver_success=np.array(solution.success),
        solver_message=np.array(solution.message),
        terminal_event=np.array(terminal_event),
        terminal_event_tau_tilde=np.array(terminal_event_tau_tilde),

        # Metadata
        units=np.array("nondimensional"),
        model=np.array("GR_reference_Schwarzschild"),
        metadata_json=np.array(json.dumps(metadata, indent=2)),
    )

    print("\nGR reference simulation completed.")
    print(f"Saved result: {output_file}")
    print(f"Number of saved samples: {tau_tilde.size}")
    print(f"Initial r_tilde: {r0_tilde:.12g}")
    print(f"Initial u_r_tilde: {ur0_tilde:.12g}")
    print(f"Initial v_phi_tilde: {vphi0_tilde:.12g}")
    print(f"h_tilde: {h_tilde:.12g}")
    print(f"k_tilde: {k_tilde:.12g}")
    print(
        "Maximum radial-constraint residual: "
        f"{np.max(np.abs(constraint_residual)):.3e}"
    )

    if horizon_crossed:
        print(
            "Horizon crossed at tau_tilde approximately "
            f"{horizon_crossing_tau_tilde:.12g}."
        )
    else:
        print("The saved trajectory did not cross the horizon.")

    if terminal_event != "none":
        print(
            f"Integration stopped at event '{terminal_event}' "
            f"at tau_tilde = {terminal_event_tau_tilde:.12g}."
        )

    return output_file


if __name__ == "__main__":
    run_simulation()
