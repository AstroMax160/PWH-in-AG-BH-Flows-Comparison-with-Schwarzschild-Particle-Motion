"""
PWH-AG <-> nondimensional GR input converter.
"""

from __future__ import annotations

from math import isfinite


# =============================================================================
# USER INPUT
# =============================================================================

CONVERSION_MODE = "PWH_TO_GR"
# Allowed values:
#   "PWH_TO_GR"
#   "GR_TO_PWH"

# PWH-AG comparison scales
HORIZON_RADIUS =  5e-2 # 5.0e-2    # r_H [m]
GROUP_VELOCITY =  0.2461  # 0.2461  #   # c_g(k_F) [m/s]


# -----------------------------------------------------------------------------
# Option A: dimensional PWH-AG inputs -> nondimensional GR inputs
# -----------------------------------------------------------------------------

PWH_X0 = 0.175       # [m]
PWH_Y0 = 0      # [m]
PWH_VX0 = 0          # [m/s]
PWH_VY0 = 0.123           # [m/s]
PWH_T_END = 2        # [s]
PWH_INNER_SPONGE_OUTER_RADIUS = 0.01  # [m]


# -----------------------------------------------------------------------------
# Option B: nondimensional GR inputs -> dimensional PWH-AG inputs
# -----------------------------------------------------------------------------

GR_X0_TILDE = 3.5
GR_Y0_TILDE = 0.0
GR_VX0_TILDE = 0.0
GR_VY0_TILDE = 0.5
GR_TAU_END_TILDE = 50
GR_CUTOFF_RADIUS_TILDE = 0.30


# =============================================================================
# CONVERSION
# =============================================================================

def validate_scales() -> None:
    """Check that the physical comparison scales are valid."""
    if not isfinite(HORIZON_RADIUS) or HORIZON_RADIUS <= 0.0:
        raise ValueError("HORIZON_RADIUS must be positive and finite.")

    if not isfinite(GROUP_VELOCITY) or GROUP_VELOCITY <= 0.0:
        raise ValueError("GROUP_VELOCITY must be positive and finite.")

    if (
        not isfinite(PWH_INNER_SPONGE_OUTER_RADIUS)
        or PWH_INNER_SPONGE_OUTER_RADIUS <= 0.0
    ):
        raise ValueError(
            "PWH_INNER_SPONGE_OUTER_RADIUS must be positive and finite."
        )

    if (
        not isfinite(GR_CUTOFF_RADIUS_TILDE)
        or GR_CUTOFF_RADIUS_TILDE <= 0.0
    ):
        raise ValueError(
            "GR_CUTOFF_RADIUS_TILDE must be positive and finite."
        )


def pwh_to_gr() -> dict[str, float]:
    """Convert dimensional PWH-AG inputs to GR simulation inputs."""
    time_scale = HORIZON_RADIUS / GROUP_VELOCITY

    return {
        "X0_TILDE": PWH_X0 / HORIZON_RADIUS,
        "Y0_TILDE": PWH_Y0 / HORIZON_RADIUS,
        "VX0_TILDE": PWH_VX0 / GROUP_VELOCITY,
        "VY0_TILDE": PWH_VY0 / GROUP_VELOCITY,
        "TAU_END_TILDE": PWH_T_END / time_scale,
        "STOP_RADIUS_TILDE": (
            PWH_INNER_SPONGE_OUTER_RADIUS / HORIZON_RADIUS
        ),
    }


def gr_to_pwh() -> dict[str, float]:
    """Convert nondimensional GR inputs to PWH-AG simulation inputs."""
    time_scale = HORIZON_RADIUS / GROUP_VELOCITY

    return {
        "PWH_X0": GR_X0_TILDE * HORIZON_RADIUS,
        "PWH_Y0": GR_Y0_TILDE * HORIZON_RADIUS,
        "PWH_VX0": GR_VX0_TILDE * GROUP_VELOCITY,
        "PWH_VY0": GR_VY0_TILDE * GROUP_VELOCITY,
        "PWH_T_END": GR_TAU_END_TILDE * time_scale,
        "PWH_INNER_SPONGE_OUTER_RADIUS": (
            GR_CUTOFF_RADIUS_TILDE * HORIZON_RADIUS
        ),
    }


def print_results(title: str, results: dict[str, float]) -> None:
    print("\n" + title)
    print("=" * len(title))

    for name, value in results.items():
        print(f"{name:20s} = {value:.12g}")


def main() -> None:
    validate_scales()
    mode = CONVERSION_MODE.strip().upper()

    if mode == "PWH_TO_GR":
        print_results(
            "Dimensional PWH-AG -> nondimensional GR",
            pwh_to_gr(),
        )

    elif mode == "GR_TO_PWH":
        print_results(
            "Nondimensional GR -> dimensional PWH-AG",
            gr_to_pwh(),
        )

    else:
        raise ValueError(
            "CONVERSION_MODE must be 'PWH_TO_GR' or 'GR_TO_PWH'."
        )


if __name__ == "__main__":
    main()
