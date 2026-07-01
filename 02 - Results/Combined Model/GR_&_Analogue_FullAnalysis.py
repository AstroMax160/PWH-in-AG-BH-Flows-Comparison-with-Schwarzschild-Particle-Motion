"""
Combined qualitative and quantitative trajectory viewer for the GR reference
and PWH-AG analogue trajectories.

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale


# =============================================================================
# USER SETTINGS
# =============================================================================

PWH_FILENAME = "PWH_AG_GPU_Escape_Out.npz"
GR_FILENAME = "GR_Escape_Out.npz"

HTML_NAME = "Escape_Out.html"

# Optional explicit folders. Leave as None for the sibling-folder structure
# shown in the module docstring.
CUSTOM_PWH_RESULTS_DIRECTORY: str | None = None
CUSTOM_GR_RESULTS_DIRECTORY: str | None = None

# -----------------------------------------------------------------------------
# PWH-AG comparison scaling
# -----------------------------------------------------------------------------
# DISTANCE_SCALE_MODE:
#   "GROUP"  -> use the group-horizon radius r_g = A / c_g(k_F)
#   "PHASE"  -> use the phase-horizon radius r_p = A / c_p(k_F)
#   "CUSTOM" -> use CUSTOM_DISTANCE_SCALE [m]
#
# SPEED_SCALE_MODE:
#   "GROUP"  -> use c_g(k_F)
#   "PHASE"  -> use c_p(k_F)
#   "CUSTOM" -> use CUSTOM_SPEED_SCALE [m/s]
#
# The nondimensional time is always
#     t̃ = speed_scale * t / distance_scale
DISTANCE_SCALE_MODE = "GROUP"
SPEED_SCALE_MODE = "GROUP"

CUSTOM_DISTANCE_SCALE = 3.667e-2  # [m], used only for "CUSTOM"
CUSTOM_SPEED_SCALE = 1.167*0.2461     # [m/s], used only for "CUSTOM"

SHOW_SELECTED_PWH_HORIZON = True
SHOW_OTHER_PWH_HORIZON = False

# Geometry toggles
SHOW_GR_HORIZON = True
SHOW_GR_STOP_RADIUS = True
SHOW_PWH_DRAIN = True
SHOW_PWH_INNER_SPONGE = True
SHOW_PWH_OUTER_SPONGE = True

SHOW_INITIAL_POSITION = True
SHOW_FINAL_POSITION = True

# Trajectory display settings
TRAJECTORY_STRIDE = 1
MAX_COLORED_SEGMENTS = 1800
TRAJECTORY_COLORSCALE = "Turbo"
TRAJECTORY_LINE_WIDTH = 3.0

# Axis settings
# None -> one shared automatic symmetric half-width for all trajectory figures.
COMMON_AXIS_HALF_WIDTH: float | None = None
AXIS_PADDING_FACTOR = 1.10

# Figure sizes
SIDE_BY_SIDE_FIGURE_WIDTH = 700
SIDE_BY_SIDE_FIGURE_HEIGHT = 760
FULL_WIDTH_FIGURE_WIDTH = 1350
FULL_WIDTH_FIGURE_HEIGHT = 760
DIFFERENCE_FIGURE_HEIGHT = 600

# Interpolation grid for the evaluation plot
COMMON_TIME_POINT_COUNT = 2000


# =============================================================================
# PATHS AND FILE LOADING
# =============================================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def pwh_results_directory() -> Path:
    if CUSTOM_PWH_RESULTS_DIRECTORY is not None:
        return Path(CUSTOM_PWH_RESULTS_DIRECTORY).expanduser().resolve()
    return SCRIPT_DIRECTORY / "Analogue Model Runs"


def gr_results_directory() -> Path:
    if CUSTOM_GR_RESULTS_DIRECTORY is not None:
        return Path(CUSTOM_GR_RESULTS_DIRECTORY).expanduser().resolve()
    return SCRIPT_DIRECTORY / "Reference GR Model Runs"


def locate_file(directory: Path, filename: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Results directory does not exist:\n{directory}")

    matches = list(directory.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' anywhere inside:\n{directory}"
        )

    if len(matches) == 1:
        return matches[0]

    print(f"Multiple files named '{filename}' were found:")
    for index, match in enumerate(matches):
        print(f"[{index}] {match}")

    choice = int(input("Choose file number: "))
    return matches[choice]


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def scalar(npz_data: np.lib.npyio.NpzFile, key: str, default=None):
    """Read a scalar safely from an NPZ file."""
    if key not in npz_data.files:
        return default

    value = npz_data[key]

    if np.ndim(value) == 0:
        return value.item()

    if np.size(value) == 1:
        return value.reshape(-1)[0].item()

    return value


def first_available_array(
    npz_data: np.lib.npyio.NpzFile,
    *keys: str,
) -> np.ndarray:
    """Return the first available NPZ array among the supplied key names."""
    for key in keys:
        if key in npz_data.files:
            return np.asarray(npz_data[key], dtype=float)

    raise KeyError(
        "None of the requested NPZ keys were found: " + ", ".join(keys)
    )


def read_json_metadata(npz_data: np.lib.npyio.NpzFile) -> dict:
    raw = scalar(npz_data, "metadata_json", None)

    if raw is None:
        return {}

    try:
        return json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return {}


def circle_xy(radius: float, count: int = 600):
    theta = np.linspace(0.0, 2.0 * np.pi, count)
    return radius * np.cos(theta), radius * np.sin(theta)


def first_inward_crossing_time(
    time: np.ndarray,
    radius: np.ndarray,
    crossing_radius: float | None,
) -> float | None:
    """
    Return the linearly interpolated first inward crossing time.

    A crossing is counted only when the trajectory moves from
    radius > crossing_radius to radius <= crossing_radius.
    """
    if crossing_radius is None or not np.isfinite(crossing_radius):
        return None

    time = np.asarray(time, dtype=float)
    radius = np.asarray(radius, dtype=float)

    if len(time) < 2 or len(radius) < 2:
        return None

    indices = np.flatnonzero(
        (radius[:-1] > crossing_radius)
        & (radius[1:] <= crossing_radius)
    )

    if indices.size == 0:
        return None

    i = int(indices[0])
    r0 = radius[i]
    r1 = radius[i + 1]
    t0 = time[i]
    t1 = time[i + 1]

    if r1 == r0:
        return float(t1)

    fraction = (crossing_radius - r0) / (r1 - r0)
    return float(t0 + fraction * (t1 - t0))


def add_circle(
    figure: go.Figure,
    radius: float | None,
    name: str,
    *,
    dash: str = "solid",
    width: float = 2.0,
    showlegend: bool = True,
    color: str = "black",
):
    if radius is None or not np.isfinite(radius) or radius <= 0.0:
        return

    x_circle, y_circle = circle_xy(float(radius))

    figure.add_trace(
        go.Scatter(
            x=x_circle,
            y=y_circle,
            mode="lines",
            name=name,
            showlegend=showlegend,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=(
                f"{name}<br>"
                f"radius = {float(radius):.6g}"
                "<extra></extra>"
            ),
        )
    )


def add_square(
    figure: go.Figure,
    half_width: float | None,
    name: str,
    *,
    dash: str = "dot",
    width: float = 1.8,
    color: str = "black",
):
    if half_width is None or not np.isfinite(half_width) or half_width <= 0.0:
        return

    h = float(half_width)
    figure.add_trace(
        go.Scatter(
            x=[-h, h, h, -h, -h],
            y=[-h, -h, h, h, -h],
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=(
                f"{name}<br>"
                f"half-width = {h:.6g}"
                "<extra></extra>"
            ),
        )
    )


def display_indices(point_count: int) -> np.ndarray:
    stride = max(1, int(TRAJECTORY_STRIDE))
    indices = np.arange(0, point_count, stride)

    if indices.size == 0 or indices[-1] != point_count - 1:
        indices = np.append(indices, point_count - 1)

    if len(indices) > MAX_COLORED_SEGMENTS + 1:
        reduction = int(np.ceil((len(indices) - 1) / MAX_COLORED_SEGMENTS))
        indices = indices[::reduction]

        if indices[-1] != point_count - 1:
            indices = np.append(indices, point_count - 1)

    return indices


def add_colored_trajectory(
    figure: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    radius: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    speed: np.ndarray,
    *,
    time_label: str,
    trajectory_name: str,
):
    """Add a trajectory coloured by speed, with hover information."""
    indices = display_indices(len(x))

    x_plot = x[indices]
    y_plot = y[indices]
    time_plot = time[indices]
    radius_plot = radius[indices]
    vx_plot = vx[indices]
    vy_plot = vy[indices]
    speed_plot = speed[indices]

    speed_min = float(np.nanmin(speed_plot))
    speed_max = float(np.nanmax(speed_plot))
    speed_range = max(speed_max - speed_min, 1.0e-15)

    for index in range(len(indices) - 1):
        midpoint_speed = 0.5 * (speed_plot[index] + speed_plot[index + 1])
        normalized_speed = (midpoint_speed - speed_min) / speed_range
        color = sample_colorscale(TRAJECTORY_COLORSCALE, normalized_speed)[0]

        figure.add_trace(
            go.Scattergl(
                x=[x_plot[index], x_plot[index + 1]],
                y=[y_plot[index], y_plot[index + 1]],
                mode="lines",
                line=dict(color=color, width=TRAJECTORY_LINE_WIDTH),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    custom_data = np.column_stack([time_plot, radius_plot, vx_plot, vy_plot, speed_plot])

    figure.add_trace(
        go.Scattergl(
            x=x_plot,
            y=y_plot,
            mode="markers",
            name=trajectory_name,
            showlegend=False,
            marker=dict(
                size=5,
                color=speed_plot,
                colorscale=TRAJECTORY_COLORSCALE,
                opacity=0.18,
                colorbar=dict(title=dict(text="nondimensional speed")),
            ),
            customdata=custom_data,
            hovertemplate=(
                "x̃ = %{x:.6g}<br>"
                "ỹ = %{y:.6g}<br>"
                + time_label
                + " = %{customdata[0]:.6g}<br>"
                "r̃ = %{customdata[1]:.6g}<br>"
                "ṽx = %{customdata[2]:.6g}<br>"
                "ṽy = %{customdata[3]:.6g}<br>"
                "|ṽ| = %{customdata[4]:.6g}"
                "<extra></extra>"
            ),
        )
    )

    if SHOW_INITIAL_POSITION:
        figure.add_trace(
            go.Scatter(
                x=[x[0]],
                y=[y[0]],
                mode="markers",
                name="initial position",
                marker=dict(size=10, color="black", symbol="circle"),
                hovertemplate=(
                    "initial position<br>"
                    "x̃₀ = %{x:.6g}<br>"
                    "ỹ₀ = %{y:.6g}"
                    "<extra></extra>"
                ),
            )
        )

    if SHOW_FINAL_POSITION:
        figure.add_trace(
            go.Scatter(
                x=[x[-1]],
                y=[y[-1]],
                mode="markers",
                name="final position",
                marker=dict(size=11, color="black", symbol="x"),
                hovertemplate=(
                    "final position<br>"
                    "x̃ = %{x:.6g}<br>"
                    "ỹ = %{y:.6g}"
                    "<extra></extra>"
                ),
            )
        )


def apply_trajectory_layout(
    figure: go.Figure,
    title: str,
    subtitle: str,
    axis_half_width: float,
    *,
    width: int,
    height: int,
):
    figure.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=18),
        ),
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(l=75, r=90, t=165, b=125),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.12,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=0,
            font=dict(size=11),
        ),
        hovermode="closest",
    )

    figure.update_xaxes(
        title_text="x̃",
        range=[-axis_half_width, axis_half_width],
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="inside",
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.25)",
        constrain="domain",
    )

    figure.update_yaxes(
        title_text="ỹ",
        range=[-axis_half_width, axis_half_width],
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="inside",
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.25)",
        scaleanchor="x",
        scaleratio=1,
    )


def apply_standard_layout(
    figure: go.Figure,
    title: str,
    subtitle: str,
    *,
    width: int,
    height: int,
    xaxis_title: str,
    yaxis_title: str,
):
    figure.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=18),
        ),
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(l=80, r=50, t=145, b=80),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.18,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )

    figure.update_xaxes(
        title_text=xaxis_title,
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="inside",
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.25)",
    )

    figure.update_yaxes(
        title_text=yaxis_title,
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="inside",
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.25)",
    )


# =============================================================================
# PWH SCALING CHOICE
# =============================================================================

def choose_pwh_scaling(
    data: np.lib.npyio.NpzFile,
) -> tuple[float, float, str, str, float, float]:
    """
    Select the PWH-AG distance and speed scales.

    Returns
    -------
    distance_scale, speed_scale, distance_label, speed_label,
    group_horizon_radius, phase_horizon_radius
    """
    group_speed = float(scalar(data, "c_g_kF"))
    phase_speed = float(scalar(data, "c_p_kF"))
    sink_strength = float(scalar(data, "A"))
    saved_group_horizon = float(scalar(data, "r_H"))

    if group_speed <= 0.0 or phase_speed <= 0.0:
        raise ValueError("The saved phase and group speeds must be positive.")

    if sink_strength <= 0.0:
        raise ValueError("The saved sink strength A must be positive.")

    group_horizon = saved_group_horizon
    phase_horizon = sink_strength / phase_speed

    distance_mode = DISTANCE_SCALE_MODE.strip().upper()
    speed_mode = SPEED_SCALE_MODE.strip().upper()

    if distance_mode == "GROUP":
        distance_scale = group_horizon
        distance_label = "group-horizon radius"
    elif distance_mode == "PHASE":
        distance_scale = phase_horizon
        distance_label = "phase-horizon radius"
    elif distance_mode == "CUSTOM":
        distance_scale = float(CUSTOM_DISTANCE_SCALE)
        distance_label = "custom distance scale"
    else:
        raise ValueError("DISTANCE_SCALE_MODE must be 'GROUP', 'PHASE', or 'CUSTOM'.")

    if speed_mode == "GROUP":
        speed_scale = group_speed
        speed_label = "group speed"
    elif speed_mode == "PHASE":
        speed_scale = phase_speed
        speed_label = "phase speed"
    elif speed_mode == "CUSTOM":
        speed_scale = float(CUSTOM_SPEED_SCALE)
        speed_label = "custom speed scale"
    else:
        raise ValueError("SPEED_SCALE_MODE must be 'GROUP', 'PHASE', or 'CUSTOM'.")

    if distance_scale <= 0.0 or not np.isfinite(distance_scale):
        raise ValueError("The selected distance scale must be positive and finite.")

    if speed_scale <= 0.0 or not np.isfinite(speed_scale):
        raise ValueError("The selected speed scale must be positive and finite.")

    return (
        distance_scale,
        speed_scale,
        distance_label,
        speed_label,
        group_horizon,
        phase_horizon,
    )


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_gr_data(gr_file: Path) -> dict:
    data = np.load(gr_file, allow_pickle=False)
    metadata = read_json_metadata(data)

    tau = first_available_array(data, "tau_tilde", "tau", "time", "t")
    x = first_available_array(data, "x_tilde", "x")
    y = first_available_array(data, "y_tilde", "y")
    radius = first_available_array(data, "r_tilde", "r")
    vx = first_available_array(data, "vx_tilde", "vx")
    vy = first_available_array(data, "vy_tilde", "vy")
    speed = first_available_array(data, "speed_tilde", "speed")

    ur = first_available_array(data, "ur_tilde", "vr_tilde", "ur", "vr")
    vphi = first_available_array(data, "vphi_tilde", "vphi")

    horizon_radius = float(
        scalar(data, "horizon_radius_tilde", scalar(data, "horizon_radius", 1.0))
    )

    stop_radius = metadata.get(
        "stop_radius_tilde",
        scalar(data, "stop_radius_tilde", None),
    )
    if stop_radius is not None:
        stop_radius = float(stop_radius)

    horizon_crossing_time = first_inward_crossing_time(
        tau,
        radius,
        horizon_radius,
    )

    h_tilde = scalar(data, "h_tilde", scalar(data, "h", None))
    k_tilde = scalar(data, "k", None)

    auto_half_width = AXIS_PADDING_FACTOR * max(
        float(np.nanmax(np.abs(x))),
        float(np.nanmax(np.abs(y))),
        horizon_radius,
        stop_radius if stop_radius is not None else 0.0,
    )

    subtitle = (
        f"file: {gr_file.name}<br>"
        f"τ̃ range: {tau[0]:.4g}–{tau[-1]:.4g} | "
        f"(x̃₀,ỹ₀)=({x[0]:.4g},{y[0]:.4g})<br>"
        f"|ṽ₀|={speed[0]:.4g}"
        + (f" | h̃={float(h_tilde):.4g}" if h_tilde is not None else "")
        + (f" | k̃={float(k_tilde):.4g}" if k_tilde is not None else "")
    )

    return dict(
        file=gr_file,
        time=tau,
        x=x,
        y=y,
        r=radius,
        vx=vx,
        vy=vy,
        ur=ur,
        vphi=vphi,
        speed=speed,
        horizon_radius=horizon_radius,
        stop_radius=stop_radius,
        horizon_crossing_time=horizon_crossing_time,
        horizon_crossing_label="horizon crossing",
        subtitle=subtitle,
        auto_half_width=auto_half_width,
    )


def prepare_pwh_data(pwh_file: Path) -> dict:
    data = np.load(pwh_file, allow_pickle=False)

    time_dimensional = first_available_array(data, "time_history")
    x_dimensional = first_available_array(data, "drop_x_history")
    y_dimensional = first_available_array(data, "drop_y_history")

    if "drop_u_history" in data.files and "drop_v_history" in data.files:
        vx_dimensional = first_available_array(data, "drop_u_history")
        vy_dimensional = first_available_array(data, "drop_v_history")
    else:
        vx_dimensional = np.gradient(x_dimensional, time_dimensional)
        vy_dimensional = np.gradient(y_dimensional, time_dimensional)

    (
        distance_scale,
        speed_scale,
        distance_label,
        speed_label,
        group_horizon_dimensional,
        phase_horizon_dimensional,
    ) = choose_pwh_scaling(data)

    x = x_dimensional / distance_scale
    y = y_dimensional / distance_scale
    vx = vx_dimensional / speed_scale
    vy = vy_dimensional / speed_scale
    time = speed_scale * time_dimensional / distance_scale

    radius = np.hypot(x, y)
    speed = np.hypot(vx, vy)

    # Polar velocity components in nondimensional form.
    ur = (x * vx + y * vy) / radius
    vphi = (-y * vx + x * vy) / radius

    group_horizon_radius = group_horizon_dimensional / distance_scale
    phase_horizon_radius = phase_horizon_dimensional / distance_scale

    distance_mode = DISTANCE_SCALE_MODE.strip().upper()
    if distance_mode == "GROUP":
        selected_horizon_radius = group_horizon_radius
        selected_horizon_name = "selected group horizon"
        other_horizon_radius = phase_horizon_radius
        other_horizon_name = "phase horizon"
    elif distance_mode == "PHASE":
        selected_horizon_radius = phase_horizon_radius
        selected_horizon_name = "selected phase horizon"
        other_horizon_radius = group_horizon_radius
        other_horizon_name = "group horizon"
    else:
        selected_horizon_radius = None
        selected_horizon_name = "selected horizon"
        other_horizon_radius = None
        other_horizon_name = "other horizon"

    horizon_crossing_time = first_inward_crossing_time(
        time,
        radius,
        selected_horizon_radius,
    )

    r_drain = scalar(data, "r_drain", None)
    if r_drain is not None:
        r_drain = float(r_drain) / distance_scale

    r_inner_sponge_outer = scalar(data, "r_inner_sponge_outer", None)
    if r_inner_sponge_outer is not None:
        r_inner_sponge_outer = float(r_inner_sponge_outer) / distance_scale

    outer_sponge_inner_half_width = scalar(data, "outer_sponge_inner_half_width", None)
    if outer_sponge_inner_half_width is not None:
        outer_sponge_inner_half_width = float(outer_sponge_inner_half_width) / distance_scale

    auto_half_width = AXIS_PADDING_FACTOR * max(
        float(np.nanmax(np.abs(x))),
        float(np.nanmax(np.abs(y))),
        selected_horizon_radius or 0.0,
        other_horizon_radius if SHOW_OTHER_PWH_HORIZON and other_horizon_radius is not None else 0.0,
        r_drain or 0.0,
        r_inner_sponge_outer or 0.0,
        outer_sponge_inner_half_width or 0.0,
    )

    subtitle = (
        f"file: {pwh_file.name}<br>"
        f"distance scale: {distance_label} = {distance_scale:.4g} m | "
        f"speed scale: {speed_label} = {speed_scale:.4g} m/s<br>"
        f"t̃ range: {time[0]:.4g}–{time[-1]:.4g} | "
        f"(x̃₀,ỹ₀)=({x[0]:.4g},{y[0]:.4g}) | "
        f"|ṽ₀|={speed[0]:.4g}"
    )

    return dict(
        file=pwh_file,
        time=time,
        x=x,
        y=y,
        r=radius,
        vx=vx,
        vy=vy,
        ur=ur,
        vphi=vphi,
        speed=speed,
        selected_horizon_radius=selected_horizon_radius,
        selected_horizon_name=selected_horizon_name,
        horizon_crossing_time=horizon_crossing_time,
        horizon_crossing_label="horizon crossing",
        other_horizon_radius=other_horizon_radius,
        other_horizon_name=other_horizon_name,
        r_drain=r_drain,
        r_inner_sponge_outer=r_inner_sponge_outer,
        outer_sponge_inner_half_width=outer_sponge_inner_half_width,
        subtitle=subtitle,
        auto_half_width=auto_half_width,
    )


# =============================================================================
# FIGURES
# =============================================================================

def create_gr_trajectory_figure(gr: dict, axis_half_width: float) -> go.Figure:
    fig = go.Figure()

    add_colored_trajectory(
        fig,
        gr["x"],
        gr["y"],
        gr["time"],
        gr["r"],
        gr["vx"],
        gr["vy"],
        gr["speed"],
        time_label="τ̃",
        trajectory_name="GR trajectory",
    )

    if SHOW_GR_HORIZON:
        add_circle(fig, gr["horizon_radius"], "Schwarzschild horizon", dash="solid", width=2.2)

    if SHOW_GR_STOP_RADIUS:
        add_circle(fig, gr["stop_radius"], "GR stopping radius", dash="dash", width=1.8)

    apply_trajectory_layout(
        fig,
        "Nondimensional Schwarzschild reference trajectory",
        gr["subtitle"],
        axis_half_width,
        width=SIDE_BY_SIDE_FIGURE_WIDTH,
        height=SIDE_BY_SIDE_FIGURE_HEIGHT,
    )
    return fig


def create_pwh_trajectory_figure(pwh: dict, axis_half_width: float) -> go.Figure:
    fig = go.Figure()

    add_colored_trajectory(
        fig,
        pwh["x"],
        pwh["y"],
        pwh["time"],
        pwh["r"],
        pwh["vx"],
        pwh["vy"],
        pwh["speed"],
        time_label="t̃",
        trajectory_name="PWH-AG trajectory",
    )

    if SHOW_SELECTED_PWH_HORIZON and pwh["selected_horizon_radius"] is not None:
        add_circle(
            fig,
            pwh["selected_horizon_radius"],
            pwh["selected_horizon_name"],
            dash="solid",
            width=2.2,
        )

    if SHOW_OTHER_PWH_HORIZON and pwh["other_horizon_radius"] is not None:
        add_circle(
            fig,
            pwh["other_horizon_radius"],
            pwh["other_horizon_name"],
            dash="dash",
            width=1.8,
        )

    if SHOW_PWH_DRAIN:
        add_circle(fig, pwh["r_drain"], "drain radius", dash="dashdot", width=1.8)

    if SHOW_PWH_INNER_SPONGE:
        add_circle(fig, pwh["r_inner_sponge_outer"], "inner sponge boundary", dash="dot", width=1.8)

    if SHOW_PWH_OUTER_SPONGE:
        add_square(fig, pwh["outer_sponge_inner_half_width"], "outer sponge boundary", dash="dot", width=1.8)

    apply_trajectory_layout(
        fig,
        "Nondimensional PWH-AG walker trajectory",
        pwh["subtitle"],
        axis_half_width,
        width=SIDE_BY_SIDE_FIGURE_WIDTH,
        height=SIDE_BY_SIDE_FIGURE_HEIGHT,
    )
    return fig


def create_velocity_figure(system: dict, *, system_name: str, time_label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=system["time"], y=system["vx"], mode="lines",
        name="ṽx", line=dict(width=2.2)
    ))
    fig.add_trace(go.Scatter(
        x=system["time"], y=system["vy"], mode="lines",
        name="ṽy", line=dict(width=2.2)
    ))
    fig.add_trace(go.Scatter(
        x=system["time"], y=system["ur"], mode="lines",
        name="ũr", line=dict(width=2.2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=system["time"], y=system["vphi"], mode="lines",
        name="ṽφ", line=dict(width=2.2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=system["time"], y=system["speed"], mode="lines",
        name="|ṽ|", line=dict(width=2.0, dash="dot")
    ))

    subtitle = (
        f"time range: {system['time'][0]:.4g}–{system['time'][-1]:.4g} | "
        f"(x̃₀,ỹ₀)=({system['x'][0]:.4g},{system['y'][0]:.4g}) | "
        f"|ṽ₀|={system['speed'][0]:.4g}"
    )

    apply_standard_layout(
        fig,
        f"Nondimensional velocity components: {system_name}",
        subtitle,
        width=SIDE_BY_SIDE_FIGURE_WIDTH,
        height=SIDE_BY_SIDE_FIGURE_HEIGHT,
        xaxis_title=time_label,
        yaxis_title="nondimensional velocity",
    )

    crossing_time = system.get("horizon_crossing_time")
    if crossing_time is not None and np.isfinite(crossing_time):
        fig.add_vline(
            x=float(crossing_time),
            line_width=1.5,
            line_dash="dash",
            line_color="black",
        )
        fig.add_annotation(
            x=float(crossing_time),
            y=1.0,
            xref="x",
            yref="paper",
            text="r_H",
            showarrow=False,
            textangle=-90,
            xanchor="right",
            yanchor="top",
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.75)",
        )

    return fig


def create_overlay_trajectory_figure(
    gr: dict,
    pwh: dict,
    axis_half_width: float,
) -> go.Figure:
    fig = go.Figure()

    # GR trajectory
    fig.add_trace(go.Scatter(
        x=gr["x"], y=gr["y"], mode="lines",
        name="GR trajectory", line=dict(width=3.0)
    ))
    if SHOW_INITIAL_POSITION:
        fig.add_trace(go.Scatter(
            x=[gr["x"][0]], y=[gr["y"][0]], mode="markers",
            name="GR initial position",
            marker=dict(size=10, symbol="circle")
        ))
    if SHOW_FINAL_POSITION:
        fig.add_trace(go.Scatter(
            x=[gr["x"][-1]], y=[gr["y"][-1]], mode="markers",
            name="GR final position",
            marker=dict(size=11, symbol="x")
        ))

    # PWH trajectory
    fig.add_trace(go.Scatter(
        x=pwh["x"], y=pwh["y"], mode="lines",
        name="PWH-AG trajectory", line=dict(width=3.0)
    ))
    if SHOW_INITIAL_POSITION:
        fig.add_trace(go.Scatter(
            x=[pwh["x"][0]], y=[pwh["y"][0]], mode="markers",
            name="PWH initial position",
            marker=dict(size=10, symbol="circle")
        ))
    if SHOW_FINAL_POSITION:
        fig.add_trace(go.Scatter(
            x=[pwh["x"][-1]], y=[pwh["y"][-1]], mode="markers",
            name="PWH final position",
            marker=dict(size=11, symbol="x")
        ))

    # Common horizon and optional inner geometry
    add_circle(fig, 1.0, "common horizon (r̃ = 1)", dash="solid", width=2.4, color="black")

    if SHOW_GR_STOP_RADIUS:
        add_circle(fig, gr["stop_radius"], "GR stopping radius", dash="dash", width=1.8, color="black")

    if SHOW_PWH_DRAIN:
        add_circle(fig, pwh["r_drain"], "PWH drain radius", dash="dashdot", width=1.8, color="gray")

    subtitle = (
        "GR and PWH-AG trajectories overlaid in the same nondimensional plane<br>"
        f"shared axis half-width = {axis_half_width:.4g}"
    )

    apply_trajectory_layout(
        fig,
        "Overlaid nondimensional trajectories",
        subtitle,
        axis_half_width,
        width=FULL_WIDTH_FIGURE_WIDTH,
        height=FULL_WIDTH_FIGURE_HEIGHT,
    )
    return fig


def compute_difference_curve(gr: dict, pwh: dict) -> dict:
    """
    Evaluate the nondimensional trajectory separation on the overlapping time
    interval using linear interpolation.
    """
    t_start = max(float(gr["time"][0]), float(pwh["time"][0]))
    t_end = min(float(gr["time"][-1]), float(pwh["time"][-1]))

    if not (t_end > t_start):
        raise ValueError(
            "The GR and PWH-AG trajectories do not have an overlapping "
            "nondimensional time interval."
        )

    point_count = max(2, int(COMMON_TIME_POINT_COUNT))
    common_time = np.linspace(t_start, t_end, point_count)

    x_gr = np.interp(common_time, gr["time"], gr["x"])
    y_gr = np.interp(common_time, gr["time"], gr["y"])
    x_pwh = np.interp(common_time, pwh["time"], pwh["x"])
    y_pwh = np.interp(common_time, pwh["time"], pwh["y"])

    dx = x_pwh - x_gr
    dy = y_pwh - y_gr
    delta_r = np.sqrt(dx**2 + dy**2)

    return dict(
        time=common_time,
        dx=dx,
        dy=dy,
        delta_r=delta_r,
        max_delta=float(np.max(delta_r)),
        mean_delta=float(np.mean(delta_r)),
        final_delta=float(delta_r[-1]),
    )


def create_difference_figure(diff: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=diff["time"], y=diff["delta_r"], mode="lines",
        name="Δr̃", line=dict(width=2.8)
    ))
    fig.add_trace(go.Scatter(
        x=diff["time"], y=np.abs(diff["dx"]), mode="lines",
        name="|Δx̃|", line=dict(width=2.0, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=diff["time"], y=np.abs(diff["dy"]), mode="lines",
        name="|Δỹ|", line=dict(width=2.0, dash="dot")
    ))

    subtitle = (
        f"common time range: {diff['time'][0]:.4g}–{diff['time'][-1]:.4g} | "
        f"mean Δr̃ = {diff['mean_delta']:.4g} | "
        f"max Δr̃ = {diff['max_delta']:.4g} | "
        f"final Δr̃ = {diff['final_delta']:.4g}"
    )

    apply_standard_layout(
        fig,
        "Trajectory-difference evaluation",
        subtitle,
        width=FULL_WIDTH_FIGURE_WIDTH,
        height=DIFFERENCE_FIGURE_HEIGHT,
        xaxis_title="common nondimensional time",
        yaxis_title="nondimensional difference",
    )
    return fig


# =============================================================================
# HTML OUTPUT
# =============================================================================

def main() -> None:
    pwh_file = locate_file(pwh_results_directory(), PWH_FILENAME)
    gr_file = locate_file(gr_results_directory(), GR_FILENAME)

    print(f"Loading PWH-AG result: {pwh_file}")
    print(f"Loading GR result:     {gr_file}")

    gr = prepare_gr_data(gr_file)
    pwh = prepare_pwh_data(pwh_file)

    if COMMON_AXIS_HALF_WIDTH is None:
        shared_axis_half_width = max(gr["auto_half_width"], pwh["auto_half_width"])
    else:
        shared_axis_half_width = float(COMMON_AXIS_HALF_WIDTH)

    gr_trajectory_fig = create_gr_trajectory_figure(gr, shared_axis_half_width)
    pwh_trajectory_fig = create_pwh_trajectory_figure(pwh, shared_axis_half_width)

    gr_velocity_fig = create_velocity_figure(gr, system_name="GR reference", time_label="τ̃")
    pwh_velocity_fig = create_velocity_figure(pwh, system_name="PWH-AG walker", time_label="t̃")

    overlay_fig = create_overlay_trajectory_figure(gr, pwh, shared_axis_half_width)

    diff = compute_difference_curve(gr, pwh)
    difference_fig = create_difference_figure(diff)

    html_path = SCRIPT_DIRECTORY / HTML_NAME

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>GR and PWH-AG trajectory analysis</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #ffffff; }",
        "h1 { text-align: center; font-size: 26px; margin-bottom: 8px; }",
        ".intro { max-width: 1200px; margin: 0 auto 24px auto; text-align: center; color: #333; }",
        ".row-two { display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; margin-bottom: 24px; }",
        ".row-one { display: flex; justify-content: center; margin-bottom: 24px; }",
        ".plot-box { background: #fff; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>GR and PWH-AG nondimensional trajectory analysis</h1>",
        "<p class='intro'>",
        "Top row: separate nondimensional trajectory plots. ",
        "Second row: separate nondimensional velocity-component plots. ",
        "Third row: overlaid nondimensional trajectories. ",
        "Final row: nondimensional trajectory-difference evaluation on the common overlapping time interval.",
        "</p>",

        "<div class='row-two'>",
        "<div class='plot-box'>",
        pio.to_html(gr_trajectory_fig, include_plotlyjs="cdn", full_html=False),
        "</div>",
        "<div class='plot-box'>",
        pio.to_html(pwh_trajectory_fig, include_plotlyjs=False, full_html=False),
        "</div>",
        "</div>",

        "<div class='row-two'>",
        "<div class='plot-box'>",
        pio.to_html(gr_velocity_fig, include_plotlyjs=False, full_html=False),
        "</div>",
        "<div class='plot-box'>",
        pio.to_html(pwh_velocity_fig, include_plotlyjs=False, full_html=False),
        "</div>",
        "</div>",

        "<div class='row-one'>",
        "<div class='plot-box'>",
        pio.to_html(overlay_fig, include_plotlyjs=False, full_html=False),
        "</div>",
        "</div>",

        "<div class='row-one'>",
        "<div class='plot-box'>",
        pio.to_html(difference_fig, include_plotlyjs=False, full_html=False),
        "</div>",
        "</div>",

        "</body>",
        "</html>",
    ]

    html_path.write_text("\n".join(html_parts), encoding="utf-8")

    print()
    print(f"Shared nondimensional axis range: [-{shared_axis_half_width:.6g}, {shared_axis_half_width:.6g}]")
    print(f"Common evaluation time range: [{diff['time'][0]:.6g}, {diff['time'][-1]:.6g}]")
    print(f"Mean trajectory difference: {diff['mean_delta']:.6g}")
    print(f"Maximum trajectory difference: {diff['max_delta']:.6g}")

    if gr["horizon_crossing_time"] is not None:
        print(
            "GR horizon crossing time: "
            f"{gr['horizon_crossing_time']:.12g}"
        )
    else:
        print("GR horizon crossing time: not found")

    if pwh["horizon_crossing_time"] is not None:
        print(
            "PWH-AG horizon crossing time: "
            f"{pwh['horizon_crossing_time']:.12g}"
        )
    else:
        print("PWH-AG horizon crossing time: not found")

    print(f"Saved combined HTML to: {html_path}")


if __name__ == "__main__":
    main()
