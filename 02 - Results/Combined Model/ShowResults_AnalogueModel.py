"""
Show combined PWH-AG simulation results.

This script is meant for .npz files produced by the combined PWH-AG solver.
It animates the surface field, shows the droplet trajectory, overlays the
group horizon / drain / sponge geometry, and plots droplet velocities against
the local background flow.
"""

from pathlib import Path
import zipfile

import numpy as np
from numpy.lib import format
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. User settings
# ============================================================

# File search.
RESULTS_DIR = Path(__file__).resolve().parent
filename_name = "PWH_AG_GPU_CircOrbit_Azimuthal.npz"  # change this name when needed

# Animation display.
interval_ms = 1
surface_cmap = "Blues"
display_stride = 2
trajectory_stride = 50

# The simulation code stores this many frames per Faraday period.
stored_frames_per_faraday_period = 6

# Choose how many of the stored frames per Faraday period are actually animated.
animated_frames_per_faraday_period = 3

# Fixed colour scale.
# If use_manual_eta_scale is False, the scale is still fixed in time, but chosen
# once from all displayed frames using the percentile below.
use_manual_eta_scale = True
eta_vmax_manual = 0.2e-5
color_percentile = 99.5
color_scale_multiplier = 1.5

# Geometry toggles.
show_group_horizon = True
show_phase_horizon = False
show_drain_radius = True
show_inner_sponge = True
show_outer_sponge = True

# Velocity diagnostic toggles.
# The ideal-flow invalid region is shown by a gray shaded region starting
# when the droplet enters the inner sponge, where the flow is no longer ideal -A/r.
shade_nonideal_background_region = True
show_horizon_crossing_line = True
show_phase_crossing_line = True
show_current_time_marker = True
stop_background_inside_drain = True

# Keep the animation lighter for local viewing.
# Use None to animate all selected frames.
max_animation_frames = 600

# Optional velocity-axis clipping for readability.
# Use None for automatic Matplotlib scaling.
speed_ylim = None
radial_velocity_ylim = None
azimuthal_velocity_ylim = None

# Optional interactive HTML outputs.
# The combined file contains the trajectory, the three droplet/background
# velocity plots, and the velocity-difference plot.
save_results_html = True
results_html_name = None          # None -> filename stem + "_results.html"

# Legacy separate outputs. These remain available but are disabled by default.
save_trajectory_html = False
save_velocity_html = False
trajectory_html_name = None       # None -> filename stem + "_trajectory.html"
velocity_html_name = None         # None -> filename stem + "_velocities.html"

# Trajectory HTML settings.
trajectory_html_stride = 10
max_colored_segments = 1500
trajectory_colorscale = "Turbo"

# Plot styling.
figure_title_prefix = "Combined PWH-AG output"
wave_colorbar_label = r"$\eta(x,y,t)$ [m]"
line_width = 1.4
geometry_line_width = 1.2


# ============================================================
# 2. Locate and load data file
# ============================================================

matches = list(RESULTS_DIR.rglob(filename_name))

if len(matches) == 0:
    raise FileNotFoundError(
        f"Could not find {filename_name} anywhere inside {RESULTS_DIR}"
    )

if len(matches) > 1:
    print("Multiple matching files found:")
    for i, match in enumerate(matches):
        print(f"[{i}] {match}")

    choice = int(input("Choose file number: "))
    filename = matches[choice]
else:
    filename = matches[0]

print(f"Loading: {filename}")

data = np.load(filename)


# ============================================================
# 3. Helper functions
# ============================================================



def mmap_uncompressed_npz_array(npz_path, array_name):
    """
    Memory-map an uncompressed .npy member directly inside an .npz archive.

    This reads selected slices from the existing .npz file without loading the
    complete array into RAM and without extracting a duplicate file. It works
    for arrays written with np.savez (ZIP_STORED), but not np.savez_compressed.
    """
    npz_path = Path(npz_path)
    member_name = f"{array_name}.npy"

    with zipfile.ZipFile(npz_path, "r") as archive:
        try:
            info = archive.getinfo(member_name)
        except KeyError as exc:
            raise KeyError(
                f"Array {array_name!r} was not found in {npz_path.name}."
            ) from exc

        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{member_name} is compressed inside {npz_path.name}. "
                "Direct memory mapping requires an uncompressed .npz created "
                "with np.savez."
            )

        local_header_offset = info.header_offset

    with npz_path.open("rb") as file_obj:
        file_obj.seek(local_header_offset)
        local_header = file_obj.read(30)

        if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
            raise ValueError(
                f"Could not parse the ZIP header for {member_name}."
            )

        filename_length = int.from_bytes(local_header[26:28], "little")
        extra_length = int.from_bytes(local_header[28:30], "little")
        npy_start = local_header_offset + 30 + filename_length + extra_length

        file_obj.seek(npy_start)
        version = format.read_magic(file_obj)
        shape, fortran_order, dtype = format._read_array_header(file_obj, version)
        array_data_offset = file_obj.tell()

    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=array_data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def scalar(npz_data, key, default=None):
    """
    Safely read a scalar value from a loaded .npz file.
    """
    if key not in npz_data.files:
        return default

    value = npz_data[key]
    if np.ndim(value) == 0:
        return value.item()
    if np.size(value) == 1:
        return value.reshape(-1)[0].item()
    return value


def fmt(value, fmt_spec=".5e"):
    """
    Format a value, while keeping missing values readable.
    """
    if value is None:
        return "not saved"
    try:
        return f"{value:{fmt_spec}}"
    except (TypeError, ValueError):
        return str(value)


def first_inward_crossing_time(radius, threshold, time):
    """
    Return the first interpolated time where radius crosses from outside to inside.
    If the trajectory starts inside, no crossing marker is returned.
    """
    if threshold is None:
        return None

    radius = np.asarray(radius)
    time = np.asarray(time)

    if len(radius) < 2 or radius[0] <= threshold:
        return None

    crossing_indices = np.flatnonzero((radius[:-1] > threshold) & (radius[1:] <= threshold))
    if len(crossing_indices) == 0:
        return None

    i = crossing_indices[0]
    r0 = radius[i]
    r1 = radius[i + 1]
    t0 = time[i]
    t1 = time[i + 1]

    if r1 == r0:
        return float(t1)

    alpha = (threshold - r0) / (r1 - r0)
    return float(t0 + alpha * (t1 - t0))


def shade_mask_regions(ax, time, mask, label, color="0.85", alpha=0.45):
    """
    Shade time intervals where a boolean mask is true.
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or not np.any(mask):
        return

    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])

    first = True
    for start, end in zip(starts, ends):
        ax.axvspan(
            time[start],
            time[end],
            color=color,
            alpha=alpha,
            linewidth=0.0,
            label=label if first else None,
        )
        first = False


def circle_xy(radius, n=600):
    """
    Return x,y arrays for a circle.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    return radius * np.cos(theta), radius * np.sin(theta)


def draw_square(ax, half_width, **kwargs):
    """
    Draw a square centered at the origin.
    """
    xs = [-half_width, half_width, half_width, -half_width, -half_width]
    ys = [-half_width, -half_width, half_width, half_width, -half_width]
    ax.plot(xs, ys, **kwargs)


def plot_geometry_matplotlib(ax, r_H, r_phase, r_drain, r_inner_sponge_outer, outer_half_width):
    """
    Overlay combined-model geometry on a Matplotlib axis.
    """
    if show_outer_sponge and outer_half_width is not None and outer_half_width > 0.0:
        draw_square(
            ax,
            outer_half_width,
            color="0.2",
            linestyle=":",
            linewidth=geometry_line_width,
            label="sponge",
        )

    if show_group_horizon and r_H is not None:
        cx, cy = circle_xy(r_H)
        ax.plot(
            cx,
            cy,
            color="k",
            linestyle="-",
            linewidth=geometry_line_width,
            label="group horizon",
        )

    if show_phase_horizon and r_phase is not None:
        cx, cy = circle_xy(r_phase)
        ax.plot(
            cx,
            cy,
            color="0.35",
            linestyle="--",
            linewidth=geometry_line_width,
            label="phase horizon",
        )

    if show_drain_radius and r_drain is not None:
        cx, cy = circle_xy(r_drain)
        ax.plot(
            cx,
            cy,
            color="k",
            linestyle="-.",
            linewidth=geometry_line_width,
            label="_nolegend_",
        )

    if show_inner_sponge and r_inner_sponge_outer is not None:
        cx, cy = circle_xy(r_inner_sponge_outer)
        ax.plot(
            cx,
            cy,
            color="0.35",
            linestyle=":",
            linewidth=geometry_line_width,
            label="_nolegend_",
        )


def setup_scientific_axes(ax):
    """
    Basic scientific-looking axis formatting.
    """
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def safe_radius_components(x_pos, y_pos):
    """
    Return r, e_r and e_theta components for a trajectory.
    """
    r = np.sqrt(x_pos**2 + y_pos**2)
    r_safe = np.where(r > 1.0e-14, r, np.nan)

    er_x = x_pos / r_safe
    er_y = y_pos / r_safe
    et_x = -y_pos / r_safe
    et_y = x_pos / r_safe

    return r, er_x, er_y, et_x, et_y


def smooth_ramp(s):
    """
    Smooth ramp from 0 to 1 for s in [0,1].
    """
    s = np.clip(s, 0.0, 1.0)
    return s**2 * (3.0 - 2.0 * s)


def add_plotly_circle(fig, radius, name, line=None, showlegend=True):
    """
    Add a circle to a Plotly figure.
    """
    if radius is None:
        return

    import plotly.graph_objects as go

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    x_c = radius * np.cos(theta)
    y_c = radius * np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=x_c,
            y=y_c,
            mode="lines",
            name=name if name is not None else "",
            showlegend=showlegend and name is not None,
            line=line or dict(width=1.5),
            hoverinfo="skip",
        )
    )


def add_plotly_square(fig, half_width, name, line=None, showlegend=True):
    """
    Add a square to a Plotly figure.
    """
    if half_width is None:
        return

    import plotly.graph_objects as go

    xs = [-half_width, half_width, half_width, -half_width, -half_width]
    ys = [-half_width, -half_width, half_width, half_width, -half_width]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name if name is not None else "",
            showlegend=showlegend and name is not None,
            line=line or dict(width=1.5, dash="dot"),
            hoverinfo="skip",
        )
    )


def add_plotly_label(fig, x_pos, y_pos, text, font_size=13):
    """
    Add a direct label to a Plotly geometry line.
    """
    if x_pos is None or y_pos is None or text is None:
        return

    fig.add_annotation(
        x=x_pos,
        y=y_pos,
        text=text,
        showarrow=False,
        font=dict(size=font_size, color="black"),
        bgcolor="rgba(255,255,255,0.70)",
        bordercolor="rgba(255,255,255,0.0)",
        xanchor="center",
        yanchor="middle",
    )


# ============================================================
# 4. Read arrays and saved parameters
# ============================================================

x = np.asarray(data["x"])
y = np.asarray(data["y"])
frames = mmap_uncompressed_npz_array(filename, "frames")
times = np.asarray(data["times"])

has_animation_frames = (
    frames.ndim == 3
    and frames.shape[0] > 0
    and len(times) > 0
)

time_history = np.asarray(data["time_history"])
drop_x = np.asarray(data["drop_x_history"])
drop_y = np.asarray(data["drop_y_history"])
drop_z = np.asarray(data["drop_z_history"])
contact = np.asarray(data["contact_history"])

if "drop_u_history" in data.files and "drop_v_history" in data.files:
    drop_u = np.asarray(data["drop_u_history"])
    drop_v = np.asarray(data["drop_v_history"])
else:
    drop_u = np.gradient(drop_x, time_history)
    drop_v = np.gradient(drop_y, time_history)

if "drop_w_history" in data.files:
    drop_w = np.asarray(data["drop_w_history"])
else:
    drop_w = np.gradient(drop_z, time_history)

if "drop_speed_history" in data.files:
    drop_speed = np.asarray(data["drop_speed_history"])
else:
    drop_speed = np.sqrt(drop_u**2 + drop_v**2)

# Main saved scalar parameters.
L_phys = scalar(data, "L_phys", float(np.max(np.abs(x))))
N_saved = int(scalar(data, "N", len(x)))
dx_saved = scalar(data, "dx", float(abs(x[1] - x[0])) if len(x) > 1 else None)
dt_saved = scalar(data, "dt", None)
n_steps_saved = scalar(data, "n_steps", len(time_history) - 1)
plot_every_saved = scalar(data, "plot_every", None)

r_H = scalar(data, "r_H", None)
A = scalar(data, "A", None)
B = scalar(data, "B", 0.0)
B_over_A = scalar(
    data,
    "B_over_A",
    (B / A if A not in (None, 0.0) else None),
)
c_p_kF = scalar(data, "c_p_kF", None)
kF = scalar(data, "kF", None)
lambdaF = scalar(data, "lambdaF", None)

r_phase = A / c_p_kF if (A is not None and c_p_kF not in (None, 0.0)) else None

r_drain = scalar(data, "r_drain", None)
inner_sponge_width = scalar(data, "inner_sponge_width", None)
r_inner_sponge_outer = scalar(data, "r_inner_sponge_outer", None)
outer_sponge_width = scalar(data, "outer_sponge_width", None)
outer_sponge_inner_half_width = scalar(data, "outer_sponge_inner_half_width", None)

if outer_sponge_inner_half_width is None and outer_sponge_width is not None:
    outer_sponge_inner_half_width = L_phys - outer_sponge_width

Gamma = scalar(data, "Gamma", None)
GammaF = scalar(data, "GammaF", None)
R0 = scalar(data, "R0", None)

# Initial 2D state from the saved history.
x0 = float(drop_x[0])
y0 = float(drop_y[0])
u0 = float(drop_u[0])
v0 = float(drop_v[0])

domain_side_length = 2.0 * L_phys


# ============================================================
# 5. Derived velocity diagnostics
# ============================================================

r_drop, er_x, er_y, et_x, et_y = safe_radius_components(drop_x, drop_y)

drop_vr = drop_u * er_x + drop_v * er_y
drop_vtheta = drop_u * et_x + drop_v * et_y

# Background-flow diagnostic along the droplet path.
# This reconstructs the same regularised draining-vortex field used in the solver.
if A is not None:
    r_safe = np.where(r_drop > 1.0e-14, r_drop, np.nan)

    if r_drain is not None and inner_sponge_width not in (None, 0.0):
        activation_path = smooth_ramp(
            (r_drop - r_drain) / inner_sponge_width
        )
        r2_safe = np.maximum(r_drop**2, r_drain**2)

        vB_r = -activation_path * A * r_drop / r2_safe
        vB_theta = activation_path * B * r_drop / r2_safe
    else:
        vB_r = -A / r_safe
        vB_theta = B / r_safe

    vB_speed = np.sqrt(vB_r**2 + vB_theta**2)

    if stop_background_inside_drain and r_drain is not None:
        inside_drain = r_drop <= r_drain
        vB_r = vB_r.astype(float)
        vB_theta = vB_theta.astype(float)
        vB_speed = vB_speed.astype(float)
        vB_r[inside_drain] = np.nan
        vB_theta[inside_drain] = np.nan
        vB_speed[inside_drain] = np.nan
else:
    vB_r = np.full_like(drop_speed, np.nan, dtype=float)
    vB_theta = np.full_like(drop_speed, np.nan, dtype=float)
    vB_speed = np.full_like(drop_speed, np.nan, dtype=float)

# Velocity of the droplet relative to the local background flow.
# Component differences are signed; the total is the magnitude of the relative
# velocity vector.
delta_vr = drop_vr - vB_r
delta_vtheta = drop_vtheta - vB_theta
delta_vtotal = np.sqrt(delta_vr**2 + delta_vtheta**2)

horizon_cross_time = first_inward_crossing_time(r_drop, r_H, time_history)
phase_cross_time = first_inward_crossing_time(r_drop, r_phase, time_history)
drain_cross_time = first_inward_crossing_time(r_drop, r_drain, time_history)

# Region where the ideal draining-vortex comparison should not be interpreted.
if r_inner_sponge_outer is not None:
    nonideal_background_mask = r_drop <= r_inner_sponge_outer
else:
    nonideal_background_mask = np.zeros_like(r_drop, dtype=bool)


# ============================================================
# 6. Print concise loaded-run summary
# ============================================================

print()
print("Loaded combined-model run")
print("-------------------------")
print(f"File: {filename.name}")
print(f"Domain side length: {fmt(domain_side_length)} m")
print(f"Grid: N = {N_saved}, dx = {fmt(dx_saved)} m")
print(f"Initial position: x0 = {fmt(x0)} m, y0 = {fmt(y0)} m")
print(f"Initial velocity: u0 = {fmt(u0)} m/s, v0 = {fmt(v0)} m/s")
print(f"Group horizon radius: r_H = {fmt(r_H)} m")
print(f"Radial sink parameter: A = {fmt(A)} m^2/s")
print(f"Azimuthal flow parameter: B = {fmt(B)} m^2/s")
print(f"Azimuthal-to-radial ratio: B/A = {fmt(B_over_A)}")
print(f"Excluded drain radius: r_drain = {fmt(r_drain)} m")
if show_phase_horizon:
    print(f"Phase horizon radius: r_p = {fmt(r_phase)} m")
if Gamma is not None and GammaF not in (None, 0.0):
    print(f"Gamma/GammaF = {Gamma / GammaF:.5f}")
print(f"Saved frames: {len(times)}")
print(f"Full history points: {len(time_history)}")
print()


# ============================================================
# 7. Prepare animation and optional Matplotlib figure
# ============================================================

fig = None
ani = None

if has_animation_frames:
    animated_frames_per_faraday_period = int(animated_frames_per_faraday_period)
    if animated_frames_per_faraday_period < 1:
        raise ValueError("animated_frames_per_faraday_period must be at least 1.")
    if animated_frames_per_faraday_period > stored_frames_per_faraday_period:
        raise ValueError(
            "animated_frames_per_faraday_period cannot exceed "
            "stored_frames_per_faraday_period."
        )

    frame_interval = stored_frames_per_faraday_period / animated_frames_per_faraday_period
    animation_indices = np.unique(
        np.round(np.arange(0, len(times), frame_interval)).astype(int)
    )
    animation_indices = animation_indices[animation_indices < len(times)]

    if max_animation_frames is not None and len(animation_indices) > max_animation_frames:
        keep = np.linspace(
            0, len(animation_indices) - 1, max_animation_frames
        ).round().astype(int)
        animation_indices = animation_indices[keep]

    times_anim = times[animation_indices]

    if display_stride > 1:
        x_display = x[::display_stride]
        y_display = y[::display_stride]
    else:
        x_display = x
        y_display = y

    def get_display_frame(animation_index):
        """Read only one selected frame from the existing .npz file."""
        stored_index = int(animation_indices[animation_index])
        if display_stride > 1:
            return np.asarray(
                frames[stored_index, ::display_stride, ::display_stride]
            )
        return np.asarray(frames[stored_index])

    extent = [x_display[0], x_display[-1], y_display[0], y_display[-1]]

    if use_manual_eta_scale:
        vmax = float(eta_vmax_manual)
    else:
        # Estimate a fixed colour scale from a small set of selected frames,
        # without loading the complete frame stack into memory.
        sample_count = min(20, len(animation_indices))
        sample_positions = np.linspace(
            0, len(animation_indices) - 1, sample_count
        ).round().astype(int)
        scale_samples = np.concatenate(
            [np.abs(get_display_frame(i)).ravel() for i in sample_positions]
        )
        eta_scale = np.percentile(scale_samples, color_percentile)
        vmax = max(color_scale_multiplier * eta_scale, 1.0e-12)
    vmin = -vmax

    print(f"Animated frames: {len(times_anim)}")
    print()

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{figure_title_prefix}: {filename.name}", fontsize=12)

    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.10, 1.35],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.30,
        hspace=0.25,
    )

    ax_speed = fig.add_subplot(gs[0, 0])
    ax_radial = fig.add_subplot(gs[1, 0], sharex=ax_speed)
    ax_azimuthal = fig.add_subplot(gs[2, 0], sharex=ax_speed)
    ax_wave = fig.add_subplot(gs[:, 1])

    ax_speed.plot(time_history, drop_speed, linewidth=line_width, label="droplet")
    ax_speed.plot(time_history, vB_speed, linewidth=line_width, linestyle="--", label="background")
    ax_speed.set_ylabel(r"$|\mathbf{v}|$ [m/s]")
    ax_speed.set_title("Velocity diagnostics")

    ax_radial.plot(time_history, drop_vr, linewidth=line_width, label="droplet")
    ax_radial.plot(time_history, vB_r, linewidth=line_width, linestyle="--", label="background")
    ax_radial.set_ylabel(r"$v_r$ [m/s]")

    ax_azimuthal.plot(time_history, drop_vtheta, linewidth=line_width, label="droplet")
    ax_azimuthal.plot(time_history, vB_theta, linewidth=line_width, linestyle="--", label="background")
    ax_azimuthal.set_ylabel(r"$v_\theta$ [m/s]")
    ax_azimuthal.set_xlabel("time [s]")

    if speed_ylim is not None:
        ax_speed.set_ylim(speed_ylim)
    if radial_velocity_ylim is not None:
        ax_radial.set_ylim(radial_velocity_ylim)
    if azimuthal_velocity_ylim is not None:
        ax_azimuthal.set_ylim(azimuthal_velocity_ylim)

    for ax in (ax_speed, ax_radial, ax_azimuthal):
        if shade_nonideal_background_region:
            shade_mask_regions(
                ax, time_history, nonideal_background_mask,
                label=None, color="0.82", alpha=0.45,
            )
        setup_scientific_axes(ax)
        if show_horizon_crossing_line and horizon_cross_time is not None:
            ax.axvline(horizon_cross_time, color="0.25", linestyle="--", linewidth=1.0)
        if show_phase_horizon and show_phase_crossing_line and phase_cross_time is not None:
            ax.axvline(phase_cross_time, color="0.45", linestyle=":", linewidth=1.0)

    ax_speed.legend(loc="upper left", fontsize=8)

    if show_current_time_marker:
        time_marker_speed = ax_speed.axvline(times_anim[0], color="k", linestyle="-", linewidth=0.8, alpha=0.7)
        time_marker_radial = ax_radial.axvline(times_anim[0], color="k", linestyle="-", linewidth=0.8, alpha=0.7)
        time_marker_azimuthal = ax_azimuthal.axvline(times_anim[0], color="k", linestyle="-", linewidth=0.8, alpha=0.7)
    else:
        time_marker_speed = None
        time_marker_radial = None
        time_marker_azimuthal = None

    im = ax_wave.imshow(
        get_display_frame(0), extent=extent, origin="lower",
        cmap=surface_cmap, vmin=vmin, vmax=vmax,
        interpolation="bilinear",
    )
    droplet_marker, = ax_wave.plot([], [], "ko", markersize=5, label="droplet")
    trajectory_line, = ax_wave.plot([], [], "k-", linewidth=1.0, alpha=0.7, label="trajectory")

    plot_geometry_matplotlib(
        ax_wave,
        r_H=r_H,
        r_phase=r_phase,
        r_drain=r_drain,
        r_inner_sponge_outer=r_inner_sponge_outer,
        outer_half_width=outer_sponge_inner_half_width,
    )

    cbar = fig.colorbar(im, ax=ax_wave)
    cbar.set_label(wave_colorbar_label)
    ax_wave.set_xlabel("x [m]")
    ax_wave.set_ylabel("y [m]")
    ax_wave.set_aspect("equal")
    ax_wave.tick_params(direction="in", top=True, right=True)
    ax_wave.legend(loc="upper right", fontsize=8)
    title = ax_wave.set_title("")

    def update(frame_index):
        t_frame = times_anim[frame_index]
        hist_index = int(np.argmin(np.abs(time_history - t_frame)))
        im.set_data(get_display_frame(frame_index))
        droplet_marker.set_data([drop_x[hist_index]], [drop_y[hist_index]])
        trajectory_line.set_data(
            drop_x[:hist_index + 1:trajectory_stride],
            drop_y[:hist_index + 1:trajectory_stride],
        )
        if show_current_time_marker:
            for marker in (time_marker_speed, time_marker_radial, time_marker_azimuthal):
                marker.set_xdata([t_frame, t_frame])
        title.set_text(f"Surface elevation, t = {t_frame:.4f} s")
        return im, droplet_marker, trajectory_line, title

    ani = FuncAnimation(
        fig, update, frames=len(times_anim), interval=interval_ms, blit=False
    )
else:
    print(
        "No wave-field frames were saved. Skipping the Matplotlib animation; "
        "HTML outputs will still be created."
    )
    print()


# ============================================================
# 9. Optional interactive trajectory HTML
# ============================================================

def save_interactive_trajectory():
    """
    Save an interactive trajectory figure.
    """
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale

    html_path = (
        filename.with_name(filename.stem + "_trajectory.html")
        if trajectory_html_name is None
        else Path(trajectory_html_name)
    )

    stride = max(1, int(trajectory_html_stride))
    idx = np.arange(0, len(drop_x), stride)

    if len(idx) > max_colored_segments + 1:
        stride_factor = int(np.ceil((len(idx) - 1) / max_colored_segments))
        idx = idx[::stride_factor]

    x_tr = drop_x[idx]
    y_tr = drop_y[idx]
    speed_tr = drop_speed[idx]
    time_tr = time_history[idx]

    speed_min = float(np.nanmin(speed_tr))
    speed_max = float(np.nanmax(speed_tr))
    denom = max(speed_max - speed_min, 1.0e-15)

    fig_traj = go.Figure()

    # Coloured trajectory segments.
    for i in range(len(idx) - 1):
        s_mid = 0.5 * (speed_tr[i] + speed_tr[i + 1])
        cval = (s_mid - speed_min) / denom
        color = sample_colorscale(trajectory_colorscale, cval)[0]

        fig_traj.add_trace(
            go.Scattergl(
                x=[x_tr[i], x_tr[i + 1]],
                y=[y_tr[i], y_tr[i + 1]],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Invisible marker trace gives a colorbar and hover values.
    fig_traj.add_trace(
        go.Scattergl(
            x=x_tr,
            y=y_tr,
            mode="markers",
            marker=dict(
                size=5,
                color=speed_tr,
                colorscale=trajectory_colorscale,
                colorbar=dict(title="speed [m/s]"),
            ),
            name="trajectory speed",
            showlegend=False,
            customdata=np.column_stack([time_tr, speed_tr]),
            hovertemplate=(
                "x = %{x:.5e} m<br>"
                "y = %{y:.5e} m<br>"
                "t = %{customdata[0]:.5e} s<br>"
                "speed = %{customdata[1]:.5e} m/s"
                "<extra></extra>"
            ),
        )
    )

    add_plotly_circle(fig_traj, r_H, "group horizon", line=dict(color="black", width=2), showlegend=False)
    if show_phase_horizon:
        add_plotly_circle(fig_traj, r_phase, "phase horizon", line=dict(color="gray", width=2, dash="dash"), showlegend=False)
    add_plotly_circle(fig_traj, r_drain, None, line=dict(color="black", width=2, dash="dashdot"), showlegend=False)
    add_plotly_circle(fig_traj, r_inner_sponge_outer, None, line=dict(color="gray", width=2, dash="dot"), showlegend=False)
    add_plotly_square(fig_traj, outer_sponge_inner_half_width, "sponge", line=dict(color="black", width=2, dash="dot"), showlegend=False)

    # Direct labels avoid legend/colorbar overlap.
    if r_H is not None:
        add_plotly_label(fig_traj, 0.0, r_H, "r_H")
    if show_phase_horizon and r_phase is not None:
        add_plotly_label(fig_traj, 0.0, r_phase, "r_p")
    if outer_sponge_inner_half_width is not None:
        add_plotly_label(fig_traj, 0.0, outer_sponge_inner_half_width, "sponge")

    subtitle = (
        f"domain = {domain_side_length:.3g} m | "
        f"N = {N_saved} | dx = {dx_saved:.3g} m | "
        f"(x0,y0)=({x0:.3g},{y0:.3g}) m | "
        f"(u0,v0)=({u0:.3g},{v0:.3g}) m/s | "
        f"A={A:.5g} m^2/s | B={B:.5g} m^2/s | B/A={B_over_A:.5g}"
    )

    fig_traj.update_layout(
        title=f"Combined PWH-AG trajectory: {filename.name}<br><sup>{subtitle}</sup>",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        template="plotly_white",
        width=900,
        height=800,
        showlegend=False,
    )
    fig_traj.update_yaxes(scaleanchor="x", scaleratio=1)

    fig_traj.write_html(html_path)
    print(f"Saved trajectory HTML to: {html_path}")


# ============================================================
# 10. Optional interactive velocity HTML
# ============================================================

def save_interactive_velocities():
    """
    Save interactive velocity diagnostic plots.

    The three diagnostics are written as separate Plotly figures in one HTML
    file so each graph has its own compact droplet/background legend.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    html_path = (
        filename.with_name(filename.stem + "_velocities.html")
        if velocity_html_name is None
        else Path(velocity_html_name)
    )

    subtitle = (
        f"domain = {domain_side_length:.3g} m | "
        f"N = {N_saved} | dx = {dx_saved:.3g} m | "
        f"(x0,y0)=({x0:.3g},{y0:.3g}) m | "
        f"(u0,v0)=({u0:.3g},{v0:.3g}) m/s | "
        f"A={A:.5g} m^2/s | B={B:.5g} m^2/s | B/A={B_over_A:.5g}"
    )

    def add_nonideal_shading(fig):
        """
        Add the gray non-ideal background-flow region.
        """
        mask = np.asarray(nonideal_background_mask, dtype=bool)
        if not np.any(mask):
            return

        starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
        ends = np.flatnonzero(mask & np.r_[~mask[1:], True])

        for start, end in zip(starts, ends):
            fig.add_vrect(
                x0=time_history[start],
                x1=time_history[end],
                fillcolor="lightgray",
                opacity=0.25,
                line_width=0,
                layer="below",
            )

    def add_crossing_lines(fig):
        """
        Add r_H and optional r_p crossing markers.
        """
        if show_horizon_crossing_line and horizon_cross_time is not None:
            fig.add_vline(
                x=horizon_cross_time,
                line_dash="dash",
                line_width=1,
                line_color="black",
            )
            fig.add_annotation(
                x=horizon_cross_time,
                y=1.0,
                xref="x",
                yref="paper",
                text="r_H",
                showarrow=False,
                textangle=-90,
                xanchor="right",
                yanchor="top",
                font=dict(size=10),
            )

        if show_phase_horizon and show_phase_crossing_line and phase_cross_time is not None:
            fig.add_vline(
                x=phase_cross_time,
                line_dash="dot",
                line_width=1,
                line_color="gray",
            )
            fig.add_annotation(
                x=phase_cross_time,
                y=1.0,
                xref="x",
                yref="paper",
                text="r_p",
                showarrow=False,
                textangle=-90,
                xanchor="right",
                yanchor="top",
                font=dict(size=10),
            )

    def make_velocity_figure(title, y_droplet, y_background, y_axis_label):
        """
        Build one compact scientific velocity figure.
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time_history,
                y=y_droplet,
                mode="lines",
                name="droplet",
                line=dict(width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_history,
                y=y_background,
                mode="lines",
                name="background",
                line=dict(width=2, dash="dash"),
            )
        )

        add_nonideal_shading(fig)
        add_crossing_lines(fig)

        fig.update_layout(
            title=f"{title}<br><sup>{subtitle}</sup>",
            template="plotly_white",
            width=1000,
            height=360,
            margin=dict(l=80, r=30, t=75, b=55),
            legend=dict(
                orientation="h",
                x=0.99,
                y=0.99,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.75)",
                borderwidth=0,
                font=dict(size=11),
            ),
            hovermode="x unified",
        )

        fig.update_xaxes(
            title_text="time [s]",
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="inside",
            gridcolor="rgba(0,0,0,0.10)",
        )
        fig.update_yaxes(
            title_text=y_axis_label,
            showline=True,
            linecolor="black",
            mirror=True,
            ticks="inside",
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.25)",
        )

        return fig

    fig_speed = make_velocity_figure(
        "Total horizontal speed",
        drop_speed,
        vB_speed,
        "speed [m/s]",
    )
    fig_radial = make_velocity_figure(
        "Radial velocity component",
        drop_vr,
        vB_r,
        "radial velocity [m/s]",
    )
    fig_azimuthal = make_velocity_figure(
        "Azimuthal velocity component",
        drop_vtheta,
        vB_theta,
        "azimuthal velocity [m/s]",
    )

    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        pio.to_html(fig_speed, include_plotlyjs="cdn", full_html=False),
        pio.to_html(fig_radial, include_plotlyjs=False, full_html=False),
        pio.to_html(fig_azimuthal, include_plotlyjs=False, full_html=False),
        "</body></html>",
    ]

    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Saved velocity HTML to: {html_path}")


def save_combined_results_html():
    """Save trajectory, velocities, and velocity differences in one HTML file."""
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.colors import sample_colorscale

    html_path = (
        filename.with_name(filename.stem + "_results.html")
        if results_html_name is None
        else Path(results_html_name)
    )

    subtitle = (
        f"domain = {domain_side_length:.3g} m | "
        f"N = {N_saved} | dx = {dx_saved:.3g} m | "
        f"(x0,y0)=({x0:.3g},{y0:.3g}) m | "
        f"(u0,v0)=({u0:.3g},{v0:.3g}) m/s | "
        f"A={A:.5g} m^2/s | B={B:.5g} m^2/s | B/A={B_over_A:.5g}"
    )

    def add_nonideal_shading(fig):
        mask = np.asarray(nonideal_background_mask, dtype=bool)
        if not np.any(mask):
            return
        starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
        ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
        for start, end in zip(starts, ends):
            fig.add_vrect(
                x0=time_history[start], x1=time_history[end],
                fillcolor="lightgray", opacity=0.25,
                line_width=0, layer="below",
            )

    def add_crossing_lines(fig):
        if show_horizon_crossing_line and horizon_cross_time is not None:
            fig.add_vline(x=horizon_cross_time, line_dash="dash", line_width=1, line_color="black")
            fig.add_annotation(
                x=horizon_cross_time, y=1.0, xref="x", yref="paper",
                text="r_H", showarrow=False, textangle=-90,
                xanchor="right", yanchor="top", font=dict(size=10),
            )
        if show_phase_horizon and show_phase_crossing_line and phase_cross_time is not None:
            fig.add_vline(x=phase_cross_time, line_dash="dot", line_width=1, line_color="gray")
            fig.add_annotation(
                x=phase_cross_time, y=1.0, xref="x", yref="paper",
                text="r_p", showarrow=False, textangle=-90,
                xanchor="right", yanchor="top", font=dict(size=10),
            )

    def style_time_figure(fig, title, ylabel):
        add_nonideal_shading(fig)
        add_crossing_lines(fig)
        fig.update_layout(
            title=f"{title}<br><sup>{subtitle}</sup>",
            template="plotly_white", width=700, height=420,
            margin=dict(l=85, r=35, t=80, b=60),
            legend=dict(
                orientation="h", x=0.99, y=0.99,
                xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.80)", borderwidth=0,
            ),
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="time [s]", showline=True, mirror=True, ticks="inside")
        fig.update_yaxes(
            title_text=ylabel, showline=True, mirror=True, ticks="inside",
            zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
        )

    def make_pair_figure(title, droplet, background, ylabel):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_history, y=droplet, mode="lines", name="droplet", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=time_history, y=background, mode="lines", name="background", line=dict(width=2, dash="dash")))
        style_time_figure(fig, title, ylabel)
        return fig

    # Trajectory figure.
    stride = max(1, int(trajectory_html_stride))
    idx = np.arange(0, len(drop_x), stride)
    if idx[-1] != len(drop_x) - 1:
        idx = np.append(idx, len(drop_x) - 1)
    if len(idx) > max_colored_segments + 1:
        factor = int(np.ceil((len(idx) - 1) / max_colored_segments))
        idx = idx[::factor]
        if idx[-1] != len(drop_x) - 1:
            idx = np.append(idx, len(drop_x) - 1)

    x_tr, y_tr = drop_x[idx], drop_y[idx]
    speed_tr, time_tr = drop_speed[idx], time_history[idx]
    smin, smax = float(np.nanmin(speed_tr)), float(np.nanmax(speed_tr))
    denom = max(smax - smin, 1.0e-15)

    fig_traj = go.Figure()
    for i in range(len(idx) - 1):
        smid = 0.5 * (speed_tr[i] + speed_tr[i + 1])
        color = sample_colorscale(trajectory_colorscale, (smid - smin) / denom)[0]
        fig_traj.add_trace(go.Scattergl(
            x=[x_tr[i], x_tr[i+1]], y=[y_tr[i], y_tr[i+1]],
            mode="lines", line=dict(color=color, width=3),
            showlegend=False, hoverinfo="skip",
        ))
    fig_traj.add_trace(go.Scattergl(
        x=x_tr, y=y_tr, mode="markers", showlegend=False,
        marker=dict(size=5, color=speed_tr, colorscale=trajectory_colorscale,
                    colorbar=dict(title="speed [m/s]")),
        customdata=np.column_stack([time_tr, speed_tr]),
        hovertemplate=(
            "x = %{x:.5e} m<br>y = %{y:.5e} m<br>"
            "t = %{customdata[0]:.5e} s<br>speed = %{customdata[1]:.5e} m/s"
            "<extra></extra>"
        ),
    ))
    add_plotly_circle(fig_traj, r_H, "group horizon", line=dict(color="black", width=2), showlegend=False)
    if show_phase_horizon:
        add_plotly_circle(fig_traj, r_phase, "phase horizon", line=dict(color="gray", width=2, dash="dash"), showlegend=False)
    add_plotly_circle(fig_traj, r_drain, None, line=dict(color="black", width=2, dash="dashdot"), showlegend=False)
    add_plotly_circle(fig_traj, r_inner_sponge_outer, None, line=dict(color="gray", width=2, dash="dot"), showlegend=False)
    add_plotly_square(fig_traj, outer_sponge_inner_half_width, "sponge", line=dict(color="black", width=2, dash="dot"), showlegend=False)
    if r_H is not None:
        add_plotly_label(fig_traj, 0.0, r_H, "r_H")
    if show_phase_horizon and r_phase is not None:
        add_plotly_label(fig_traj, 0.0, r_phase, "r_p")
    if outer_sponge_inner_half_width is not None:
        add_plotly_label(fig_traj, 0.0, outer_sponge_inner_half_width, "sponge")
    fig_traj.update_layout(
        title=f"Combined PWH-AG trajectory: {filename.name}<br><sup>{subtitle}</sup>",
        xaxis_title="x [m]", yaxis_title="y [m]", template="plotly_white",
        width=1100, height=820, showlegend=False,
    )
    fig_traj.update_yaxes(scaleanchor="x", scaleratio=1)

    fig_speed = make_pair_figure("Total horizontal speed", drop_speed, vB_speed, "speed [m/s]")
    fig_radial = make_pair_figure("Radial velocity component", drop_vr, vB_r, "radial velocity [m/s]")
    fig_azimuthal = make_pair_figure("Azimuthal velocity component", drop_vtheta, vB_theta, "azimuthal velocity [m/s]")

    fig_difference = go.Figure()
    fig_difference.add_trace(go.Scatter(x=time_history, y=delta_vr, mode="lines", name="radial difference", line=dict(width=2)))
    fig_difference.add_trace(go.Scatter(x=time_history, y=delta_vtheta, mode="lines", name="azimuthal difference", line=dict(width=2)))
    fig_difference.add_trace(go.Scatter(x=time_history, y=delta_vtotal, mode="lines", name="total relative speed", line=dict(width=2, dash="dot")))
    style_time_figure(fig_difference, "Droplet velocity relative to the local background flow", "velocity difference [m/s]")

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Combined PWH-AG results</title>",
        "<style>"
        "body{font-family:Arial,sans-serif;margin:0;padding:22px;background:white;}"
        "h1{text-align:center;margin-bottom:8px;}"
        ".intro{max-width:1200px;margin:0 auto 18px auto;text-align:center;color:#333;}"
        ".plot-full{display:flex;justify-content:center;margin-bottom:28px;}"
        ".plot-grid{display:grid;grid-template-columns:repeat(2,minmax(720px,1fr));gap:24px;justify-items:center;align-items:start;max-width:1500px;margin:0 auto;}"
        ".plot-cell{display:flex;justify-content:center;}"
        "@media (max-width: 1500px){.plot-grid{grid-template-columns:1fr;max-width:760px;}}"
        "</style>",
        "</head><body>",
        f"<h1>Combined PWH-AG results: {filename.name}</h1>",
        "<p class='intro'>The trajectory is shown first. The velocity diagnostics and the droplet-to-background velocity-difference plot are arranged below in two columns for easier inspection of the vertical axis.<br>"
        f"Background-flow parameters: A = {A:.6g} m^2/s, B = {B:.6g} m^2/s, B/A = {B_over_A:.6g}.</p>",
        "<div class='plot-full'>", pio.to_html(fig_traj, include_plotlyjs='cdn', full_html=False), "</div>",
        "<div class='plot-grid'>",
        "<div class='plot-cell'>", pio.to_html(fig_speed, include_plotlyjs=False, full_html=False), "</div>",
        "<div class='plot-cell'>", pio.to_html(fig_radial, include_plotlyjs=False, full_html=False), "</div>",
        "<div class='plot-cell'>", pio.to_html(fig_azimuthal, include_plotlyjs=False, full_html=False), "</div>",
        "<div class='plot-cell'>", pio.to_html(fig_difference, include_plotlyjs=False, full_html=False), "</div>",
        "</div>",
        "</body></html>",
    ]
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Saved combined results HTML to: {html_path}")


if save_trajectory_html:
    save_interactive_trajectory()

if save_velocity_html:
    save_interactive_velocities()

if save_results_html:
    save_combined_results_html()


if fig is not None:
    plt.show()
