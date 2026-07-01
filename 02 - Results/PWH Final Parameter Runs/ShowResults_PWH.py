"""
Visualize saved PWH simulation output.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. User settings
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent

filename_name = "PWH_GPU_30e-3.npz"  # change this name when needed

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


# Display / animation settings.
interval_ms = 1
color_percentile = 99.6
color_scale_multiplier = 1.5
surface_cmap = "Blues"

# The simulation code stores this many frames per Faraday period.
stored_frames_per_faraday_period = 12

# Choose how many of the stored frames per Faraday period are actually animated.
animated_frames_per_faraday_period = 12

# Spatial downsampling only for display.
# Use 1 for full resolution, 2 for every second grid point, 4 for every fourth grid point, etc.
display_stride = 1

# Trajectory downsampling. Draw every nth point.
trajectory_stride = 10


# ============================================================
# 2. Load data file
# ============================================================

print(f"Loading: {filename}")

data = np.load(filename)

x = data["x"]
y = data["y"]
frames = data["frames"]          # stored eta frames, already dimensional [m]
times = data["times"]            # times corresponding to frames

time_history = data["time_history"]
drop_x = data["drop_x_history"]
drop_y = data["drop_y_history"]
drop_z = data["drop_z_history"]
drop_speed = data["drop_speed_history"]
contact = data["contact_history"]

# Sponge-layer display information.
# The simulation damps waves in a square layer near the edge of the domain.
# This code draws the inner boundary of that sponge layer as a dotted square.
L_phys_saved = data["L_phys"].item() if "L_phys" in data.files else float(np.max(np.abs(x)))
length_scale_saved = data["length_scale"].item() if "length_scale" in data.files else None

if "sponge_width" in data.files and length_scale_saved is not None:
    # sponge_width is saved in nondimensional units in the simulation output.
    sponge_width_phys = data["sponge_width"].item() * length_scale_saved
else:
    # Fallback for older files: simulation default was sponge_width_fraction = 0.22.
    sponge_width_phys = 0.22 * L_phys_saved

sponge_inner_half_width = L_phys_saved - sponge_width_phys
show_sponge_boundary = True


# ============================================================
# 3. Print simulation parameters stored in the .npz file
# ============================================================

def _scalar(npz_data, key, default=None):
    """
    Safely read a scalar value from the loaded .npz file.
    Returns default if the key is missing.
    """
    if key not in npz_data.files:
        return default
    value = npz_data[key]
    if np.ndim(value) == 0:
        return value.item()
    if np.size(value) == 1:
        return value.reshape(-1)[0].item()
    return value


def _fmt(value, fmt=".5e"):
    """
    Format a value, while keeping missing values readable.
    """
    if value is None:
        return "not saved in file"
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return str(value)


def print_simulation_parameters(npz_data):
    """
    Print the main parameter block stored inside the saved .npz output file.
    """
    epsilon_nd = _scalar(npz_data, "epsilon_nd")
    Bo_nd = _scalar(npz_data, "Bo_nd")
    G_nd = _scalar(npz_data, "G_nd")
    Omega = _scalar(npz_data, "Omega")

    c1 = _scalar(npz_data, "c1")
    c2 = _scalar(npz_data, "c2")
    c3 = _scalar(npz_data, "c3")
    c4 = _scalar(npz_data, "c4")

    Gamma = _scalar(npz_data, "Gamma")
    GammaF = _scalar(npz_data, "GammaF")
    gamma_peak = _scalar(npz_data, "gamma_peak")

    length_scale = _scalar(npz_data, "length_scale")
    time_scale = _scalar(npz_data, "time_scale")
    mass_scale = _scalar(npz_data, "mass_scale")
    kF = _scalar(npz_data, "kF")

    L_phys = _scalar(npz_data, "L_phys")
    L_nd = _scalar(npz_data, "L_nd")
    N_saved = _scalar(npz_data, "N")

    dt_phys = _scalar(npz_data, "dt")
    dt_nd = _scalar(npz_data, "dt_nd")

    x_phys = npz_data["x"]
    x_nd = npz_data["x_nd"] if "x_nd" in npz_data.files else None
    dx_phys = x_phys[1] - x_phys[0] if len(x_phys) > 1 else None
    dx_nd = x_nd[1] - x_nd[0] if x_nd is not None and len(x_nd) > 1 else None

    time_history_local = npz_data["time_history"]
    n_steps = len(time_history_local) - 1

    times_local = npz_data["times"]
    if len(times_local) > 1 and dt_phys not in (None, 0):
        saved_stride = int(round((times_local[1] - times_local[0]) / dt_phys))
    else:
        saved_stride = None

    dt_direct_wave_estimate_nd = None
    dt_direct_wave_estimate_phys = None
    if x_nd is not None and all(v is not None for v in (N_saved, G_nd, Bo_nd, time_scale, dx_nd)):
        kx = 2.0 * np.pi * np.fft.fftfreq(int(N_saved), d=dx_nd)
        ky = 2.0 * np.pi * np.fft.fftfreq(int(N_saved), d=dx_nd)
        KX, KY = np.meshgrid(kx, ky)
        k_abs = np.sqrt(KX**2 + KY**2)
        omega_max = np.max(np.sqrt(np.maximum(k_abs * (G_nd + Bo_nd * k_abs**2), 0.0)))
        dt_direct_wave_estimate_nd = 0.18 / max(omega_max, 1.0)
        dt_direct_wave_estimate_phys = dt_direct_wave_estimate_nd * time_scale

    print()
    print("Simulation parameters")
    print("---------------------")
    print(
        "Wave parameters: "
        f"epsilon = {_fmt(epsilon_nd)}, "
        f"Bo = {_fmt(Bo_nd)}, "
        f"G = {_fmt(G_nd)}, "
        f"Omega = {_fmt(Omega)}"
    )
    print(f"Impact constants: c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}")

    if Gamma is not None and GammaF not in (None, 0):
        print(f"Gamma/GammaF = {Gamma / GammaF:.5f}")
    else:
        print("Gamma/GammaF = not saved in file")

    print(f"Gamma = {_fmt(Gamma)}, gamma_peak = {_fmt(gamma_peak)} m/s^2")
    print(f"Length scale: l_c = {_fmt(length_scale)} m")
    print(f"Time scale: T0 = {_fmt(time_scale)} s")
    print(f"Mass scale: M0 = {_fmt(mass_scale)} kg")
    print(f"Faraday wavenumber: kF = {_fmt(kF)} 1/m")

    print(
        "Domain half-width: "
        f"L = {_fmt(L_phys)} m = {_fmt(L_nd)} nondimensional units"
    )

    print(
        "Grid: "
        f"N = {N_saved}, "
        f"dx = {_fmt(dx_phys)} m = {_fmt(dx_nd)} nondimensional units"
    )

    print(f"Number of RK4 steps: {n_steps}")
    print(
        "Integrating-factor dt = "
        f"{_fmt(dt_phys)} s = {_fmt(dt_nd)} bath periods"
    )

    print(
        "Direct explicit wave dt estimate = "
        f"{_fmt(dt_direct_wave_estimate_phys)} s = "
        f"{_fmt(dt_direct_wave_estimate_nd)} bath periods"
    )

    if saved_stride is not None:
        print(f"Saved-frame stride: one frame every {saved_stride} RK4 steps")
    else:
        print("Saved-frame stride: not available")

    print()


print_simulation_parameters(data)


# ============================================================
# 4. Prepare displayed frames
# ============================================================

# Validate animation-frame choice.
animated_frames_per_faraday_period = int(animated_frames_per_faraday_period)
if animated_frames_per_faraday_period < 1:
    raise ValueError("animated_frames_per_faraday_period must be at least 1.")
if animated_frames_per_faraday_period > stored_frames_per_faraday_period:
    raise ValueError("animated_frames_per_faraday_period cannot exceed stored_frames_per_faraday_period.")

frames_array = np.asarray(frames)
times_array = np.asarray(times)

# Time-downsample the saved frames for animation.
# This keeps approximately animated_frames_per_faraday_period out of
# stored_frames_per_faraday_period saved frames each Faraday period.
frame_interval = stored_frames_per_faraday_period / animated_frames_per_faraday_period
animation_indices = np.unique(
    np.round(np.arange(0, len(times_array), frame_interval)).astype(int)
)
animation_indices = animation_indices[animation_indices < len(times_array)]

frames_anim = frames_array[animation_indices]
times_anim = times_array[animation_indices]

# Spatial downsampling for display only.
if display_stride > 1:
    frames_anim_display = frames_anim[:, ::display_stride, ::display_stride]
    x_display = x[::display_stride]
    y_display = y[::display_stride]
else:
    frames_anim_display = frames_anim
    x_display = x
    y_display = y

extent = [x_display[0], x_display[-1], y_display[0], y_display[-1]]

eta_scale = np.percentile(np.abs(frames_anim_display), color_percentile)
vmax = max(color_scale_multiplier * eta_scale, 1.0e-12)
vmin = -vmax

print(f"Saved frames: {len(times_array)}")
print(f"Animated frames: {len(times_anim)}")
print(
    "Animated frames per Faraday period: "
    f"{animated_frames_per_faraday_period}/{stored_frames_per_faraday_period}"
)
print(f"Display stride: {display_stride}")


# ============================================================
# 5. Combined diagnostics + animation figure
# ============================================================

fig = plt.figure(figsize=(14, 8))
fig.suptitle(f"PWH output: {filename.name}", fontsize=12)

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[1.10, 1.35],
    height_ratios=[1.0, 1.0, 1.0],
    wspace=0.30,
    hspace=0.25,
)

ax_z = fig.add_subplot(gs[0, 0])
ax_speed = fig.add_subplot(gs[1, 0], sharex=ax_z)
ax_contact = fig.add_subplot(gs[2, 0], sharex=ax_z)
ax_wave = fig.add_subplot(gs[:, 1])

# Diagnostics on the left.
ax_z.plot(time_history, drop_z)
ax_z.set_ylabel("z base [m]")
ax_z.set_title("Droplet diagnostics")

ax_speed.plot(time_history, drop_speed)
ax_speed.set_ylabel(r"in-plane speed [m/s]")

ax_contact.plot(time_history, contact)
ax_contact.set_ylabel("contact")
ax_contact.set_xlabel("time [s]")

# Moving vertical time markers on diagnostic plots.
time_marker_z = ax_z.axvline(times_anim[0], linestyle="--", linewidth=1.0)
time_marker_speed = ax_speed.axvline(times_anim[0], linestyle="--", linewidth=1.0)
time_marker_contact = ax_contact.axvline(times_anim[0], linestyle="--", linewidth=1.0)

# Animation on the right.
im = ax_wave.imshow(
    frames_anim_display[0],
    extent=extent,
    origin="lower",
    cmap=surface_cmap,
    vmin=vmin,
    vmax=vmax,
    interpolation="bilinear",
)

droplet_marker, = ax_wave.plot([], [], "ko", markersize=5, label="droplet")
trajectory_line, = ax_wave.plot([], [], "k-", linewidth=1.0, alpha=0.7, label="trajectory")

# Dotted square marking the inner boundary of the sponge layer.
# Outside this square, the numerical sponge damping is active.
if show_sponge_boundary and sponge_inner_half_width > 0.0:
    sponge_x = [
        -sponge_inner_half_width,
        sponge_inner_half_width,
        sponge_inner_half_width,
        -sponge_inner_half_width,
        -sponge_inner_half_width,
    ]
    sponge_y = [
        -sponge_inner_half_width,
        -sponge_inner_half_width,
        sponge_inner_half_width,
        sponge_inner_half_width,
        -sponge_inner_half_width,
    ]
    sponge_boundary, = ax_wave.plot(
        sponge_x,
        sponge_y,
        linestyle=":",
        color="k",
        linewidth=1.2,
        alpha=0.9,
        label="sponge boundary",
    )

cbar = fig.colorbar(im, ax=ax_wave)
cbar.set_label(r"$\eta(x,y,t)$ [m]")

ax_wave.set_xlabel("x [m]")
ax_wave.set_ylabel("y [m]")
ax_wave.set_aspect("equal")
ax_wave.legend(loc="upper right")
title = ax_wave.set_title("")


def update(frame_index):
    t_frame = times_anim[frame_index]

    # Match animation-frame time to closest full diagnostic index.
    hist_index = int(np.argmin(np.abs(time_history - t_frame)))

    im.set_data(frames_anim_display[frame_index])
    droplet_marker.set_data([drop_x[hist_index]], [drop_y[hist_index]])
    trajectory_line.set_data(
        drop_x[:hist_index + 1:trajectory_stride],
        drop_y[:hist_index + 1:trajectory_stride],
    )

    for marker in (time_marker_z, time_marker_speed, time_marker_contact):
        marker.set_xdata([t_frame, t_frame])

    title.set_text(f"Surface elevation, t = {t_frame:.4f} s")

    return (
        im,
        droplet_marker,
        trajectory_line,
        time_marker_z,
        time_marker_speed,
        time_marker_contact,
        title,
    )


ani = FuncAnimation(
    fig,
    update,
    frames=len(times_anim),
    interval=interval_ms,
    blit=False,
)

plt.show()
