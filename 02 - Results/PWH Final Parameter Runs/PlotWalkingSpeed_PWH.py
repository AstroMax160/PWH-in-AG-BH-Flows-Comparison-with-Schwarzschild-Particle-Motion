"""
Interactive walking-speed plot for saved PWH simulation output.

"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# ============================================================
# 1. User settings
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent

filename_name = "PWH_GPU_PDE_Test.npz"  # change this name when needed

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

# Average only over the final part of the simulation.
# This defines the steady-state averaging window.
end_average_fraction = 0.30

save_html = True
html_name = filename.with_name(filename.stem + "_walking_speed.html")

# Downsample only for display if the data is very large.
# Use 1 to plot every point, 10 to plot every 10th point, etc.
display_stride = 1


# ============================================================
# 2. Load data
# ============================================================

data = np.load(filename)

time_history = np.asarray(data["time_history"])
drop_speed = np.asarray(data["drop_speed_history"])

# Optional extra fields, if available.
Gamma = float(data["Gamma"]) if "Gamma" in data.files else None
GammaF = float(data["GammaF"]) if "GammaF" in data.files else None
R0 = float(data["R0"]) if "R0" in data.files else None
N = int(data["N"]) if "N" in data.files else None

# Compute physical grid spacing from saved x array, if available.
dx = None
if "x" in data.files:
    x_values = np.asarray(data["x"])
    if len(x_values) > 1:
        dx = float(abs(x_values[1] - x_values[0]))

# Basic validation.
if len(time_history) != len(drop_speed):
    raise ValueError(
        f"time_history and drop_speed_history have different lengths: "
        f"{len(time_history)} vs {len(drop_speed)}"
    )
if not (0.0 < end_average_fraction <= 1.0):
    raise ValueError("end_average_fraction must be in the interval (0, 1].")

# Last-fraction average.
t_start = time_history[0]
t_end = time_history[-1]
t_cut = t_start + (1.0 - end_average_fraction) * (t_end - t_start)
end_mask = time_history >= t_cut

if not np.any(end_mask):
    raise ValueError("No data points found in the selected final averaging window.")

end_average_speed = float(np.mean(drop_speed[end_mask]))
end_std_speed = float(np.std(drop_speed[end_mask]))
end_min_speed = float(np.min(drop_speed[end_mask]))
end_max_speed = float(np.max(drop_speed[end_mask]))

# Downsample for display only.
time_plot = time_history[::display_stride]
speed_plot = drop_speed[::display_stride]


# ============================================================
# 3. Print diagnostics
# ============================================================

print("Walking-speed diagnostics")
print("-------------------------")
print(f"Data points: {len(time_history)}")
print(f"Displayed points: {len(time_plot)}")
print(f"Simulation duration: {t_end - t_start:.6g} s")
print(f"Final speed: {drop_speed[-1]:.6g} m/s")
print(f"Steady-state averaging window: t >= {t_cut:.6g} s")
print(f"Steady-state walking speed: {end_average_speed:.6g} m/s")
print(f"Steady-state speed std: {end_std_speed:.6g} m/s")
print(f"Steady-state speed range: {end_min_speed:.6g} to {end_max_speed:.6g} m/s")

if Gamma is not None and GammaF is not None:
    print(f"Gamma/GammaF: {Gamma / GammaF:.6g}")
if R0 is not None:
    print(f"R0: {R0:.6g} m")
if N is not None:
    print(f"Grid N: {N}")
if dx is not None:
    print(f"Grid dx: {dx:.6g} m")


# ============================================================
# 4. Interactive speed plot
# ============================================================

subtitle_parts = [f"steady-state walking speed = {end_average_speed:.4e} m/s"]
if Gamma is not None and GammaF is not None:
    subtitle_parts.append(f"Γ/ΓF = {Gamma / GammaF:.4f}")
if R0 is not None:
    subtitle_parts.append(f"R0 = {R0:.3e} m")
if N is not None:
    subtitle_parts.append(f"N = {N}")
if dx is not None:
    subtitle_parts.append(f"dx = {dx:.3e} m")

subtitle = " | ".join(subtitle_parts)
title = f"Walking speed: {filename.name}<br><sup>{subtitle}</sup>"

fig = go.Figure()

fig.add_trace(
    go.Scattergl(
        x=time_plot,
        y=speed_plot,
        mode="lines",
        name="walking speed",
        hovertemplate=(
            "time = %{x:.6f} s<br>"
            "speed = %{y:.6e} m/s"
            "<extra></extra>"
        ),
    )
)

# Horizontal line showing the stable/end average speed.
fig.add_trace(
    go.Scatter(
        x=[t_cut, t_end],
        y=[end_average_speed, end_average_speed],
        mode="lines",
        name="steady-state walking speed",
        line=dict(dash="dash"),
        hoverinfo="skip",
    )
)

# Shade the time interval used for the final average.
fig.add_vrect(
    x0=t_cut,
    x1=t_end,
    fillcolor="LightGray",
    opacity=0.25,
    layer="below",
    line_width=0,
    annotation_text="steady-state averaging window",
    annotation_position="top left",
)

layout_kwargs = dict(
    title=title,
    xaxis_title="time [s]",
    yaxis_title="in-plane speed [m/s]",
    hovermode="x unified",
    template="plotly_white",
)

fig.update_layout(**layout_kwargs)

fig.show()

if save_html:
    fig.write_html(html_name)
    print(f"Saved interactive HTML to: {html_name}")
