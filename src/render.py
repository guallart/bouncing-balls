import subprocess

import matplotlib
import numpy as np

BACKGROUND = "#0b0e14"
PANEL = "#10141c"
FOREGROUND = "#c9d1d9"
MUTED = "#5c6773"
GRID = "#1c2330"
GROUND = "#e6edf3"
GROUND_FILL = "#1a2233"

WIDTH_IN, HEIGHT_IN, DPI = 16, 9, 120


def apply_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "text.color": FOREGROUND,
        "axes.labelcolor": FOREGROUND,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })


def h264_codec():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    return "libx264" if "libx264" in out else "libopenh264"


def pi_formatter(value, _):
    k = round(value / (np.pi / 2))
    if abs(value - k * np.pi / 2) > 1e-6 or k == 0:
        return "$0$" if abs(value) < 1e-9 else f"${value:g}$"
    sign = "-" if k < 0 else ""
    k = abs(k)
    if k == 1:
        return rf"${sign}\pi/2$"
    if k == 2:
        return rf"${sign}\pi$"
    if k % 2 == 0:
        return rf"${sign}{k // 2}\pi$"
    return rf"${sign}{k}\pi/2$"


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def build_figure(ground_x, ground_y, bounds, title):
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    x_min, x_max, y_min, y_max = bounds

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.grid(True)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelsize=13, pad=8)

    half_pi_units = [v / (np.pi / 2) for v in (x_min, x_max)]
    if all(abs(u - round(u)) < 1e-6 for u in half_pi_units):
        ax.xaxis.set_major_locator(MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18, rotation=0, labelpad=14)

    ax.fill_between(ground_x, ground_y, y_min, color=GROUND_FILL, zorder=1)
    ax.plot(ground_x, ground_y, color=GROUND, lw=6, alpha=0.08, zorder=2)
    ax.plot(ground_x, ground_y, color=GROUND, lw=2, zorder=3)

    if title:
        fig.text(0.07, 0.94, title, fontsize=26, ha="left", va="center")
    return fig, ax


def render(xs, ys, ground_x, ground_y, bounds, fps, output, point_size, title):
    if output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    n_frames, n_balls = xs.shape
    colors = matplotlib.colormaps["rainbow"](np.linspace(0.0, 1.0, n_balls))

    fig, ax = build_figure(ground_x, ground_y, bounds, title)
    scat = ax.scatter(xs[0], ys[0], s=point_size, c=colors,
                      linewidths=0, alpha=0.95, zorder=4)
    label = ax.text(1.0, 1.04, "", transform=ax.transAxes, fontsize=20,
                    ha="right", va="bottom", color=MUTED)

    def update(i):
        scat.set_offsets(np.column_stack((xs[i], ys[i])))
        label.set_text(rf"$t = {i / fps:.2f}\,\mathrm{{s}}$")
        return scat, label

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=False)
    if output:
        writer = FFMpegWriter(
            fps=fps,
            codec=h264_codec(),
            bitrate=16000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        anim.save(output, writer=writer)
    else:
        plt.show()
    plt.close(fig)
