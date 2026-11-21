import matplotlib


def render(xs, ys, ground_x, ground_y, bounds, fps, output, point_size):
    if output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    x_min, x_max, y_min, y_max = bounds
    n_frames = xs.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.plot(ground_x, ground_y, color="black", lw=2)
    scat = ax.scatter(xs[0], ys[0], s=point_size)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(i):
        scat.set_offsets(list(zip(xs[i], ys[i])))
        label.set_text(f"t = {i / fps:.2f} s")
        return scat, label

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=True)
    if output:
        anim.save(output, fps=fps)
    else:
        plt.show()
    plt.close(fig)
