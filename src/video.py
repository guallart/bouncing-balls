import ctypes
import os
import subprocess

import matplotlib
import numpy as np

matplotlib.use("Agg")

import render as style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

LABEL_WIDTH, LABEL_HEIGHT = 380, 64


class Scene:
    def __init__(self, ground_x, ground_y, bounds, title, note="", cmap="rainbow"):
        fig, ax = style.build_figure(ground_x, ground_y, bounds, title, note)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        self.height, self.width = rgba.shape[:2]
        self.bg = np.ascontiguousarray(rgba[..., :3])

        origin, unit = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
        self.affine = (
            float(unit[0] - origin[0]),
            float(origin[0]),
            -float(unit[1] - origin[1]),
            float(self.height - origin[1]),
        )
        left, bottom, right, top = ax.get_window_extent().extents
        self.clip = (float(left), float(right),
                     float(self.height - top), float(self.height - bottom))

        anchor = ax.transAxes.transform((1.0, 1.04))
        self.label_x = int(round(anchor[0])) - LABEL_WIDTH
        self.label_y = int(round(self.height - anchor[1])) - LABEL_HEIGHT

        self.lut = np.ascontiguousarray(
            matplotlib.colormaps[cmap](np.linspace(0.0, 1.0, 256))[:, :3],
            dtype=np.float32,
        )
        style.plt_close(fig)

        dpi = style.DPI
        self._label_fig = Figure(figsize=(LABEL_WIDTH / dpi, LABEL_HEIGHT / dpi),
                                 dpi=dpi)
        self._label_fig.patch.set_alpha(0.0)
        self._label_canvas = FigureCanvasAgg(self._label_fig)
        self._label = self._label_fig.text(
            0.99, 0.08, "", fontsize=20, ha="right", va="bottom",
            color=style.MUTED)

    def paste_label(self, frame, seconds):
        self._label.set_text(rf"$t = {seconds:.2f}\,\mathrm{{s}}$")
        self._label_canvas.draw()
        overlay = np.asarray(self._label_canvas.buffer_rgba())
        alpha = overlay[..., 3:4].astype(np.float32) / 255.0
        y0, x0 = self.label_y, self.label_x
        region = frame[y0:y0 + LABEL_HEIGHT, x0:x0 + LABEL_WIDTH]
        blended = region * (1.0 - alpha) + overlay[..., :3] * alpha
        region[:] = blended.astype(np.uint8)


class VideoWriter:
    def __init__(self, scene, output, fps):
        self.scene = scene
        self.fps = fps
        self.nbytes = scene.width * scene.height * 3
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        if str(output).lower().endswith(".webm"):
            encode = ["-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0",
                      "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
                      "-pix_fmt", "yuv420p"]
        else:
            codec = style.h264_codec()
            quality = (["-crf", "16", "-preset", "medium"]
                       if codec == "libx264" else ["-b:v", "16M"])
            encode = ["-c:v", codec, *quality, "-pix_fmt", "yuv420p",
                      "-movflags", "+faststart"]
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{scene.width}x{scene.height}", "-r", str(fps), "-i", "-",
             "-an", *encode, output],
            stdin=subprocess.PIPE)

    def write(self, address, index):
        raw = (ctypes.c_ubyte * self.nbytes).from_address(address)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.scene.height, self.scene.width, 3).copy()
        self.scene.paste_label(frame, index / self.fps)
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
