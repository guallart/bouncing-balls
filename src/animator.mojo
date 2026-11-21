from std.python import Python, PythonObject
from .ball import Ball, FloatFn
from .pyutil import to_numpy, ground_arrays


struct Animator:
    var n_balls: Int
    var n_frames: Int
    var fps: Int
    var x_min: Float64
    var x_max: Float64
    var y_min: Float64
    var y_max: Float64
    var xs: List[Float64]
    var ys: List[Float64]

    def __init__(
        out self,
        n_balls: Int,
        fps: Int,
        x_min: Float64,
        x_max: Float64,
        y_min: Float64,
        y_max: Float64,
    ):
        self.n_balls = n_balls
        self.n_frames = 0
        self.fps = fps
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.xs = List[Float64]()
        self.ys = List[Float64]()

    def capture(mut self, balls: List[Ball]):
        for ref ball in balls:
            self.xs.append(ball.pos.x)
            self.ys.append(ball.pos.y)
        self.n_frames += 1

    def capture_xy(mut self, x: List[Float64], y: List[Float64]):
        for i in range(len(x)):
            self.xs.append(x[i])
            self.ys.append(y[i])
        self.n_frames += 1

    def render[ground: FloatFn](
        self,
        output: String = "",
        title: String = "",
        point_size: Float64 = 6.0,
    ) raises:
        var np = Python.import_module("numpy")
        var xs = to_numpy(np, self.xs, self.n_frames, self.n_balls)
        var ys = to_numpy(np, self.ys, self.n_frames, self.n_balls)

        var ground_xy = ground_arrays[ground](self.x_min, self.x_max, 400)
        var gxn = ground_xy[0]
        var gyn = ground_xy[1]

        var bounds = Python.tuple(self.x_min, self.x_max, self.y_min, self.y_max)

        Python.add_to_path("src")
        var render = Python.import_module("render")
        render.render(
            xs, ys, gxn, gyn, bounds, self.fps, output, point_size, title
        )
