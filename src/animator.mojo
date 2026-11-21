from std.python import Python, PythonObject
from .ball import Ball, FloatFn


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

    def render[ground: FloatFn](self, output: String = "", point_size: Float64 = 4.0) raises:
        var np = Python.import_module("numpy")
        var xs = self._to_numpy(np, self.xs, self.n_frames, self.n_balls)
        var ys = self._to_numpy(np, self.ys, self.n_frames, self.n_balls)

        var n_ground = 400
        var gx = List[Float64]()
        var gy = List[Float64]()
        for i in range(n_ground):
            var x = self.x_min + (self.x_max - self.x_min) * Float64(i) / Float64(n_ground - 1)
            gx.append(x)
            gy.append(ground(x))
        var gxn = self._to_numpy(np, gx, 1, n_ground).reshape(n_ground)
        var gyn = self._to_numpy(np, gy, 1, n_ground).reshape(n_ground)

        var bounds = Python.tuple(self.x_min, self.x_max, self.y_min, self.y_max)

        Python.add_to_path("src")
        var render = Python.import_module("render")
        render.render(xs, ys, gxn, gyn, bounds, self.fps, output, point_size)

    @staticmethod
    def _to_numpy(np: PythonObject, data: List[Float64], rows: Int, cols: Int) raises -> PythonObject:
        var ctypes = Python.import_module("ctypes")
        var address = Int(data.unsafe_ptr())
        var c_ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_double))
        var view = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(rows * cols))
        return np.array(view, copy=True).reshape(Python.tuple(rows, cols))
