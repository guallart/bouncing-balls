from std.math import sqrt
from std.python import Python
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from .animator import Animator
from .pyutil import ground_arrays
from .ball import GRAVITY, RESTITUTION, MAX_BOUNCES, FloatFn

comptime FP = DType.float32
comptime Real = Scalar[FP]
comptime GpuFn = def(Real) thin -> Real

comptime BLOCK_SIZE = 256
comptime GX = Real(GRAVITY.x)
comptime GY = Real(GRAVITY.y)
comptime REST = Real(RESTITUTION)
comptime EPS = Real(1e-3)
comptime BISECTION_ITERS = 24


def _reflect[ground: GpuFn](x: Real, mut vx: Real, mut vy: Real):
    var m = (ground(x + EPS) - ground(x - EPS)) / (2 * EPS)
    var inv_len = 1.0 / sqrt(1.0 + m * m)
    var nx = -m * inv_len
    var ny = inv_len

    var vn = vx * nx + vy * ny
    if vn < 0:
        vx -= (1 + REST) * vn * nx
        vy -= (1 + REST) * vn * ny


def _step_ball[
    ground: GpuFn
](mut x: Real, mut y: Real, mut vx: Real, mut vy: Real, dt: Real):
    var gx = GX
    var gy = GY

    var surface = ground(x)
    if y < surface:
        y = surface
        _reflect[ground](x, vx, vy)

    var remaining = dt
    for _ in range(MAX_BOUNCES):
        var ex = x + vx * remaining + gx * (0.5 * remaining * remaining)
        var ey = y + vy * remaining + gy * (0.5 * remaining * remaining)
        if ey >= ground(ex):
            x = ex
            y = ey
            vx += gx * remaining
            vy += gy * remaining
            return

        var lo: Real = 0.0
        var hi = remaining
        for _ in range(BISECTION_ITERS):
            var mid = 0.5 * (lo + hi)
            var px = x + vx * mid + gx * (0.5 * mid * mid)
            var py = y + vy * mid + gy * (0.5 * mid * mid)
            if py >= ground(px):
                lo = mid
            else:
                hi = mid

        x = x + vx * lo + gx * (0.5 * lo * lo)
        y = y + vy * lo + gy * (0.5 * lo * lo)
        vx += gx * lo
        vy += gy * lo
        _reflect[ground](x, vx, vy)
        remaining -= lo

    x = x + vx * remaining + gx * (0.5 * remaining * remaining)
    y = y + vy * remaining + gy * (0.5 * remaining * remaining)
    vx += gx * remaining
    vy += gy * remaining
    surface = ground(x)
    if y < surface:
        y = surface
        _reflect[ground](x, vx, vy)


def _update_kernel[
    ground: GpuFn
](
    x: Pointer[Real, MutAnyOrigin],
    y: Pointer[Real, MutAnyOrigin],
    vx: Pointer[Real, MutAnyOrigin],
    vy: Pointer[Real, MutAnyOrigin],
    n: Int32,
    steps: Int32,
    dt: Real,
):
    var i = Int(global_idx.x)
    if i >= Int(n):
        return

    var px = x[unsafe_offset=i]
    var py = y[unsafe_offset=i]
    var pvx = vx[unsafe_offset=i]
    var pvy = vy[unsafe_offset=i]

    for _ in range(Int(steps)):
        _step_ball[ground](px, py, pvx, pvy, dt)

    x[unsafe_offset=i] = px
    y[unsafe_offset=i] = py
    vx[unsafe_offset=i] = pvx
    vy[unsafe_offset=i] = pvy


def _splat_kernel(
    x: Pointer[Real, MutAnyOrigin],
    y: Pointer[Real, MutAnyOrigin],
    lut: Pointer[Real, MutAnyOrigin],
    bg: Pointer[UInt8, MutAnyOrigin],
    frame: Pointer[UInt8, MutAnyOrigin],
    n: Int32,
    width: Int32,
    height: Int32,
    ax: Real,
    bx: Real,
    ay: Real,
    by: Real,
    clip_l: Real,
    clip_r: Real,
    clip_t: Real,
    clip_b: Real,
    radius: Real,
    alpha: Real,
):
    var i = Int(global_idx.x)
    if i >= Int(n):
        return

    var cx = x[unsafe_offset=i] * ax + bx
    var cy = y[unsafe_offset=i] * ay + by
    if cx < clip_l or cx > clip_r or cy < clip_t or cy > clip_b:
        return

    var denom = Real(max(Int(n) - 1, 1))
    var li = Int(Real(i) / denom * 255.0 + 0.5)
    var cr = lut[unsafe_offset=3 * li] * 255.0
    var cg = lut[unsafe_offset=3 * li + 1] * 255.0
    var cb = lut[unsafe_offset=3 * li + 2] * 255.0

    var reach = Int(radius) + 1
    var ix = Int(cx)
    var iy = Int(cy)
    var w = Int(width)
    var h = Int(height)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            var px = ix + dx
            var py = iy + dy
            if px < 0 or px >= w or py < 0 or py >= h:
                continue
            var ox = Real(px) + 0.5 - cx
            var oy = Real(py) + 0.5 - cy
            if sqrt(ox * ox + oy * oy) > radius:
                continue
            var a = alpha
            var o = (py * w + px) * 3
            frame[unsafe_offset=o] = UInt8(
                Real(bg[unsafe_offset=o]) * (1.0 - a) + cr * a + 0.5
            )
            frame[unsafe_offset=o + 1] = UInt8(
                Real(bg[unsafe_offset=o + 1]) * (1.0 - a) + cg * a + 0.5
            )
            frame[unsafe_offset=o + 2] = UInt8(
                Real(bg[unsafe_offset=o + 2]) * (1.0 - a) + cb * a + 0.5
            )


struct GpuSimulation:
    var ctx: DeviceContext[]
    var n: Int
    var x: DeviceBuffer[FP]
    var y: DeviceBuffer[FP]
    var vx: DeviceBuffer[FP]
    var vy: DeviceBuffer[FP]
    var host_x: HostBuffer[FP]
    var host_y: HostBuffer[FP]

    def __init__(out self, x0: List[Float64], y0: List[Float64]) raises:
        self.ctx = DeviceContext()
        self.n = len(x0)
        self.x = self.ctx.enqueue_create_buffer[FP](self.n)
        self.y = self.ctx.enqueue_create_buffer[FP](self.n)
        self.vx = self.ctx.enqueue_create_buffer[FP](self.n)
        self.vy = self.ctx.enqueue_create_buffer[FP](self.n)
        self.host_x = self.ctx.enqueue_create_host_buffer[FP](self.n)
        self.host_y = self.ctx.enqueue_create_host_buffer[FP](self.n)

        for i in range(self.n):
            self.host_x[i] = Real(x0[i])
            self.host_y[i] = Real(y0[i])
        self.ctx.enqueue_copy(self.x, self.host_x)
        self.ctx.enqueue_copy(self.y, self.host_y)
        self.vx.enqueue_fill(0.0)
        self.vy.enqueue_fill(0.0)
        self.ctx.synchronize()

    def run[
        ground: GpuFn
    ](
        mut self,
        mut anim: Animator,
        frames: Int,
        steps_per_frame: Int,
        dt: Float64,
    ) raises:
        var kernel = self.ctx.compile_function[_update_kernel[ground]]()
        var blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE

        var xs = List[Float64](capacity=self.n)
        var ys = List[Float64](capacity=self.n)
        for _ in range(frames):
            self.ctx.enqueue_copy(self.host_x, self.x)
            self.ctx.enqueue_copy(self.host_y, self.y)
            self.ctx.synchronize()
            xs.clear()
            ys.clear()
            for i in range(self.n):
                xs.append(Float64(self.host_x[i]))
                ys.append(Float64(self.host_y[i]))
            anim.capture_xy(xs, ys)

            self.ctx.enqueue_function(
                kernel,
                self.x,
                self.y,
                self.vx,
                self.vy,
                Int32(self.n),
                Int32(steps_per_frame),
                Real(dt),
                grid_dim=blocks,
                block_dim=BLOCK_SIZE,
            )
        self.ctx.synchronize()

    def record[
        ground_gpu: GpuFn, ground_host: FloatFn
    ](
        mut self,
        output: String,
        title: String,
        x_min: Float64,
        x_max: Float64,
        y_min: Float64,
        y_max: Float64,
        fps: Int,
        frames: Int,
        steps_per_frame: Int,
        dt: Float64,
        point_radius: Float64 = 2.0,
        opacity: Float64 = 0.95,
    ) raises:
        var np = Python.import_module("numpy")
        var ctypes = Python.import_module("ctypes")
        Python.add_to_path("src")
        var video = Python.import_module("video")

        var ground_xy = ground_arrays[ground_host](x_min, x_max, 4000)
        var bounds = Python.tuple(x_min, x_max, y_min, y_max)
        var scene = video.Scene(ground_xy[0], ground_xy[1], bounds, title)

        var width = Int(py=scene.width)
        var height = Int(py=scene.height)
        var nbytes = width * height * 3
        var affine = scene.affine
        var clip = scene.clip

        var bg = self.ctx.enqueue_create_buffer[DType.uint8](nbytes)
        var frame = self.ctx.enqueue_create_buffer[DType.uint8](nbytes)
        var host_bg = self.ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        var host_frame = self.ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
        var lut = self.ctx.enqueue_create_buffer[FP](256 * 3)
        var host_lut = self.ctx.enqueue_create_host_buffer[FP](256 * 3)

        ctypes.memmove(Int(host_bg.unsafe_ptr()), Int(py=scene.bg.ctypes.data), nbytes)
        ctypes.memmove(Int(host_lut.unsafe_ptr()), Int(py=scene.lut.ctypes.data), 256 * 3 * 4)
        self.ctx.enqueue_copy(bg, host_bg)
        self.ctx.enqueue_copy(lut, host_lut)
        self.ctx.synchronize()

        var advance = self.ctx.compile_function[_update_kernel[ground_gpu]]()
        var splat = self.ctx.compile_function[_splat_kernel]()
        var blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE

        var writer = video.VideoWriter(scene, output, fps)
        for f in range(frames):
            self.ctx.enqueue_copy(frame, bg)
            self.ctx.enqueue_function(
                splat,
                self.x,
                self.y,
                lut,
                bg,
                frame,
                Int32(self.n),
                Int32(width),
                Int32(height),
                Real(Float64(py=affine[0])),
                Real(Float64(py=affine[1])),
                Real(Float64(py=affine[2])),
                Real(Float64(py=affine[3])),
                Real(Float64(py=clip[0])),
                Real(Float64(py=clip[1])),
                Real(Float64(py=clip[2])),
                Real(Float64(py=clip[3])),
                Real(point_radius),
                Real(opacity),
                grid_dim=blocks,
                block_dim=BLOCK_SIZE,
            )
            self.ctx.enqueue_copy(host_frame, frame)
            if f < frames - 1:
                self.ctx.enqueue_function(
                    advance,
                    self.x,
                    self.y,
                    self.vx,
                    self.vy,
                    Int32(self.n),
                    Int32(steps_per_frame),
                    Real(dt),
                    grid_dim=blocks,
                    block_dim=BLOCK_SIZE,
                )
            self.ctx.synchronize()
            writer.write(Int(host_frame.unsafe_ptr()), f)
        writer.close()
