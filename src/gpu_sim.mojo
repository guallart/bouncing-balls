from std.math import sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from .animator import Animator
from .ball import GRAVITY, RESTITUTION, MAX_BOUNCES

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
