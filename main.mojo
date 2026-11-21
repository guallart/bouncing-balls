from std.math import round
from src.gpu_sim import GpuSimulation   

from std import math

comptime BALL_COUNT = 1_000_000
comptime FINAL_TIME = 30.0
comptime DT = 0.01
comptime FPS = 25
comptime STEPS_PER_FRAME = Int(round(1.0 / (Float64(FPS) * DT)))

comptime X_MIN = -1.62
comptime X_MAX = 1.62
comptime Y_MIN = -0.2
comptime Y_MAX = 1.3


def ground[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return 0.5 * x * x + 0.06 * math.sin(12 * x * x)


def main() raises:
    var x0 = List[Float64](capacity=BALL_COUNT)
    var y0 = List[Float64](capacity=BALL_COUNT)
    for i in range(BALL_COUNT):
        var t = Float64(i) / Float64(BALL_COUNT - 1)
        x0.append(t * 2 - 1)
        y0.append(1.0)

    var sim = GpuSimulation(x0, y0)
    var frames = Int(FINAL_TIME * Float64(FPS))
    sim.record[ground[DType.float32], ground[DType.float64]](
        "outputs/chirp_bowl_e09.webm",
        r"$y = \frac{x^2}{2} + 0.06\,\sin(12x^2)$",
        X_MIN,
        X_MAX,
        Y_MIN,
        Y_MAX,
        FPS,
        frames,
        STEPS_PER_FRAME,
        DT,
    )
