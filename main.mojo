from std.math import round
from src.animator import Animator
from src.gpu_sim import GpuSimulation

from std import math

comptime BALL_COUNT = 100000
comptime FINAL_TIME = 30.0
comptime DT = 0.01
comptime FPS = 25
comptime STEPS_PER_FRAME = Int(round(1.0 / (Float64(FPS) * DT)))

comptime X_MIN = -math.pi * 1.5
comptime X_MAX = math.pi * 0.5
comptime Y_MIN = -1.1
comptime Y_MAX = 2.0


def ground[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return math.sin(x)


def main() raises:
    var x0 = List[Float64](capacity=BALL_COUNT)
    var y0 = List[Float64](capacity=BALL_COUNT)
    for i in range(BALL_COUNT):
        var t = Float64(i) / Float64(BALL_COUNT - 1)
        x0.append(t * 2 - 1)
        y0.append(1.0)

    var sim = GpuSimulation(x0, y0)
    var anim = Animator(BALL_COUNT, FPS, X_MIN, X_MAX, Y_MIN, Y_MAX)

    var frames = Int(FINAL_TIME * Float64(FPS))
    sim.run[ground[DType.float32]](anim, frames, STEPS_PER_FRAME, DT)

    # anim.render[ground[DType.float64]]("balls.mp4")
    anim.render[ground[DType.float64]]()
