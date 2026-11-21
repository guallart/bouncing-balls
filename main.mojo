from std.math import round
from src.vec2 import Vec2
from src.ball import Ball
from src.animator import Animator

from std import math

comptime BALL_COUNT = 10000
comptime FINAL_TIME = 30.0
comptime DT = 0.01
comptime FPS = 25
comptime STEPS_PER_FRAME = Int(round(1.0 / (Float64(FPS) * DT)))

comptime X_MIN = -math.pi * 1.5
comptime X_MAX = math.pi * 0.5
comptime Y_MIN = -1.1
comptime Y_MAX = 2.0


def ground(x: Float64) -> Float64:
    return math.sin(x)


def main() raises:
    var balls = List[Ball](capacity=BALL_COUNT)
    for i in range(BALL_COUNT):
        var t = Float64(i) / Float64(BALL_COUNT - 1)
        var pos = Vec2(t * 2 - 1, 1)
        var vel = Vec2(0, 0)
        balls.append(Ball(pos^, vel^))

    var anim = Animator(BALL_COUNT, FPS, X_MIN, X_MAX, Y_MIN, Y_MAX)

    var time = 0.0
    var step = 0
    while time < FINAL_TIME:
        if step % STEPS_PER_FRAME == 0:
            anim.capture(balls)
        for ref ball in balls:
            ball.update[ground](DT)
        time += DT
        step += 1

    # anim.render[ground]("balls.mp4")
    anim.render[ground]()
