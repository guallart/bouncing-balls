from std.math import sqrt
from .vec2 import Vec2

comptime FloatFn = def(Float64) thin -> Float64

comptime GRAVITY = Vec2(0.0, -1.0)
comptime RESTITUTION = 1.0
comptime EPS = 1e-5


struct Ball(Copyable, Movable):
    var pos: Vec2
    var vel: Vec2

    def __init__(out self, var pos: Vec2, var vel: Vec2):
        self.pos = pos^
        self.vel = vel^

    def update[ground: FloatFn](mut self, dt: Float64):
        self.vel += GRAVITY * dt
        self.pos += self.vel * dt

        var surface = ground(self.pos.x)
        if self.pos.y < surface:
            var m = (ground(self.pos.x + EPS) - ground(self.pos.x - EPS)) / (2 * EPS)
            var inv_len = 1.0 / sqrt(1.0 + m * m)
            var nx = -m * inv_len
            var ny = inv_len

            self.pos.y = surface

            var vn = self.vel.x * nx + self.vel.y * ny
            if vn < 0:
                self.vel.x -= (1 + RESTITUTION) * vn * nx
                self.vel.y -= (1 + RESTITUTION) * vn * ny
