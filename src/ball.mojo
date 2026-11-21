from std.math import sqrt
from .vec2 import Vec2

comptime FloatFn = def(Float64) thin -> Float64

comptime GRAVITY = Vec2(0.0, -1.0)
comptime RESTITUTION = 0.9
comptime EPS = 1e-5
comptime MAX_BOUNCES = 8
comptime BISECTION_ITERS = 50


struct Ball(Copyable, Movable):
    var pos: Vec2
    var vel: Vec2

    def __init__(out self, var pos: Vec2, var vel: Vec2):
        self.pos = pos^
        self.vel = vel^

    def _reflect[ground: FloatFn](mut self):
        var m = (ground(self.pos.x + EPS) - ground(self.pos.x - EPS)) / (2 * EPS)
        var inv_len = 1.0 / sqrt(1.0 + m * m)
        var nx = -m * inv_len
        var ny = inv_len

        var vn = self.vel.x * nx + self.vel.y * ny
        if vn < 0:
            self.vel.x -= (1 + RESTITUTION) * vn * nx
            self.vel.y -= (1 + RESTITUTION) * vn * ny

    def _advance(self, t: Float64) -> Vec2:
        return self.pos + self.vel * t + GRAVITY * (0.5 * t * t)

    def update[ground: FloatFn](mut self, dt: Float64):
        if self.pos.y < ground(self.pos.x):
            self.pos.y = ground(self.pos.x)
            self._reflect[ground]()

        var remaining = dt
        for _ in range(MAX_BOUNCES):
            var end = self._advance(remaining)
            if end.y >= ground(end.x):
                self.pos = end^
                self.vel += GRAVITY * remaining
                return

            var lo = 0.0
            var hi = remaining
            for _ in range(BISECTION_ITERS):
                var mid = 0.5 * (lo + hi)
                var p = self._advance(mid)
                if p.y >= ground(p.x):
                    lo = mid
                else:
                    hi = mid

            self.pos = self._advance(lo)
            self.vel += GRAVITY * lo
            self._reflect[ground]()
            remaining -= lo

        self.pos = self._advance(remaining)
        self.vel += GRAVITY * remaining
        var surface = ground(self.pos.x)
        if self.pos.y < surface:
            self.pos.y = surface
            self._reflect[ground]()
