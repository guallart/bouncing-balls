@fieldwise_init
struct Vec2(ImplicitlyCopyable, Movable):
    var x: Float64
    var y: Float64

    def __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y)

    def __iadd__(mut self, other: Self):
        self.x += other.x
        self.y += other.y

    def __isub__(mut self, other: Self):
        self.x -= other.x
        self.y -= other.y

    def __mul__(self, k: Float64) -> Self:
        return Self(self.x * k, self.y * k)

    def __rmul__(self, k: Float64) -> Self:
        return Self(self.x * k, self.y * k)
