from std.python import Python, PythonObject
from .ball import FloatFn


def to_numpy(
    np: PythonObject, data: List[Float64], rows: Int, cols: Int
) raises -> PythonObject:
    var ctypes = Python.import_module("ctypes")
    var address = Int(data.unsafe_ptr())
    var c_ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_double))
    var view = np.ctypeslib.as_array(c_ptr, shape=Python.tuple(rows * cols))
    return np.array(view, copy=True).reshape(Python.tuple(rows, cols))


def ground_arrays[
    ground: FloatFn
](x_min: Float64, x_max: Float64, n: Int) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var gx = List[Float64]()
    var gy = List[Float64]()
    for i in range(n):
        var x = x_min + (x_max - x_min) * Float64(i) / Float64(n - 1)
        gx.append(x)
        gy.append(ground(x))
    var gxn = to_numpy(np, gx, 1, n).reshape(n)
    var gyn = to_numpy(np, gy, 1, n).reshape(n)
    return Python.tuple(gxn, gyn)
