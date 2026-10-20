import typing
from math import factorial
from collections import OrderedDict
from functools import lru_cache
import numpy as np

Vector = typing.List[float]


def memoize(f):
    class MemoClass:
        def __init__(self, f):
            self.f = f
            self.cache = OrderedDict()
            self.maxsize = 32768
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_evictions = 0

        def __call__(self, x, knots, degree, endpoint=False, r=0, knotspan_norm=None):
            if not isinstance(r, (int, np.integer)) or r < 0:
                raise ValueError("Derivative order must be a nonnegative integer.")
            knots_arr = np.asarray(knots)
            key = (float(x), int(degree), bool(endpoint), int(r), tuple(knots_arr.tolist()),
                   None if knotspan_norm is None else tuple(knotspan_norm))
            if key in self.cache:
                self.cache_hits += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            self.cache_misses += 1
            value = self.f(x, knots_arr, degree, endpoint, r, knotspan_norm)
            if self.maxsize:
                self.cache[key] = value
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
                    self.cache_evictions += 1
            return value

        def cache_clear(self):
            """Release cached values and reset statistics."""
            self.cache.clear()
            self.cache_hits = self.cache_misses = self.cache_evictions = 0

        def set_cache_maxsize(self, maxsize):
            """Set the entry limit; zero disables caching. Trim immediately."""
            if isinstance(maxsize, (bool, np.bool_)) or not isinstance(maxsize, (int, np.integer)) or maxsize < 0:
                raise ValueError("Cache size must be a nonnegative integer.")
            self.maxsize = int(maxsize)
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
                self.cache_evictions += 1

        def cache_info(self):
            return dict(hits=self.cache_hits, misses=self.cache_misses,
                        evictions=self.cache_evictions, maxsize=self.maxsize,
                        currsize=len(self.cache))

        def get_cache_hits(self):
            return self.cache_hits

    return MemoClass(f)


@memoize
def _evaluate_univariate_b_spline(x: float, knots: typing.Union[Vector, np.ndarray], degree: int,
                                  endpoint=False, r=0, knotspan_norm=None) -> float:
    """
    Evaluates a univariate BSpline corresponding to the given knot vector and polynomial degree at the point x.

    :param endpoint:
    :param x: point of evaluation
    :param knots: knot vector
    :param degree: polynomial degree
    :param r: derivative
    :return: B(x)
    """

    if r < 0 or int(r) != r:
        raise ValueError("Derivative order must be a nonnegative integer.")
    if r > degree:
        return 0.0
    knots = np.array(knots)
    i = _find_knot_interval(x, knots, endpoint=endpoint, knotspan_norm=knotspan_norm)
    if i == -1:
        return 0
    t = _augment_knots(knots, degree)
    i += degree + 1


    c = np.zeros(len(t) - degree - 1)
    c[degree + 1] = 1
    c = c[i - degree: i + 1]

    for k in range(degree, degree - r, -1):
        t1 = t[i - k + 1: i + 1]
        t2 = t[i + 1: i + k + 1]

        c = np.divide((c[1:] - c[:-1]), (t2 - t1), out=np.zeros_like(t1, dtype=np.float64), where=(t2 - t1) != 0)

    for k in range(degree - r, 0, -1):
        t1 = t[i - k + 1: i + 1]
        t2 = t[i + 1: i + k + 1]
        omega = np.divide((x - t1), (t2 - t1), out=np.zeros_like(t1, dtype=np.float64), where=(t2 - t1) != 0)

        a = np.multiply((1 - omega), c[:-1])
        b = np.multiply(omega, c[1:])
        c = a + b

    return factorial(degree) * c.squeeze() / factorial(degree - r)


def _augment_knots(knots: Vector, degree: int) -> np.ndarray:
    """
    Adds degree + 1 values to either end of the knot vector, in order to facilitate matrix based evaluation.

    :param knots: knot vector
    :param degree: polynomial degree
    :return: padded knot vector
    """
    return _cached_augmented_knots(tuple(knots), int(degree))


@lru_cache(maxsize=4096)
def _cached_augmented_knots(knots, degree):
    padded = np.concatenate((np.full(degree+1, knots[0]-1.), knots,
                             np.full(degree+1, knots[-1]+1.)))
    padded.setflags(write=False)
    return padded



def _find_knot_interval(x, knots, endpoint=False, knotspan_norm=None):
    """Right-hand value at interior knots; left-hand value at the patch end.

    An explicit cell span selects its polynomial piece at an element boundary.
    """
    if knotspan_norm is not None:
        a, b = knotspan_norm
        tolerance = 8*np.finfo(float).eps*max(abs(a), abs(b), abs(x), b-a)
        if not a < b or x < a-tolerance or x > b+tolerance:
            raise ValueError("Evaluation point must lie in the supplied nonempty cell span.")
        probe = a + (b - a) / 2
        if probe < knots[0] or probe >= knots[-1]:
            return -1
    elif x < knots[0] or x > knots[-1]:
        return -1
    elif x == knots[-1]:
        if not endpoint:
            return -1
        return int(np.searchsorted(knots, x, side='left') - 1)
    else:
        probe = x
    i = int(np.searchsorted(knots, probe, side='right') - 1)
    return i if 0 <= i < len(knots) - 1 else -1
