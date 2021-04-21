import math
import cmath

from typing import cast, List, Optional, Tuple, Union

# (bool, msg) tuple, where msg is None if and only if bool is True.
_compare_return_type = Tuple[bool, Optional[str]]


# Checks if two scalars are equal(-ish), returning (True, None)
# when they are and (False, debug_msg) when they are not.
def _compare_scalars_internal(a, b, *, rtol: float, atol: float, equal_nan: Union[str, bool]) -> _compare_return_type:
    def _helper(a, b, s) -> _compare_return_type:
        # Short-circuits on identity
        if a == b or ((equal_nan in {"relaxed", True}) and a != a and b != b):
            return (True, None)

        # Special-case for NaN comparisions when equal_nan=False
        if not (equal_nan in {"relaxed", True}) and (a != a or b != b):
            msg = ("Found {0} and {1} while comparing" + s + "and either one "
                   "is nan and the other isn't, or both are nan and "
                   "equal_nan is False").format(a, b)
            return (False, msg)

        diff = abs(a - b)
        allowed_diff = atol + rtol * abs(b)
        result = diff <= allowed_diff

        # Special-case for infinity comparisons
        # NOTE: if b is inf then allowed_diff will be inf when rtol is not 0
        if ((math.isinf(a) or math.isinf(b)) and a != b):
            result = False

        msg = None
        if not result:
            msg = ("Comparing" + s + "{0} and {1} gives a "
                   "difference of {2}, but the allowed difference "
                   "with rtol={3} and atol={4} is "
                   "only {5}!").format(a, b, diff,
                                       rtol, atol, allowed_diff)

        return result, msg

    if isinstance(a, complex) or isinstance(b, complex):
        a = complex(a)
        b = complex(b)

        if equal_nan == "relaxed":
            if cmath.isnan(a) and cmath.isnan(b):
                return (True, None)

        result, msg = _helper(a.real, b.real, " the real part ")

        if not result:
            return (False, msg)

        return _helper(a.imag, b.imag, " the imaginary part ")

    return _helper(a, b, " ")