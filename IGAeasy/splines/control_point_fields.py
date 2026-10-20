"""Discover numeric coefficient fields, excluding spline topology and identity."""
import numpy as np

_STRUCTURAL = {
    'p', 'id', 'loc_id', 'level', 'weight', 'knots_u', 'knots_v',
    'norm_knots_u', 'norm_knots_v', 'end_u', 'end_v', 'trunc',
    'coefficient', 'children', 'active', 'trimmed', 'void',
    'elements_of_support',
}


def control_point_fields(basis):
    """Return shapes of numeric scalar/vector/tensor attributes.

    Missing fields represent zero coefficients. Non-numeric metadata is not
    prolonged. All occurrences of a field must have the same shape.
    """
    fields = {}
    for b in basis:
        for name, value in vars(b).items():
            if name.startswith('_') or name in _STRUCTURAL:
                continue
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                continue
            if array.dtype.kind not in 'fiu':
                continue
            if name in fields and fields[name] != array.shape:
                raise ValueError(f"Inconsistent shape for control-point field {name!r}.")
            fields[name] = array.shape
    return fields
