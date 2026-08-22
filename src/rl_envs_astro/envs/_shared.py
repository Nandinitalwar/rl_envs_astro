"""Shared validation and circular-distance helpers."""

from typing import Any, Mapping, Optional

import numpy as np


def circular_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return shortest distances between sign indices on a 12-position ring."""

    direct = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return np.minimum(direct, 12 - direct)


def option_signs(
    options: Optional[Mapping[str, Any]],
    key: str,
    size: int,
) -> Optional[np.ndarray]:
    """Validate an optional reset vector of zodiac sign indices."""

    if options is None or key not in options:
        return None
    values = np.asarray(options[key])
    if values.shape != (size,):
        raise ValueError("options[%r] must contain exactly %d values" % (key, size))
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("options[%r] must contain integer sign indices" % key)
    if np.any(values < 0) or np.any(values >= 12):
        raise ValueError("options[%r] values must be between 0 and 11" % key)
    return values.astype(np.int64, copy=True)
