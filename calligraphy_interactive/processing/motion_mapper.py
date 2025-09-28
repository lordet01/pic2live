from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np


def map_gesture_to_effect(position_norm: Tuple[float, float], velocity_norm: Tuple[float, float], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Map gesture to an effect control dictionary.

    Returns a dict including direction, magnitude, and mix controls that can
    be consumed by visualization layers.
    """
    vx, vy = velocity_norm
    # magnitude based on L2 norm in normalized space
    magnitude = float(np.sqrt(vx * vx + vy * vy))
    # normalize direction; avoid division by zero
    if magnitude > 1e-6:
        dir_x = float(vx / magnitude)
        dir_y = float(vy / magnitude)
    else:
        dir_x, dir_y = 0.0, 0.0

    # scale by user parameter
    scale = float(settings.get("effects", {}).get("cv2", {}).get("motion_magnitude_scale", 1.0))
    magnitude *= scale

    # clamp magnitude to a reasonable range for visual stability
    magnitude = float(np.clip(magnitude, 0.0, 0.25))

    return {
        "direction": (dir_x, dir_y),
        "magnitude": magnitude,
        "position": position_norm,
    }


