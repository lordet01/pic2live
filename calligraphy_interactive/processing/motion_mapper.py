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


def map_dual_gesture_to_effect(state, settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """양손 제스처를 각각 이펙트 컨트롤로 매핑
    
    Returns:
        (ctrl_left, ctrl_right): 왼손과 오른손의 이펙트 컨트롤 딕셔너리
    """
    # 왼손 매핑
    if state.left_hand.present:
        ctrl_left = map_gesture_to_effect(
            state.left_hand.position_norm, 
            state.left_hand.velocity_norm, 
            settings
        )
    else:
        ctrl_left = {
            "direction": (0.0, 0.0),
            "magnitude": 0.0,
            "position": (0.5, 0.5),
        }
    
    # 오른손 매핑
    if state.right_hand.present:
        ctrl_right = map_gesture_to_effect(
            state.right_hand.position_norm, 
            state.right_hand.velocity_norm, 
            settings
        )
    else:
        ctrl_right = {
            "direction": (0.0, 0.0),
            "magnitude": 0.0,
            "position": (0.5, 0.5),
        }
    
    return ctrl_left, ctrl_right


