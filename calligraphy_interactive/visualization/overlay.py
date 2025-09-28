from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from ..processing.motion_mapper import map_gesture_to_effect


def load_or_generate_base_image(paths: List[str]) -> np.ndarray:
    if paths:
        p = Path(paths[0])
        if p.exists():
            img = Image.open(str(p)).convert('RGB')
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # fallback: generate simple white canvas with black strokes
    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'Calligraphy', (120, 380), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8, cv2.LINE_AA)
    return canvas


def apply_cv2_ink_effect(frame: np.ndarray, ctrl: Dict[str, Any], settings: Dict[str, Any]) -> np.ndarray:
    dx, dy = ctrl['direction']
    mag = ctrl['magnitude']
    blur_max = int(settings.get('effects', {}).get('cv2', {}).get('blur_max_radius', 18))
    base_mix = float(settings.get('effects', {}).get('cv2', {}).get('base_mix', 0.85))

    # directional motion blur approximation
    ksize = max(1, int(1 + mag * blur_max))
    ksize = (ksize if ksize % 2 == 1 else ksize + 1)
    angle = float(np.degrees(np.arctan2(dy, dx))) if (abs(dx) + abs(dy)) > 1e-6 else 0.0

    kernel = directional_kernel(ksize, angle)
    blurred = cv2.filter2D(frame, -1, kernel)

    out = cv2.addWeighted(frame, base_mix, blurred, 1.0 - base_mix, 0)
    return out


def directional_kernel(size: int, angle_deg: float) -> np.ndarray:
    size = max(3, size)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    for i in range(size):
        x = i - center
        y = int(round(x * np.tan(angle)))
        xi = center + x
        yi = center + y
        if 0 <= xi < size and 0 <= yi < size:
            kernel[yi, xi] = 1.0
    s = np.sum(kernel)
    if s > 0:
        kernel /= s
    else:
        kernel[center, center] = 1.0
    return kernel


def _draw_debug_overlay(img: np.ndarray, state, ctrl: Dict[str, Any], settings: Dict[str, Any]) -> None:
    debug = settings.get('debug', {})
    if not debug.get('enabled', False):
        return
    h, w = img.shape[:2]
    # draw landmarks
    if debug.get('draw_landmarks', True) and state.landmarks_norm:
        for (lx, ly) in state.landmarks_norm:
            cx = int(lx * w)
            cy = int(ly * h)
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
    # draw velocity vector
    if debug.get('draw_velocity', True):
        x = int(state.position_norm[0] * w)
        y = int(state.position_norm[1] * h)
        vx = int(ctrl['direction'][0] * 120)
        vy = int(ctrl['direction'][1] * 120)
        cv2.arrowedLine(img, (x, y), (x + vx, y + vy), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    # draw text
    if debug.get('draw_text', True):
        mag = ctrl['magnitude']
        cv2.putText(img, f"mag={mag:.3f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        thr = float(settings.get('effects', {}).get('cv2', {}).get('trigger_threshold', 0.01))
        if mag > thr:
            cv2.putText(img, "TRIGGER", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 80), 2, cv2.LINE_AA)
        dx, dy = ctrl['direction']
        cv2.putText(img, f"dir=({dx:.2f},{dy:.2f})", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 80), 2, cv2.LINE_AA)


## CV2 visualization removed


