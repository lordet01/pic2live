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


def _draw_debug_overlay(img: np.ndarray, state, ctrl_left: Dict[str, Any], ctrl_right: Dict[str, Any], settings: Dict[str, Any]) -> None:
    debug = settings.get('debug', {})
    if not debug.get('enabled', False):
        return
    h, w = img.shape[:2]
    
    # 왼손 그리기 (파란색 계열)
    if state.left_hand.present:
        # draw landmarks
        if debug.get('draw_landmarks', True) and state.left_hand.landmarks_norm:
            for (lx, ly) in state.left_hand.landmarks_norm:
                cx = int(lx * w)
                cy = int(ly * h)
                cv2.circle(img, (cx, cy), 3, (255, 100, 0), -1, cv2.LINE_AA)  # 파란색
        
        # draw thumb-index line
        if debug.get('draw_thumb_index_line', True) and state.left_hand.landmarks_norm and len(state.left_hand.landmarks_norm) > 8:
            thumb_tip = state.left_hand.landmarks_norm[4]
            index_tip = state.left_hand.landmarks_norm[8]
            thumb_px = int(thumb_tip[0] * w)
            thumb_py = int(thumb_tip[1] * h)
            index_px = int(index_tip[0] * w)
            index_py = int(index_tip[1] * h)
            
            # 엄지-검지 직선 그리기 (굵은 파란색)
            cv2.line(img, (thumb_px, thumb_py), (index_px, index_py), (255, 100, 100), 3, cv2.LINE_AA)
            # 양 끝점 강조
            cv2.circle(img, (thumb_px, thumb_py), 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (index_px, index_py), 6, (255, 0, 0), -1, cv2.LINE_AA)
            
            # 거리와 각도 표시 (직선 중간 위치)
            mid_x = (thumb_px + index_px) // 2
            mid_y = (thumb_py + index_py) // 2
            dist_cm = state.left_hand.thumb_index_distance_cm
            angle_rad = state.left_hand.thumb_index_angle_rad
            cv2.putText(img, f"{dist_cm:.1f}cm", (mid_x + 10, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 2, cv2.LINE_AA)
            cv2.putText(img, f"{angle_rad:.2f}rad", (mid_x + 10, mid_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 2, cv2.LINE_AA)
        
        # draw velocity vector
        if debug.get('draw_velocity', True):
            x = int(state.left_hand.position_norm[0] * w)
            y = int(state.left_hand.position_norm[1] * h)
            vx = int(ctrl_left['direction'][0] * 120)
            vy = int(ctrl_left['direction'][1] * 120)
            cv2.arrowedLine(img, (x, y), (x + vx, y + vy), (255, 200, 0), 2, cv2.LINE_AA, tipLength=0.3)  # 하늘색
    
    # 오른손 그리기 (녹색 계열)
    if state.right_hand.present:
        # draw landmarks
        if debug.get('draw_landmarks', True) and state.right_hand.landmarks_norm:
            for (lx, ly) in state.right_hand.landmarks_norm:
                cx = int(lx * w)
                cy = int(ly * h)
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)  # 녹색
        
        # draw thumb-index line
        if debug.get('draw_thumb_index_line', True) and state.right_hand.landmarks_norm and len(state.right_hand.landmarks_norm) > 8:
            thumb_tip = state.right_hand.landmarks_norm[4]
            index_tip = state.right_hand.landmarks_norm[8]
            thumb_px = int(thumb_tip[0] * w)
            thumb_py = int(thumb_tip[1] * h)
            index_px = int(index_tip[0] * w)
            index_py = int(index_tip[1] * h)
            
            # 엄지-검지 직선 그리기 (굵은 녹색)
            cv2.line(img, (thumb_px, thumb_py), (index_px, index_py), (100, 255, 100), 3, cv2.LINE_AA)
            # 양 끝점 강조
            cv2.circle(img, (thumb_px, thumb_py), 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (index_px, index_py), 6, (0, 255, 0), -1, cv2.LINE_AA)
            
            # 거리와 각도 표시 (직선 중간 위치)
            mid_x = (thumb_px + index_px) // 2
            mid_y = (thumb_py + index_py) // 2
            dist_cm = state.right_hand.thumb_index_distance_cm
            angle_rad = state.right_hand.thumb_index_angle_rad
            cv2.putText(img, f"{dist_cm:.1f}cm", (mid_x + 10, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2, cv2.LINE_AA)
            cv2.putText(img, f"{angle_rad:.2f}rad", (mid_x + 10, mid_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2, cv2.LINE_AA)
        
        # draw velocity vector
        if debug.get('draw_velocity', True):
            x = int(state.right_hand.position_norm[0] * w)
            y = int(state.right_hand.position_norm[1] * h)
            vx = int(ctrl_right['direction'][0] * 120)
            vy = int(ctrl_right['direction'][1] * 120)
            cv2.arrowedLine(img, (x, y), (x + vx, y + vy), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)  # 노란색
    
    # draw text info
    if debug.get('draw_text', True):
        y_offset = 32
        thr = float(settings.get('effects', {}).get('cv2', {}).get('trigger_threshold', 0.01))
        
        # 왼손 정보
        if state.left_hand.present:
            mag_l = ctrl_left['magnitude']
            cv2.putText(img, f"L mag={mag_l:.3f}", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2, cv2.LINE_AA)
            y_offset += 28
            if mag_l > thr:
                cv2.putText(img, "L TRIGGER", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 80), 2, cv2.LINE_AA)
                y_offset += 28
            dx, dy = ctrl_left['direction']
            cv2.putText(img, f"L dir=({dx:.2f},{dy:.2f})", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 80), 2, cv2.LINE_AA)
            y_offset += 28
        
        # 오른손 정보
        if state.right_hand.present:
            mag_r = ctrl_right['magnitude']
            cv2.putText(img, f"R mag={mag_r:.3f}", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2, cv2.LINE_AA)
            y_offset += 28
            if mag_r > thr:
                cv2.putText(img, "R TRIGGER", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
                y_offset += 28
            dx, dy = ctrl_right['direction']
            cv2.putText(img, f"R dir=({dx:.2f},{dy:.2f})", (16, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 200), 2, cv2.LINE_AA)
            y_offset += 28


## CV2 visualization removed


