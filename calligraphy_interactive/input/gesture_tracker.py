from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class GestureState:
    hand_present: bool = False
    position_norm: Tuple[float, float] = (0.5, 0.5)  # (x, y) in [0,1]
    velocity_norm: Tuple[float, float] = (0.0, 0.0)
    landmarks_norm: Optional[List[Tuple[float, float]]] = None


class GestureTracker:
    """MediaPipe Hands-based tracker returning coarse gesture state.

    For performance, we only compute a simple centroid and a velocity vector
    estimated by exponential smoothing over frames.
    """

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        self._prev_pos: Optional[Tuple[float, float]] = None
        self._alpha = 0.6  # smoothing factor for velocity
        self._prev_velocity: Tuple[float, float] = (0.0, 0.0)

    def process(self, frame_bgr: "cv2.Mat") -> GestureState:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            # Decay velocity when no hand
            vx, vy = self._prev_velocity
            vx *= 0.8
            vy *= 0.8
            self._prev_velocity = (vx, vy)
            return GestureState(False, (0.5, 0.5), (vx, vy), None)

        # single hand only
        hand_landmarks = result.multi_hand_landmarks[0]
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        cx = float(np.clip(np.mean(xs), 0.0, 1.0))
        cy = float(np.clip(np.mean(ys), 0.0, 1.0))

        if self._prev_pos is None:
            self._prev_pos = (cx, cy)
            inst_vx, inst_vy = 0.0, 0.0
        else:
            px, py = self._prev_pos
            inst_vx = cx - px
            inst_vy = cy - py
            self._prev_pos = (cx, cy)

        # exponential smoothing
        vx = self._alpha * inst_vx + (1.0 - self._alpha) * self._prev_velocity[0]
        vy = self._alpha * inst_vy + (1.0 - self._alpha) * self._prev_velocity[1]
        self._prev_velocity = (vx, vy)

        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        return GestureState(True, (cx, cy), (vx, vy), landmarks)

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            finally:
                self._hands = None


