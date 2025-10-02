from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class HandData:
    """단일 손의 데이터"""
    present: bool = False
    position_norm: Tuple[float, float] = (0.5, 0.5)  # (x, y) in [0,1]
    velocity_norm: Tuple[float, float] = (0.0, 0.0)
    landmarks_norm: Optional[List[Tuple[float, float]]] = None
    # 엄지-검지 fingertip 데이터
    thumb_index_distance_norm: float = 0.0  # 정규화된 거리 [0,1]
    thumb_index_distance_cm: float = 0.0  # cm 단위 거리
    thumb_index_angle_rad: float = 0.0  # radian 단위 각도 (-π ~ π)


@dataclass
class GestureState:
    """양손 제스처 상태"""
    left_hand: HandData = None
    right_hand: HandData = None
    
    def __post_init__(self):
        if self.left_hand is None:
            self.left_hand = HandData()
        if self.right_hand is None:
            self.right_hand = HandData()
    
    @property
    def any_hand_present(self) -> bool:
        return self.left_hand.present or self.right_hand.present
    
    # 하위 호환성을 위한 속성들 (주 손 = 오른손 우선, 없으면 왼손)
    @property
    def hand_present(self) -> bool:
        return self.any_hand_present
    
    @property
    def position_norm(self) -> Tuple[float, float]:
        if self.right_hand.present:
            return self.right_hand.position_norm
        return self.left_hand.position_norm
    
    @property
    def velocity_norm(self) -> Tuple[float, float]:
        if self.right_hand.present:
            return self.right_hand.velocity_norm
        return self.left_hand.velocity_norm
    
    @property
    def landmarks_norm(self) -> Optional[List[Tuple[float, float]]]:
        if self.right_hand.present:
            return self.right_hand.landmarks_norm
        return self.left_hand.landmarks_norm


class GestureTracker:
    """MediaPipe Hands-based tracker returning coarse gesture state.

    For performance, we only compute a simple centroid and a velocity vector
    estimated by exponential smoothing over frames.
    
    Now supports both hands detection and tracking.
    """

    def __init__(self, hand_reference_size_cm: float = 18.0) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 양손 인식
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        # 각 손에 대한 이전 위치와 속도 추적
        self._prev_pos_left: Optional[Tuple[float, float]] = None
        self._prev_pos_right: Optional[Tuple[float, float]] = None
        self._alpha = 0.6  # smoothing factor for velocity
        self._prev_velocity_left: Tuple[float, float] = (0.0, 0.0)
        self._prev_velocity_right: Tuple[float, float] = (0.0, 0.0)
        self._hand_reference_size_cm = hand_reference_size_cm  # 손 크기 기준값 (cm)
    
    def _calculate_thumb_index_metrics(self, landmarks: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """엄지와 검지 fingertip 간의 거리와 각도 계산
        
        Returns:
            (distance_norm, distance_cm, angle_rad): 정규화된 거리, cm 거리, radian 각도
        """
        # MediaPipe Hand Landmarks:
        # 4 = THUMB_TIP
        # 8 = INDEX_FINGER_TIP
        # 0 = WRIST
        # 5 = INDEX_FINGER_MCP (손바닥 기준점)
        
        if len(landmarks) < 9:
            return 0.0, 0.0, 0.0
        
        thumb_tip = np.array(landmarks[4])
        index_tip = np.array(landmarks[8])
        wrist = np.array(landmarks[0])
        index_mcp = np.array(landmarks[5])
        
        # 엄지-검지 거리 (정규화된 좌표 공간)
        distance_norm = float(np.linalg.norm(thumb_tip - index_tip))
        
        # 손 크기 추정: 손목에서 검지 MCP까지의 거리를 손바닥 길이로 사용
        hand_size_norm = float(np.linalg.norm(index_mcp - wrist))
        
        # 실제 cm로 변환
        # 손바닥 길이를 약 10cm로 가정하고 스케일링
        if hand_size_norm > 0.01:  # 너무 작은 값 방지
            scale_factor = self._hand_reference_size_cm / hand_size_norm
            distance_cm = distance_norm * scale_factor
        else:
            distance_cm = 0.0
        
        # 각도 계산: 검지에서 엄지로의 벡터 각도
        # atan2(dy, dx)는 -π ~ π 범위의 각도 반환
        # 수평 오른쪽이 0, 반시계방향이 양수
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        angle_rad = float(np.arctan2(dy, dx))
        
        return distance_norm, distance_cm, angle_rad

    def process(self, frame_bgr: "cv2.Mat") -> GestureState:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        left_hand_data = HandData()
        right_hand_data = HandData()

        if not result.multi_hand_landmarks or not result.multi_handedness:
            # Decay velocity when no hands detected
            vx_l, vy_l = self._prev_velocity_left
            vx_l *= 0.8
            vy_l *= 0.8
            self._prev_velocity_left = (vx_l, vy_l)
            left_hand_data.velocity_norm = (vx_l, vy_l)
            
            vx_r, vy_r = self._prev_velocity_right
            vx_r *= 0.8
            vy_r *= 0.8
            self._prev_velocity_right = (vx_r, vy_r)
            right_hand_data.velocity_norm = (vx_r, vy_r)
            
            return GestureState(left_hand=left_hand_data, right_hand=right_hand_data)

        # Process each detected hand
        for idx, (hand_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            # MediaPipe의 handedness는 카메라 관점에서 좌우가 반대임 (미러링)
            # classification[0].label이 "Left"이면 실제로는 오른손
            label = handedness.classification[0].label
            is_right = (label == "Left")  # 미러링 보정
            
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            cx = float(np.clip(np.mean(xs), 0.0, 1.0))
            cy = float(np.clip(np.mean(ys), 0.0, 1.0))
            
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            
            if is_right:
                # 오른손 처리
                if self._prev_pos_right is None:
                    self._prev_pos_right = (cx, cy)
                    inst_vx, inst_vy = 0.0, 0.0
                else:
                    px, py = self._prev_pos_right
                    inst_vx = cx - px
                    inst_vy = cy - py
                    self._prev_pos_right = (cx, cy)
                
                # exponential smoothing
                vx = self._alpha * inst_vx + (1.0 - self._alpha) * self._prev_velocity_right[0]
                vy = self._alpha * inst_vy + (1.0 - self._alpha) * self._prev_velocity_right[1]
                self._prev_velocity_right = (vx, vy)
                
                # 엄지-검지 메트릭 계산
                dist_norm, dist_cm, angle_rad = self._calculate_thumb_index_metrics(landmarks)
                
                right_hand_data = HandData(
                    present=True,
                    position_norm=(cx, cy),
                    velocity_norm=(vx, vy),
                    landmarks_norm=landmarks,
                    thumb_index_distance_norm=dist_norm,
                    thumb_index_distance_cm=dist_cm,
                    thumb_index_angle_rad=angle_rad
                )
            else:
                # 왼손 처리
                if self._prev_pos_left is None:
                    self._prev_pos_left = (cx, cy)
                    inst_vx, inst_vy = 0.0, 0.0
                else:
                    px, py = self._prev_pos_left
                    inst_vx = cx - px
                    inst_vy = cy - py
                    self._prev_pos_left = (cx, cy)
                
                # exponential smoothing
                vx = self._alpha * inst_vx + (1.0 - self._alpha) * self._prev_velocity_left[0]
                vy = self._alpha * inst_vy + (1.0 - self._alpha) * self._prev_velocity_left[1]
                self._prev_velocity_left = (vx, vy)
                
                # 엄지-검지 메트릭 계산
                dist_norm, dist_cm, angle_rad = self._calculate_thumb_index_metrics(landmarks)
                
                left_hand_data = HandData(
                    present=True,
                    position_norm=(cx, cy),
                    velocity_norm=(vx, vy),
                    landmarks_norm=landmarks,
                    thumb_index_distance_norm=dist_norm,
                    thumb_index_distance_cm=dist_cm,
                    thumb_index_angle_rad=angle_rad
                )
        
        return GestureState(left_hand=left_hand_data, right_hand=right_hand_data)

    def close(self) -> None:
        if self._hands is not None:
            try:
                self._hands.close()
            finally:
                self._hands = None


