from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass
class CameraConfig:
    camera_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class CameraStream:
    """Thin wrapper around OpenCV VideoCapture.

    Provides a minimal interface for reading frames and reporting status.
    """

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        # On Windows, CAP_DSHOW often gives more reliable device opening
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

    @property
    def is_opened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional["cv2.Mat"]]:
        if not self.is_opened:
            return False, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def frame_size(self) -> Tuple[int, int]:
        if not self.is_opened:
            return self.width, self.height
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None


