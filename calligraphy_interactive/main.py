import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import yaml

from .utils.logger import get_logger
from .utils.device_check import is_gl_available
from .input.camera_stream import CameraStream
from .input.gesture_tracker import GestureTracker
from .processing.motion_mapper import map_gesture_to_effect
from .visualization.overlay import (
    load_or_generate_base_image,
)


def load_settings(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_mode(cli_mode: Optional[str], settings: dict) -> str:
    # GL only
    return "gl"


## CV2 mode removed


def run_phase2_gl(settings: dict) -> None:
    logger = get_logger(settings.get("logging", {}).get("level", "INFO"))

    if not is_gl_available():
        logger.error("OpenGL/GLFW/moderngl not available. Try mode=cv2 instead.")
        sys.exit(1)

    cam_cfg = settings.get("camera", {})
    camera_id = cam_cfg.get("id", 0)
    width = cam_cfg.get("width", 1280)
    height = cam_cfg.get("height", 720)
    fps = cam_cfg.get("fps", 30)

    images_cfg = settings.get("images", {})
    image_paths = images_cfg.get("paths", [])
    base_image = load_or_generate_base_image(image_paths)

    camera = CameraStream(camera_id=camera_id, width=width, height=height, fps=fps)
    
    # 손 크기 설정 로드
    gesture_cfg = settings.get('gesture', {})
    hand_ref_size = float(gesture_cfg.get('hand_reference_size_cm', 18.0))
    tracker = GestureTracker(hand_reference_size_cm=hand_ref_size)

    try:
        from .visualization.renderer import run_gl_ink_renderer
        run_gl_ink_renderer(camera, tracker, base_image, settings)
    finally:
        camera.release()
        tracker.close()
        logger.info("Phase 2 GL renderer stopped.")


## Diffusion mode removed


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive calligraphy demo")
    parser.add_argument("--config", type=str, default=str(Path("calligraphy_interactive/config/settings.yaml").resolve()), help="Path to settings.yaml")
    # GL only
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    settings = load_settings(config_path)
    mode = resolve_mode(None, settings)
    run_phase2_gl(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


