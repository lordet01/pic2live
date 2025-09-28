from __future__ import annotations

import importlib


def is_gl_available() -> bool:
    try:
        importlib.import_module('glfw')
        importlib.import_module('moderngl')
        return True
    except Exception:
        return False


