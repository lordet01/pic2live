from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import cv2
import glfw
import moderngl

from ..processing.motion_mapper import map_gesture_to_effect, map_dual_gesture_to_effect
from ..processing.shader_effect import INK_VERTEX_GLSL, INK_FRAGMENT_GLSL
from .overlay import _draw_debug_overlay
from ..utils.osc_sender import OSCSender


def _create_fullscreen_quad(ctx: moderngl.Context):
    vertices = np.array([
        # x, y,  u, v
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype=np.float32)
    vbo = ctx.buffer(vertices.tobytes())
    vao_content = [
        (vbo, '2f 2f', 'in_pos', 'in_uv')
    ]
    vao = ctx.vertex_array(ctx.program(vertex_shader=INK_VERTEX_GLSL, fragment_shader=INK_FRAGMENT_GLSL), vao_content)
    return vao


def _frame_to_texture(ctx: moderngl.Context, frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    tex = ctx.texture((w, h), 3, rgb.tobytes())
    tex.build_mipmaps()
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def _image_to_texture(ctx: moderngl.Context, img_bgr: np.ndarray):
    return _frame_to_texture(ctx, img_bgr)


def run_gl_ink_renderer(camera, tracker, base_image: np.ndarray, settings: Dict[str, Any]) -> None:
    window_cfg = settings.get('effects', {}).get('gl', {}).get('window', {})
    width = int(window_cfg.get('width', 1280))
    height = int(window_cfg.get('height', 720))
    fullscreen = bool(window_cfg.get('fullscreen', False))

    if not glfw.init():
        raise RuntimeError('GLFW init failed')
    try:
        monitor = glfw.get_primary_monitor() if fullscreen else None
        window = glfw.create_window(width, height, 'Ink Renderer', monitor, None)
        if not window:
            raise RuntimeError('GLFW window creation failed')
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        program = ctx.program(vertex_shader=INK_VERTEX_GLSL, fragment_shader=INK_FRAGMENT_GLSL)
        # left quad (camera): x in [-1, 0]
        vertices_left = np.array([
            -1.0, -1.0, 0.0, 0.0,
             0.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             0.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)
        vbo_left = ctx.buffer(vertices_left.tobytes())
        vao_left = ctx.vertex_array(program, [(vbo_left, '2f 2f', 'in_pos', 'in_uv')])

        # right quad (calligraphy effect): x in [0, 1]
        vertices_right = np.array([
             0.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             0.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)
        vbo_right = ctx.buffer(vertices_right.tobytes())
        vao_right = ctx.vertex_array(program, [(vbo_right, '2f 2f', 'in_pos', 'in_uv')])

        # full-screen quad (effect-only fullscreen mode)
        vertices_full = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)
        vbo_full = ctx.buffer(vertices_full.tobytes())
        vao_full = ctx.vertex_array(program, [(vbo_full, '2f 2f', 'in_pos', 'in_uv')])

        # static base
        base_tex = _image_to_texture(ctx, base_image)
        program['u_base'] = 1  # texture unit 1

        strength_scale = float(settings.get('effects', {}).get('gl', {}).get('ink', {}).get('strength_scale', 1.5))
        noise_amt = float(settings.get('effects', {}).get('gl', {}).get('ink', {}).get('noise_amount', 0.45))

        # OSC 송신 초기화
        osc_cfg = settings.get('osc', {})
        osc_sender = OSCSender(
            host=osc_cfg.get('host', '127.0.0.1'),
            port=int(osc_cfg.get('port', 9000)),
            enabled=bool(osc_cfg.get('enabled', True))
        )

        current_mode = 1
        effect_only = False
        # store original window position/size for restoring from fullscreen
        try:
            orig_x, orig_y = glfw.get_window_pos(window)
            orig_w, orig_h = glfw.get_window_size(window)
        except Exception:
            orig_x, orig_y = 100, 100
            orig_w, orig_h = width, height
        while not glfw.window_should_close(window):
            # Ensure viewport matches current framebuffer size (handles fullscreen, DPI scaling)
            vw, vh = glfw.get_framebuffer_size(window)
            try:
                ctx.viewport = (0, 0, int(vw), int(vh))
            except Exception:
                pass
            ok, frame = camera.read()
            if not ok or frame is None:
                frame = base_image

            state = tracker.process(frame)
            ctrl_left, ctrl_right = map_dual_gesture_to_effect(state, settings)
            
            # 주 손 (오른손 우선) 사용
            ctrl = ctrl_right if state.right_hand.present else ctrl_left
            dx, dy = ctrl['direction']
            mag = ctrl['magnitude'] * strength_scale
            
            # OSC로 양손 제스처 데이터 전송
            osc_sender.send_gesture_state(state, ctrl_left, ctrl_right)
            osc_sender.send_custom('/effect/mode', current_mode)

            # resize base image to camera size and upload both textures
            fbw, fbh = frame.shape[1], frame.shape[0]
            base_img = cv2.resize(base_image, (fbw, fbh), interpolation=cv2.INTER_LINEAR)
            # annotate overlays only on left (camera) side; right/effect uses clean base image
            frame_anno = frame.copy()
            _draw_debug_overlay(frame_anno, state, ctrl_left, ctrl_right, settings)

            frame_tex = _frame_to_texture(ctx, frame_anno)
            base_tex = _image_to_texture(ctx, base_img)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            if effect_only:
                # Fullscreen effect-only on calligraphy (no overlays)
                base_tex.use(location=1)
                program['u_base'] = 1
                program['u_resolution'] = (float(fbw), float(fbh))
                program['u_dir'] = (float(dx), float(-dy))
                program['u_strength'] = float(mag)
                program['u_noise_amt'] = float(noise_amt)
                program['u_pos'] = (float(state.position_norm[0]), float(1.0 - state.position_norm[1]))
                program['u_mode'] = int(current_mode)
                vao_full.render(moderngl.TRIANGLE_STRIP)
            else:
                # Left half: camera (no effect), with overlays baked on CPU
                frame_tex.use(location=1)
                program['u_base'] = 1
                program['u_resolution'] = (float(fbw), float(fbh))
                program['u_dir'] = (0.0, 0.0)
                program['u_strength'] = 0.0
                program['u_noise_amt'] = 0.0
                program['u_pos'] = (float(state.position_norm[0]), float(1.0 - state.position_norm[1]))
                program['u_mode'] = 0
                vao_left.render(moderngl.TRIANGLE_STRIP)

                # Right half: calligraphy with effect
                base_tex.use(location=1)
                program['u_base'] = 1
                program['u_resolution'] = (float(fbw), float(fbh))
                program['u_dir'] = (float(dx), float(-dy))
                program['u_strength'] = float(mag)
                program['u_noise_amt'] = float(noise_amt)
                program['u_pos'] = (float(state.position_norm[0]), float(1.0 - state.position_norm[1]))
                program['u_mode'] = int(current_mode)
                vao_right.render(moderngl.TRIANGLE_STRIP)

            # release per-frame textures to avoid memory growth
            frame_tex.release()
            base_tex.release()

            glfw.swap_buffers(window)
            glfw.poll_events()
            # keyboard handling for mode switch 1..9
            for key, mode in [
                (glfw.KEY_1, 1), (glfw.KEY_2, 2), (glfw.KEY_3, 3),
                (glfw.KEY_4, 4), (glfw.KEY_5, 5), (glfw.KEY_6, 6),
                (glfw.KEY_7, 7), (glfw.KEY_8, 8), (glfw.KEY_9, 9),
            ]:
                if glfw.get_key(window, key) == glfw.PRESS:
                    current_mode = mode

            # toggle fullscreen + effect-only on '+'; return to windowed split on '-'
            if glfw.get_key(window, glfw.KEY_KP_ADD) == glfw.PRESS or glfw.get_key(window, glfw.KEY_EQUAL) == glfw.PRESS:
                if not effect_only:
                    monitor = glfw.get_primary_monitor()
                    vm = glfw.get_video_mode(monitor)
                    try:
                        mw = vm.size.width
                        mh = vm.size.height
                        rr = vm.refresh_rate
                    except Exception:
                        # Fallback if attributes differ
                        mw, mh, rr = 1920, 1080, 0
                    glfw.set_window_monitor(window, monitor, 0, 0, mw, mh, rr)
                    effect_only = True

            if glfw.get_key(window, glfw.KEY_KP_SUBTRACT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_MINUS) == glfw.PRESS:
                if effect_only:
                    glfw.set_window_monitor(window, None, orig_x, orig_y, orig_w, orig_h, 0)
                    effect_only = False

        # cleanup
        osc_sender.close()
        glfw.destroy_window(window)
    finally:
        glfw.terminate()


