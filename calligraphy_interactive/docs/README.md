## Calligraphy Interactive (Python)

입력: USB 카메라 실시간 피드 + 이미지 N장
출력: 인터랙티브 실시간 영상 (손동작 방향에 따른 먹 번짐 효과)

구성:
- MediaPipe (손 좌표/속도)
- OpenCV / GLSL 셰이더 / CPU diffusion
- PyOpenGL 대체: moderngl + glfw

실행 (PowerShell, Windows 11, Python 3.11.4):

```powershell
# 프로젝트 루트에서
pip install -r requirements.txt
# (선택) 확장 diffusion CPU 의존성
pip install -r requirements-diffusion.txt

# 기본 CV2 파이프라인
python -m calligraphy_interactive.main --mode cv2

# GLSL 셰이더 기반 렌더러
python -m calligraphy_interactive.main --mode gl

# CPU anisotropic diffusion 파이프라인
python -m calligraphy_interactive.main --mode diffusion
```

설정 변경: `calligraphy_interactive/config/settings.yaml`

자산:
- `assets/calligraphy.png`를 `images.paths`에 지정하면 사용됩니다. 없으면 기본 캔버스가 생성됩니다.

키:
- ESC: 종료

참고:
- 카메라가 열리지 않으면 기본 이미지로 대체되어 동작합니다.
- mediapipe 설치가 실패하면 Visual C++ 빌드 도구가 필요할 수 있습니다.


