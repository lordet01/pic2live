## Calligraphy Interactive (GL 전용)

입력: USB 카메라 실시간 피드 + 이미지 N장
출력: 인터랙티브 실시간 영상 (손동작 방향에 따른 먹 번짐/다양한 이펙트)

구성:
- MediaPipe (손 좌표/속도)
- GLSL 셰이더 (moderngl + glfw)

실행 (PowerShell, Windows 11, Python 3.11.4):

```powershell
# 프로젝트 루트에서
pip install -r requirements.txt

# GL 렌더러 실행 (GL-only)
python -m calligraphy_interactive.main --config D:\works\pic2live\calligraphy_interactive\config\settings.yaml
```

설정 변경: `calligraphy_interactive/config/settings.yaml`

자산:
- `assets/calligraphy.png`를 `images.paths`에 지정. 없으면 기본 캔버스 생성.

키 (실행 중):
- 1~9: 이펙트 모드 전환 (우측 서예 영역)
- + (또는 =): 전체화면 + 효과만 표시 (카메라/오버레이 숨김)
- -: 창모드 복귀 (좌: 카메라+오버레이 / 우: 이펙트)
- ESC: 종료

참고:
- 카메라가 열리지 않으면 베이스 이미지만으로 동작합니다.


