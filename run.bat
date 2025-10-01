@echo off
chcp 65001 > nul
echo ========================================
echo Calligraphy Interactive 실행 스크립트
echo ========================================
echo.

REM 가상환경 폴더 이름
set VENV_DIR=venv

REM 가상환경이 없으면 생성
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/3] 가상환경 생성 중...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: 가상환경 생성 실패. Python이 설치되어 있는지 확인하세요.
        pause
        exit /b 1
    )
    echo 가상환경 생성 완료!
    echo.
)

REM 가상환경 활성화
echo [2/3] 가상환경 활성화 중...
call %VENV_DIR%\Scripts\activate.bat

REM 필요한 패키지 설치
echo [3/3] 필수 패키지 설치 확인 중...
python -c "import cv2, mediapipe, moderngl, glfw" 2>nul
if errorlevel 1 (
    echo 패키지 설치 중... (최초 1회만 소요됩니다)
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: 패키지 설치 실패
        pause
        exit /b 1
    )
)
echo 패키지 준비 완료!
echo.

REM 프로그램 실행
echo ========================================
echo 프로그램 실행 중...
echo ========================================
echo.
echo [키보드 단축키]
echo   1-9: 이펙트 모드 변경
echo   + : 전체화면 + 이펙트만 표시
echo   - : 분할 화면 복귀
echo.

python -m calligraphy_interactive.main

REM 에러 체크
if errorlevel 1 (
    echo.
    echo ERROR: 프로그램 실행 중 오류 발생
    pause
)

