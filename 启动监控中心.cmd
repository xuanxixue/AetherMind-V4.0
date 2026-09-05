@echo off
title AetherMind V4 Monitor
cd /d "%~dp0monitor-app"

echo ============================================
echo   AetherMind V4 Training Monitor
echo   Electron + xterm.js + node-pty
echo ============================================
echo.

if not exist "node_modules" (
    echo [i] First run: installing dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
    echo.
)

echo [i] Launching monitor...
npx electron .
if errorlevel 1 (
    echo.
    echo [ERROR] Electron failed, code: %errorlevel%
    pause
)
