@echo off
setlocal

echo ============================================================
echo  RPG Agent Arena -- Windows Setup
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python --version
echo.

if not exist ".env" (
    echo Creating .env from template...
    copy ".env.example" ".env" >nul
    echo.
    echo IMPORTANT: Open .env in Notepad and add your ANTHROPIC_API_KEY
    echo Then run setup.bat again, or run: python main.py
    echo.
    pause
    exit /b 0
)

echo Installing dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

echo.
echo Setup complete.
echo.
echo Usage:
echo   python main.py                          -- single battle
echo   python main.py --games 5               -- 5-game series
echo   python main.py --status                -- check API keys
echo   python main.py --help                  -- all options
echo.
pause
endlocal
