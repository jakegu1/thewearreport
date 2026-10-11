@echo off
rem Install the two city shortcuts. The policy bypass applies to this process only.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-label-shortcuts.ps1" %*
set "SHORTCUT_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %SHORTCUT_EXIT%
