@echo off
rem One labelling session on Windows: double-click this file. It runs scripts\label.ps1
rem with Windows PowerShell, bypassing the execution policy for that one script only
rem (this process; nothing is changed for the machine or the user), and keeps the window
rem open at the end. Options are passed on, e.g. label.cmd -Source london -Target 40.
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\label.ps1" %*
set "LABEL_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %LABEL_EXIT%
