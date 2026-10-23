@echo off
rem Offline help/planning by default. A present human may explicitly use -StartLondon.
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\label-local.ps1" %*
set "LABEL_LOCAL_EXIT=%ERRORLEVEL%"
exit /b %LABEL_LOCAL_EXIT%
