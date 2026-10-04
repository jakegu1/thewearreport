@echo off
rem One London labelling session on Windows: double-click this file. It runs
rem scripts\label.ps1 as label.cmd does, with -Source london -MaxPasses 1 -NoJudge: London
rem only, one pass (a second pass minutes later would show some of the same people again),
rem and no hosted judge. Further options are passed on after these, e.g.
rem label-london.cmd -TempDir E:\tmp.
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\label.ps1" -Source london -MaxPasses 1 -NoJudge %*
set "LABEL_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %LABEL_EXIT%
