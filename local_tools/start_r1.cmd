@echo off
rem Start the R1 panel. Args go to start_r1.ps1, e.g. -Offline / -Restart / -Stop / -DryRun / -Backend strata
where pwsh.exe >nul 2>nul
if errorlevel 1 goto windows_powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_r1.ps1" %*
exit /b %errorlevel%
:windows_powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_r1.ps1" %*
exit /b %errorlevel%
