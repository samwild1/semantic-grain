@echo off
title Stopping Semantic Grain
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:7860" ^| findstr "LISTENING"') do (
    echo Stopping Semantic Grain server (PID %%a)...
    taskkill /PID %%a /F
)
echo Done.
timeout /t 3
