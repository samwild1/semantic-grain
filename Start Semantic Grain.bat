@echo off
title Semantic Grain

:: Try miniconda3 first, then anaconda3, then conda on PATH
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate sgrain
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate sgrain
) else (
    call conda activate sgrain
)

cd /d "%~dp0"
python -m semantic_grain
pause
