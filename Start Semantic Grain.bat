@echo off
title Semantic Grain
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate sgrain
cd /d "%~dp0"
python -m semantic_grain
pause
