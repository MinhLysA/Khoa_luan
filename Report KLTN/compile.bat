@echo off
chcp 65001 >nul
set MIKTEX=C:\Users\HP\AppData\Local\Programs\MiKTeX\miktex\bin\x64
set TARGET=main
if not "%1"=="" set TARGET=%1

cd /d "%~dp0"

echo [1/3] XeLaTeX pass 1...
"%MIKTEX%\xelatex.exe" -synctex=1 -interaction=nonstopmode %TARGET%.tex
if errorlevel 1 goto fail

echo [2/3] Biber...
"%MIKTEX%\biber.exe" %TARGET%

echo [3/3] XeLaTeX pass 2...
"%MIKTEX%\xelatex.exe" -synctex=1 -interaction=nonstopmode %TARGET%.tex
if errorlevel 1 goto fail

echo Done! %TARGET%.pdf
start "" "%~dp0%TARGET%.pdf"
goto end

:fail
echo ERROR! Check %TARGET%.log
:end
