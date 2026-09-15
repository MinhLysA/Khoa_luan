@echo off
chcp 65001 >nul
setlocal

set MIKTEX=C:\Users\HP\AppData\Local\Programs\MiKTeX\miktex\bin\x64
cd /d "%~dp0"

if "%1"=="clean" goto clean
if "%1"=="full" goto full

:: --- QUICK RUN (1 PASS) ---
echo [*] Dang bien dich nhanh main.tex...
"%MIKTEX%\xelatex.exe" -synctex=1 -interaction=nonstopmode main.tex
if errorlevel 1 goto error
echo [V] Hoan tat bien dich!
goto end

:full
echo [1/3] XeLaTeX pass 1...
"%MIKTEX%\xelatex.exe" -synctex=1 -interaction=nonstopmode main.tex
if errorlevel 1 goto error
echo [2/3] Biber...
"%MIKTEX%\biber.exe" main
echo [3/3] XeLaTeX pass 2...
"%MIKTEX%\xelatex.exe" -synctex=1 -interaction=nonstopmode main.tex
if errorlevel 1 goto error
echo [V] Hoan tat bien dich day du!
goto end

:clean
echo [*] Dang xoa cac file tam...
del /s /q *.aux *.log *.toc *.lof *.lot *.out *.bbl *.blg *.bcf *.run.xml *.synctex.gz *.fls *.fdb_latexmk *.xdv >nul 2>&1
echo [V] Da don dep sach se!
goto end

:error
echo [!] Co loi xay ra, vui long kiem tra file log.

:end
endlocal
