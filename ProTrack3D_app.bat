@echo off
rem Starts the ProTrack3D app on Windows. The first run creates the conda environment (several minutes).
setlocal
cd /d "%~dp0"
set "ENV_NAME=protrack3d-torch"
if not "%PROTRACK3D_ENV%"=="" set "ENV_NAME=%PROTRACK3D_ENV%"

where conda >nul 2>nul
if errorlevel 1 call :find_conda
where conda >nul 2>nul
if errorlevel 1 (
    echo conda was not found. Install Miniforge first: https://github.com/conda-forge/miniforge
    pause
    exit /b 1
)

call conda env list | findstr /b /c:"%ENV_NAME% " >nul
if errorlevel 1 call :setup
if errorlevel 1 (
    echo Setup failed.
    pause
    exit /b 1
)

rem Start the app without a console window.
for /f "delims=" %%P in ('conda run -n %ENV_NAME% python -c "import sys; print(sys.prefix)"') do set "ENV_DIR=%%P"
start "" "%ENV_DIR%\pythonw.exe" -W ignore -m protrack3d.app
exit /b 0

:find_conda
for %%D in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if exist "%%~D\condabin\conda.bat" set "PATH=%%~D\condabin;%PATH%"
)
exit /b 0

:setup
echo First run: setting up ProTrack3D. This takes several minutes and happens only once.
call conda create -y -n %ENV_NAME% python=3.11 pip
if errorlevel 1 exit /b 1
rem PyTorch from PyPI has no GPU support on Windows; with an NVIDIA card, take the CUDA build.
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo NVIDIA GPU found: installing the GPU build of PyTorch.
    call conda run -n %ENV_NAME% --no-capture-output pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 goto :undo
)
call conda run -n %ENV_NAME% --no-capture-output pip install -r requirements.txt
if errorlevel 1 goto :undo
exit /b 0

:undo
call conda env remove -y -n %ENV_NAME%
exit /b 1
