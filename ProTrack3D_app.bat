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
    echo Setup failed. Check the internet connection and start ProTrack3D_app.bat again.
    pause
    exit /b 1
)

for /f "delims=" %%P in ('conda run -n %ENV_NAME% python -c "import sys; print(sys.prefix)"') do set "ENV_DIR=%%P"

rem An update made from inside the app may have left new packages to install; Windows cannot replace them while the app runs.
if exist ".protrack3d_update_pending" (
    echo Finishing an update: installing new packages.
    "%ENV_DIR%\python.exe" -m pip install -r requirements.txt
    if not errorlevel 1 del ".protrack3d_update_pending"
)

rem Check that the app loads before starting it without a window, so a problem is shown here and not swallowed.
"%ENV_DIR%\python.exe" -W ignore -c "import protrack3d.app.main_window"
if errorlevel 1 (
    echo.
    echo ProTrack3D could not start. The lines above say why.
    echo If they mention a missing module, delete the environment and start again:  conda env remove -n %ENV_NAME%
    pause
    exit /b 1
)

call :shortcut
start "" "%ENV_DIR%\pythonw.exe" -W ignore -m protrack3d.app
exit /b 0

:find_conda
for %%D in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if exist "%%~D\condabin\conda.bat" set "PATH=%%~D\condabin;%PATH%"
)
exit /b 0

:shortcut
rem A Start menu entry with the ProTrack3D icon, made once.
set "PT3D_LINK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\ProTrack3D.lnk"
set "PT3D_DIR=%~dp0"
if exist "%PT3D_LINK%" exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:PT3D_LINK); $s.TargetPath = Join-Path $env:PT3D_DIR 'ProTrack3D_app.bat'; $s.WorkingDirectory = $env:PT3D_DIR; $s.IconLocation = Join-Path $env:PT3D_DIR 'packaging\ProTrack3D.ico'; $s.WindowStyle = 7; $s.Description = 'ProTrack3D'; $s.Save()" >nul 2>nul
if exist "%PT3D_LINK%" echo Added ProTrack3D to the Start menu.
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
