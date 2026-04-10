@echo off

call:pull_submodules
call:create_venv
call:create_exec

:create_exec
(
    echo @echo off
    echo setlocal
    echo call activate 
    echo set "MOVERSCORE_MODEL=neuralmind/bert-large-portuguese-cased"
    echo python src\run.py
    echo endlocal
) > run.bat
exit /b

:create_venv
python -m venv .venv
call :create_activate
call activate
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
exit /b

:create_activate
(
    echo @echo off
    echo .venv\Scripts\activate 
) > activate.bat
exit /b

:pull_submodules
git submodule init
git submodule update
call:fix_moverscore
exit /b

:fix_moverscore
set "source=src\external\moverscore\moverscore_v2.py"
set "temp=src\external\moverscore\moverscore_v2.temp"
rename %source% "moverscore_v2.temp"
setlocal enableDelayedExpansion
(
   for /F "tokens=1* delims=:" %%a in ('findstr /N "^" %temp%') do (
      set "line=%%b"
      if defined line set "line=!line:np.float=float!"
      if defined line set "line=!line:float32=float!"
      echo(!line!)
   ) > %source%
)
endlocal
del %temp%
exit /b