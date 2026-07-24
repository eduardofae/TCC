@echo off
uv sync
call:pull_submodules
call:create_exec

:create_exec
(
   echo @echo off
   echo uv run src\run.py
) > run.bat
exit /b

:pull_submodules
git submodule init
git submodule update
call:fix_moverscore
call:fix_bartscore
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
      if defined line set "line=!line:model_name, o=model_name, attn_implementation="eager", o!"
      echo(!line!)
   ) > %source%
)
endlocal
del %temp%
exit /b

:fix_bartscore
set "source=src\external\BARTScore\bart_score.py"
set "temp=src\external\BARTScore\bart_score.temp"
rename %source% "bart_score.temp"
setlocal enableDelayedExpansion
(
   for /F "tokens=1* delims=:" %%a in ('findstr /N "^" %temp%') do (
      set "line=%%b"
      if defined line set "line=!line:BartTokenizer=RobertaTokenizer!"
      echo(!line!)
   ) > %source%
)
endlocal
del %temp%
exit /b