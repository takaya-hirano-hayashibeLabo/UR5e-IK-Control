@echo off
setlocal

cd /d %~dp0
call C:\Users\3meko\miniconda3\Scripts\activate.bat
call conda activate mink-env

set TIMESCALE_LIST=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10

for %%t in (%TIMESCALE_LIST%) do (
    python get_ideal_trj.py --configpath conf.yml --timescale %%t --saveto 20241031_output
)

endlocal