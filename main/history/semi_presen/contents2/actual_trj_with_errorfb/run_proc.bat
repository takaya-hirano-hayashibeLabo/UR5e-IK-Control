@echo off
setlocal

cd /d %~dp0
call C:\Users\3meko\miniconda3\Scripts\activate.bat
call conda activate mink-env

set TIMESCALE_LIST=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5
set MODELPATH=C:\Users\3meko\Dev\HayashibeLab\workspace\DynamicSNN\train-trajectory\output\20241024\circle_beta0.8_identity_noise0.05
set TRJPATH=C:\Users\3meko\Dev\HayashibeLab\workspace\ur5e_ik_control\main\collect_dataset\20241024\circle\output\datasets.csv


for %%t in (%TIMESCALE_LIST%) do (
    python get_actual_trj.py --modelpath %MODELPATH% --trjpath %TRJPATH% --timescale %%t --saveto 20241104_output/dynasnn --nloop 10
)

endlocal