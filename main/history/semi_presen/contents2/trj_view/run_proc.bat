@echo off
setlocal

cd /d %~dp0
call C:\Users\3meko\miniconda3\Scripts\activate.bat
call conda activate mink-env

set TIMESCALE_LIST=0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 2.00 3.00 4.00 5.00

set TRJPATH=C:\Users\3meko\Dev\HayashibeLab\workspace\ur5e_ik_control\main\history\semi_presen\contents2\actual_trj\20241031_output
set MODELTYPELIST=dynasnn paramsnn

for %%t in (%TIMESCALE_LIST%) do (
    for %%m in (%MODELTYPELIST%) do (
        @REM echo python trj_view.py --trjpath %TRJPATH%/%%m/trajectory_timescale_%%t
        python trj_view.py --trjpath %TRJPATH%/%%m/trajectory_timescale%%t
    )
)
