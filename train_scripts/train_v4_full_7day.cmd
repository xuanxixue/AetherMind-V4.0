@echo off
setlocal

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set CUDA_LAUNCH_BLOCKING=0

cd /d "%~dp0.."

echo ============================================================
echo   AetherMind V4.0  7-Day Full Training Pipeline
echo ============================================================
echo   Phase A+B+C: from scratch (dual-weight evolution, 140K)
echo   Phase D:     LTP consolidation (freeze W, 5K)
echo   Phase E:     language SFT (unfreeze 2 layers, 3K)
echo   Phase F:     KG alignment (freeze trunk, 2K)
echo   Phase G:     clean dialogue SFT (6 layers, 100K, lr=2e-5)
echo ============================================================
echo   Validation: every 500 steps numeric + physics ratio
echo               every 5000 steps migration KL check
echo   Logs: checkpoints_v4_fixed/validation_logs/
echo   Estimated: ~6.5-7 days
echo ============================================================
echo.

set PY=C:\Python312\python.exe
set TRAIN=src\training\train_v4.py
set OUT=d:/AetherMind-Nano3/checkpoints_v4_fixed
set DATA=d:/AetherMind-Nano3/03_dialogue
set CLEAN_DATA=d:/AetherMind-Nano3/03_dialogue_clean
set KG=d:/AetherMind-Nano3/03_dialogue/knowledge_graph.json
set MAX_RETRY=5

REM ===== Phase A+B+C =====
:phase_abc
echo.
echo [%date% %time%] ===== PHASE A+B+C START (140K steps) =====
echo.

%PY% %TRAIN% --d_model 384 --n_layers 6 --n_heads 6 --max_seq 256 --batch_size 1 --grad_accum 8 --lr 3e-4 --phase_A 20000 --phase_B 40000 --phase_C 80000 --log_interval 50 --eval_interval 5000 --save_interval 2000 --output_dir %OUT% --data_dir %DATA%

REM check final checkpoint exists; if not, treat as crash
if not exist "%OUT%\v4_checkpoint_140000.pt" (
    echo [%date% %time%] Phase A+B+C: final checkpoint NOT found, treating as crash
    goto :phase_abc_retry
)
echo [%date% %time%] ===== PHASE A+B+C DONE =====
echo.
goto :phase_d

:phase_abc_retry
set /a ABC_RETRY+=1
if %ABC_RETRY% geq %MAX_RETRY% (
    echo [%date% %time%] Phase A+B+C failed after %MAX_RETRY% retries, ABORTING
    goto :all_fail
)
echo [%date% %time%] Phase A+B+C crashed (retry %ABC_RETRY%/%MAX_RETRY%), cleaning up...
taskkill /F /IM python.exe /T 2>nul
timeout /t 30 /nobreak >nul
echo [%date% %time%] Restarting...
goto :phase_abc

REM ===== Phase D =====
:phase_d
echo [%date% %time%] ===== PHASE D START (consolidation, 5K) =====
echo.

if not exist "%OUT%\v4_checkpoint_140000.pt" (
    echo [%date% %time%] ERROR: Phase D base checkpoint missing, aborting
    goto :all_fail
)

%PY% %TRAIN% --phase_d --resume %OUT%/v4_checkpoint_140000.pt --d_model 384 --n_layers 6 --n_heads 6 --max_seq 256 --phase_D 5000 --log_interval 10 --save_interval 1000 --eval_interval 999999 --output_dir %OUT% --data_dir %DATA%

if not exist "%OUT%\v4_checkpoint_5000_phaseD_final.pt" (
    echo [%date% %time%] Phase D: final checkpoint NOT found, treating as crash
    goto :phase_d_retry
)
echo [%date% %time%] ===== PHASE D DONE =====
echo.
goto :phase_e

:phase_d_retry
set /a D_RETRY+=1
if %D_RETRY% geq %MAX_RETRY% (
    echo [%date% %time%] Phase D failed after %MAX_RETRY% retries, ABORTING
    goto :all_fail
)
echo [%date% %time%] Phase D crashed (retry %D_RETRY%/%MAX_RETRY%)
taskkill /F /IM python.exe /T 2>nul
timeout /t 30 /nobreak >nul
goto :phase_d

REM ===== Phase E =====
:phase_e
echo [%date% %time%] ===== PHASE E START (language SFT, 3K) =====
echo.

%PY% %TRAIN% --phase_e --resume %OUT%/v4_checkpoint_5000_phaseD_final.pt --d_model 384 --n_layers 6 --n_heads 6 --max_seq 256 --phase_E 3000 --phaseE_lr 1e-5 --unfreeze_layers 2 --log_interval 10 --save_interval 500 --eval_interval 999999 --output_dir %OUT% --data_dir %DATA%

if not exist "%OUT%\v4_checkpoint_3000_phaseE_final.pt" goto :phase_e_retry
echo [%date% %time%] ===== PHASE E DONE =====
echo.
goto :phase_f

:phase_e_retry
echo [%date% %time%] Phase E crashed, cleaning up...
taskkill /F /IM python.exe /T 2>nul
timeout /t 30 /nobreak >nul
goto :phase_e

REM ===== Phase F =====
:phase_f
echo [%date% %time%] ===== PHASE F START (KG alignment, 2K) =====
echo.

%PY% %TRAIN% --phase_f --resume %OUT%/v4_checkpoint_3000_phaseE_final.pt --d_model 384 --n_layers 6 --n_heads 6 --max_seq 256 --phase_F 2000 --kg_path %KG% --log_interval 10 --save_interval 500 --eval_interval 999999 --output_dir %OUT% --data_dir %DATA%

if not exist "%OUT%\v4_checkpoint_2000_phaseF_final.pt" goto :phase_f_retry
echo [%date% %time%] ===== PHASE F DONE =====
echo.
goto :phase_g

:phase_f_retry
echo [%date% %time%] Phase F crashed, cleaning up...
taskkill /F /IM python.exe /T 2>nul
timeout /t 30 /nobreak >nul
goto :phase_f

REM ===== Phase G =====
:phase_g
echo [%date% %time%] ===== PHASE G START (clean dialogue SFT, 100K, lr=2e-5) =====
echo.

%PY% %TRAIN% --phase_g --d_model 384 --n_layers 6 --n_heads 6 --max_seq 256 --phase_G 100000 --phaseG_lr 2e-5 --unfreeze_layers 6 --resume %OUT%/v4_checkpoint_2000_phaseF_final.pt --log_interval 50 --save_interval 2000 --eval_interval 999999 --output_dir %OUT% --phaseG_data_dir %CLEAN_DATA%

if not exist "%OUT%\v4_checkpoint_100000_phaseG_final.pt" goto :phase_g_retry
echo [%date% %time%] ===== PHASE G DONE =====
echo.
goto :all_done

:phase_g_retry
echo [%date% %time%] Phase G crashed, cleaning up...
taskkill /F /IM python.exe /T 2>nul
timeout /t 30 /nobreak >nul
goto :phase_g

:all_fail
echo ============================================================
echo   TRAINING FAILED - check logs above
echo ============================================================
pause
exit /b 1

:all_done
echo ============================================================
echo   ALL 5 PHASES COMPLETE!
echo   Final: %OUT%/v4_checkpoint_*_phaseG_final.pt
echo   Reports: %OUT%/validation_logs/
echo ============================================================
echo.
pause
endlocal
