@echo off
chcp 65001 >nul 2>&1
setlocal

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
set CUDA_LAUNCH_BLOCKING=0

cd /d "%~dp0.."

echo ============================================================
echo   AetherMind V4.0 FRESH TRAINING with GPU GUARD
echo   Start from scratch (--fresh)
echo   Auto-restart on crash (with GPU cleanup + auto-resume)
echo   Tokenizer: Qwen2.5-0.5B (151680 tokens)
echo   d=384  L=6  h=6  seq=512
echo   batch=1  grad_accum=16  lr=3e-4
echo   Phase A=5000  B=10000  C=20000
echo ============================================================
echo.
echo [!] FRESH MODE: Will start training from step 0
echo [!] Old checkpoints in checkpoints_v4_fixed will remain
echo.

:restart
echo [%date% %time%] Starting training...
echo.

C:\Python312\python.exe src\training\train_v4.py ^
  --fresh ^
  --d_model 384 ^
  --n_layers 6 ^
  --n_heads 6 ^
  --max_seq 512 ^
  --batch_size 1 ^
  --grad_accum 16 ^
  --lr 3e-4 ^
  --phase_A 5000 ^
  --phase_B 10000 ^
  --phase_C 20000 ^
  --log_interval 10 ^
  --eval_interval 2000 ^
  --save_interval 2000 ^
  --output_dir d:/AetherMind-Nano3/checkpoints_v4_fixed ^
  --data_dir d:/AetherMind-Nano3/03_dialogue

if %ERRORLEVEL% EQU 0 (
  echo.
  echo ============================================================
  echo   TRAINING COMPLETED SUCCESSFULLY!
  echo ============================================================
  goto :end
)

echo.
echo ============================================================
echo   TRAINING CRASHED (exit code %ERRORLEVEL%)
echo   Waiting 15 seconds for GPU memory to free...
echo   Will auto-restart and resume from latest checkpoint...
echo ============================================================
echo.
timeout /t 15 /nobreak >nul
echo Restarting...
goto :restart

:end
endlocal
pause
