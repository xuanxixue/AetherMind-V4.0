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
echo   AetherMind V4.0 PHASE G v3 - Clean Dialogue SFT (FIXED)
echo   Base: Phase D final (best language state, loss~4.1)
echo   Fix: label mask applied (only assistant reply in loss)
echo   Unfreeze ALL 6 decoder layers + final_norm + lm_head
echo   lr=2e-5 peak, warmup=10%% + cosine decay
echo   Data: CLEANED MOSS + Muice (03_dialogue_clean)
echo   phase_G=30000 steps
echo ============================================================
echo.
echo [i] CRITICAL FIX: dataset.py now masks human input tokens.
echo [i] v2 issue: lr=5e-6 + 10000 steps, entire run was warmup.
echo [i] v3 fix: lr=2e-5 (6x), 30000 steps, all 6 layers unfrozen.
echo [i] Scheduler: 3000-step linear warmup + 27000-step cosine decay.
echo [i] Starting fresh from Phase D to avoid corrupted weights.
echo.

C:\Python312\python.exe src\training\train_v4.py ^
  --phase_g ^
  --d_model 384 ^
  --n_layers 6 ^
  --n_heads 6 ^
  --max_seq 512 ^
  --phase_G 30000 ^
  --phaseG_lr 2e-5 ^
  --unfreeze_layers 6 ^
  --resume d:/AetherMind-Nano3/checkpoints_v4_fixed/v4_checkpoint_2000_phaseD_final.pt ^
  --log_interval 50 ^
  --save_interval 1000 ^
  --eval_interval 999999 ^
  --output_dir d:/AetherMind-Nano3/checkpoints_v4_fixed ^
  --phaseG_data_dir d:/AetherMind-Nano3/03_dialogue_clean

echo.
echo ============================================================
echo   PHASE G v3 DONE!
echo   Convert: python scripts/convert_train_to_inference.py
echo   Chat: run_inference.cmd
echo ============================================================
echo.
pause
endlocal
