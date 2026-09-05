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
echo   AetherMind V4.0 PHASE D - LTP Consolidation Training
echo   Load base: v4_checkpoint_35000_final.pt
echo   Freeze W, no backprop, evolve tau + quantile consolidation
echo   Tokenizer: Qwen2.5-0.5B
echo   d=384  L=6  h=6  seq=512
echo   phase_D=2000 (warmup 100, consolidate every 50, ~38 rounds)
echo ============================================================
echo.
echo [i] Standalone Phase D. No need to retrain from scratch.
echo [i] Output: v4_checkpoint_*_phaseD_final.pt
echo [i] Then run run_inference.cmd to load it and check the result.
echo.

C:\Python312\python.exe src\training\train_v4.py ^
  --phase_d ^
  --resume d:/AetherMind-Nano3/checkpoints_v4_fixed\v4_checkpoint_35000_final.pt ^
  --d_model 384 ^
  --n_layers 6 ^
  --n_heads 6 ^
  --max_seq 512 ^
  --phase_D 2000 ^
  --log_interval 10 ^
  --save_interval 500 ^
  --eval_interval 999999 ^
  --output_dir d:/AetherMind-Nano3/checkpoints_v4_fixed ^
  --data_dir d:/AetherMind-Nano3/03_dialogue

echo.
echo ============================================================
echo   PHASE D DONE!
echo   Latest: checkpoints_v4_fixed\v4_checkpoint_*_phaseD_final.pt
echo   Run run_inference.cmd to check the result.
echo ============================================================
echo.
pause
endlocal
