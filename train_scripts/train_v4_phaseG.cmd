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
echo   AetherMind V4.0 PHASE G - Clean Dialogue SFT
echo   Resume: step 2000 (v4_checkpoint_2000_phaseG.pt)
echo   Unfreeze last 3 decoder layers + final_norm + lm_head
echo   lr=3e-5, evolution/consolidation OFF
echo   Data: CLEANED MOSS + Muice (system prompt stripped)
echo   Tokenizer: Qwen2.5-0.5B  d=384  L=6  h=6  seq=512
echo   phase_G=20000 steps (full-data, ~2/3 epoch)
echo ============================================================
echo.
echo [i] Goal: teach the model to CHAT normally (clean dialogue).
echo [i] Run clean_dialogue_data.py first to produce 03_dialogue_clean.
echo [i] Output: v4_checkpoint_*_phaseG_final.pt
echo.

C:\Python312\python.exe src\training\train_v4.py ^
  --phase_g ^
  --d_model 384 ^
  --n_layers 6 ^
  --n_heads 6 ^
  --max_seq 512 ^
  --phase_G 20000 ^
  --phaseG_lr 3e-5 ^
  --unfreeze_layers 3 ^
  --resume d:/AetherMind-Nano3/checkpoints_v4_fixed/v4_checkpoint_2000_phaseG.pt ^
  --log_interval 10 ^
  --save_interval 2000 ^
  --eval_interval 999999 ^
  --output_dir d:/AetherMind-Nano3/checkpoints_v4_fixed ^
  --phaseG_data_dir d:/AetherMind-Nano3/03_dialogue_clean

echo.
echo ============================================================
echo   PHASE G DONE!
echo   Latest: checkpoints_v4_fixed\v4_checkpoint_*_phaseG_final.pt
echo   Run run_inference.cmd to chat with it.
echo ============================================================
echo.
pause
endlocal