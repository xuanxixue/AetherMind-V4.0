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
echo   AetherMind V4.0 PHASE E - Language Organization SFT
echo   Load base: auto (PhaseD newest, else Final)
echo   Unfreeze last 2 decoder layers + final_norm + lm_head
echo   Low lr=1e-5, evolution/consolidation OFF
echo   Data: MOSS + Muice(Muice-Dataset) under 03_dialogue
echo   Tokenizer: Qwen2.5-0.5B  d=384  L=6  h=6  seq=512
echo   phase_E=3000 steps
echo ============================================================
echo.
echo [i] Goal: teach the model to ORGANIZE language (no KG yet).
echo [i] Run after Phase D. Output: v4_checkpoint_*_phaseE_final.pt
echo [i] Then run train_v4_phaseF.cmd for KG alignment.
echo.

C:\Python312\python.exe src\training\train_v4.py ^
  --phase_e ^
  --d_model 384 ^
  --n_layers 6 ^
  --n_heads 6 ^
  --max_seq 512 ^
  --phase_E 3000 ^
  --phaseE_lr 1e-5 ^
  --unfreeze_layers 2 ^
  --log_interval 10 ^
  --save_interval 500 ^
  --eval_interval 999999 ^
  --output_dir d:/AetherMind-Nano3/checkpoints_v4_fixed ^
  --data_dir d:/AetherMind-Nano3/03_dialogue

echo.
echo ============================================================
echo   PHASE E DONE!
echo   Latest: checkpoints_v4_fixed\v4_checkpoint_*_phaseE_final.pt
echo   Run run_inference.cmd to hear it speak.
echo ============================================================
echo.
pause
endlocal
