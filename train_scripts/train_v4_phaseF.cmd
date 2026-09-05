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
echo   AetherMind V4.0 PHASE F - Knowledge Graph Alignment
echo   Load base: auto (PhaseE newest, else PhaseD/Final)
echo   Freeze 100%% backbone (Qwen/PGTA/Langevin/GMM means)
echo   Train only: Entity-Atom mapper + logic C/E assoc matrix
echo   KG: 03_dialogue\knowledge_graph.json
echo   Tokenizer: Qwen2.5-0.5B  d=384  L=6  h=6  seq=512
echo   phase_F=2000 steps
echo ============================================================
echo.
echo [i] Goal: inject structured facts into Logic domain atoms.
echo [i] Run after Phase E. Output: v4_checkpoint_*_phaseF_final.pt
echo [i] Then run run_inference.cmd to ask it factual questions.
echo.

C:\Python312\python.exe src\training\train_v4.py ^
  --phase_f ^
  --d_model 384 ^
  --n_layers 6 ^
  --n_heads 6 ^
  --max_seq 512 ^
  --phase_F 2000 ^
  --kg_path d:/AetherMind-Nano3/03_dialogue/knowledge_graph.json ^
  --log_interval 10 ^
  --save_interval 500 ^
  --eval_interval 999999 ^
  --output_dir d:/AetherMind-Nano3/checkpoints_v4_fixed ^
  --data_dir d:/AetherMind-Nano3/03_dialogue

echo.
echo ============================================================
echo   PHASE F DONE!
echo   Latest: checkpoints_v4_fixed\v4_checkpoint_*_phaseF_final.pt
echo   Run run_inference.cmd to check the result.
echo ============================================================
echo.
pause
endlocal
