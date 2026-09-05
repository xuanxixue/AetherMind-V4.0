@echo off
chcp 65001 >nul 2>&1
setlocal

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /d "%~dp0"

rem ============================================================
rem  Locate a Python interpreter that actually has torch.
rem  Prefer the known project interpreter C:\Python312, then
rem  fall back to python on PATH. Skip interpreters where
rem  `import torch` fails (e.g. embedded store stubs).
rem ============================================================
set PY=

if exist "C:\Python312\python.exe" (
  "C:\Python312\python.exe" -c "import torch" >nul 2>&1
  if not errorlevel 1 set "PY=C:\Python312\python.exe"
)

if "%PY%"=="" (
  where python >nul 2>&1
  if not errorlevel 1 (
    python -c "import torch" >nul 2>&1
    if not errorlevel 1 set "PY=python"
  )
)

if "%PY%"=="" (
  echo.
  echo ============================================================
  echo   [ERROR] No Python with torch found.
  echo.
  echo   Fix - install dependencies once, e.g.:
  echo     C:\Python312\python.exe -m pip install -r requirements.txt
  echo.
  echo   Or if torch is installed elsewhere, run directly:
  echo     "C:\Path\To\python.exe" scripts\inference_v4.py
  echo ============================================================
  echo.
  pause
  exit /b 1
)

echo [Python] %PY%
%PY% -c "import torch,sys; print('[torch]', torch.__version__, '| cuda:', torch.cuda.is_available())"

echo ============================================================
echo   AetherMind V4 Inference
echo   Arch: auto-detect (train / inference relative-kernel)
echo   Tokenizer: Qwen2.5-0.5B
echo   Commands: /learn /stats /consolidate /temp /max /reset /memory /kg /ref /rag /exit
echo   KG: daily_knowledge_graph.json + knowledge_graph.json (RAG-lite)
echo   Dialog: muice.jsonl (RAG dialogue /ref)
echo   Memory: Agent-style (compress+cache+retrieve, no truncation)
echo   RAG: high-confidence hit answers directly (/rag to toggle)
echo   Tip: run convert_train_to_inference.py first for inference arch
echo ============================================================
echo.

%PY% scripts\inference_v4.py --warmup 50 --kg_path "03_dialogue/daily_knowledge_graph.json,03_dialogue/knowledge_graph.json" --kg_topk 3 --dialogue_path "03_dialogue/muice-jsonl/muice.jsonl" --dialogue_topk 2 %*

echo.
pause
endlocal
