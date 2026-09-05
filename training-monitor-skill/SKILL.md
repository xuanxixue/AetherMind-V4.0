---
name: training-monitor
description: 为任意深度学习训练任务快速生成 Electron 监控终端。支持 xterm.js 伪终端、实时指标面板、GPU 显存监控、训练曲线、多标签页、日志导出。当用户需要为某个模型/训练脚本搭建可视化监控终端时使用。
---

# Training Monitor Generator

为任意训练脚本一键生成带实时指标面板的 Electron 终端。

## 快速使用

```bash
# 1. 复制示例配置，改成你的模型
cp configs/aethermind_v4.json configs/my_model.json

# 2. 编辑配置：模型名、训练命令、指标正则、面板布局
#    （关键：提供几行训练输出样例，正则就能照着写）

# 3. 生成终端
python generator.py configs/my_model.json --output D:/my_monitor

# 4. 安装依赖并启动
cd D:/my_monitor && npm install && npm start
```

## 配置文件格式

```json
{
  "model_name": "MyModel",
  "window_title": "MyModel 训练监控",
  "train_command": ["train.cmd"],
  "project_root": "D:/my_project",
  "metrics": {
    "loss":   {"regex": "loss=([\\d.]+)", "label": "Loss", "panel": "training"},
    "lr":     {"regex": "lr=([\\d.eE+-]+)", "label": "LR", "panel": "training"},
    "step":   {"regex": "\\(\\s*(\\d+)\\s*/\\s*(\\d+)\\s*\\)", "label": "Step", "type": "step_total", "panel": "training"},
    "phase":  {"regex": "ph=([A-Z])", "label": "Phase", "panel": "training"},
    "gpu":    {"type": "gpu", "label": "VRAM", "panel": "gpu"}
  },
  "charts": ["loss", "lr", "gpu"],
  "panels": ["training", "gpu"]
}
```

### metric 字段说明

| 字段 | 说明 |
|------|------|
| `regex` | 匹配该指标的正则，捕获组 1 是值（step_total 类型用组 1=当前, 组 2=总数） |
| `label` | 面板显示名（支持中英，如 "Loss / 损失"） |
| `type` | 可选：`step_total`（步/总数）、`gpu`（自动从 nvidia-smi 取） |
| `panel` | 归属哪个面板（training/gpu/custom） |

### 内置能力（无需配置）

- 伪终端：xterm.js + node-pty，支持输入、复制、多标签页
- GPU 监控：nvidia-smi 每 2 秒轮询，阈值告警
- 训练曲线：loss/lr/gpu 等实时图表
- 日志导出：终端内容保存为文件
- 缓冲解析：不依赖换行符，ConPTY 兼容

## 适配新模型的步骤

1. 跑一次训练，复制 5-10 行典型输出
2. 为每个想监控的指标写正则（在 https://regex101.com 验证）
3. 填入配置文件的 metrics
4. 运行 generator.py 生成
5. npm install && npm start

## 注意

- Windows 下 ConPTY 会吃掉 `\r`，生成器默认用缓冲解析（取最后 4000 字符匹配），无需担心
- GPU 监控依赖 nvidia-smi，AMD 卡需自行修改 queryGpuMemory
- 首次运行需 npm install（electron + xterm + node-pty）
