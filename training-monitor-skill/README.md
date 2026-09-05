# Training Monitor Skill · 训练监控终端生成器

为任意深度学习训练任务一键生成带实时指标面板的 Electron 终端。

## 特性

- **伪终端**：xterm.js + node-pty，支持输入、复制、多标签页
- **实时指标**：通过正则从训练输出中提取 loss/lr/step 等任意指标
- **GPU 监控**：nvidia-smi 每 2 秒轮询，阈值告警
- **训练曲线**：loss/lr/gpu 等实时图表
- **缓冲解析**：不依赖换行符，Windows ConPTY 兼容
- **日志导出**：终端内容保存为文件
- **配置驱动**：一个 JSON 配置适配任意模型

## 快速开始

```bash
# 1. 复制示例配置
cp configs/aethermind_v4.json configs/my_model.json

# 2. 编辑配置：模型名、训练命令、指标正则、面板布局

# 3. 生成监控终端
python generator.py configs/my_model.json --output D:/my_monitor

# 4. 安装依赖并启动
cd D:/my_monitor
npm install
npm start
```

## 配置文件

```json
{
  "model_name": "MyModel",
  "window_title": "MyModel 训练监控",
  "train_command": ["train.cmd"],
  "project_root": "D:/my_project",
  "metrics": {
    "loss": {"regex": "loss=([\\d.]+)", "label": "Loss", "panel": "training"},
    "lr":   {"regex": "lr=([\\d.eE+-]+)", "label": "LR", "panel": "training"},
    "step": {"regex": "\\(\\s*(\\d+)\\s*/\\s*(\\d+)\\s*\\)", "label": "Step", "type": "step_total", "panel": "training"},
    "gpu":  {"type": "gpu", "label": "VRAM", "panel": "gpu"}
  },
  "charts": ["loss", "lr", "gpu"],
  "panels": ["training", "gpu"]
}
```

### 指标字段

| 字段 | 说明 |
|------|------|
| `regex` | 匹配该指标的正则，捕获组 1 是值 |
| `label` | 面板显示名（支持中英） |
| `type` | `step_total`（步/总数）、`gpu`（自动取显存） |
| `panel` | 归属面板：training/physics/validation/gpu/custom |
| `format` | `exp`（科学计数）、`fixed4`（4位小数） |

## 适配新模型

1. 跑一次训练，复制 5-10 行典型输出
2. 为每个指标写正则（在 regex101.com 验证）
3. 填入配置文件
4. 运行 generator.py 生成

## 环境要求

- Python 3.8+（仅生成器需要）
- Node.js 16+（运行 Electron 终端需要）
- NVIDIA GPU + nvidia-smi（GPU 监控，AMD 需自行修改）

## 目录结构

```
training-monitor-skill/
├── SKILL.md              # Skill 元数据与说明
├── README.md             # 本文件
├── generator.py          # 生成器核心
├── template/             # Electron 应用模板
│   ├── main.js.template
│   ├── preload.js
│   ├── package.json.template
│   └── renderer/
│       ├── index.html.template
│       ├── style.css
│       └── renderer.js.template
└── configs/
    └── aethermind_v4.json  # 示例配置
```

## License

MIT
