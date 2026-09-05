# AetherMind V4

物理机制增强的小型对话模型：信息素热力学注意力（Pheromone-Thermodynamic Attention）+
双域动力学 + 在线演化（LTP 固化）+ Agent 式记忆 + 知识图谱/对话检索增强（RAG-lite）。

- **参数量**：约 1.01 亿（d_model=384，6 层，上下文 256 token）
- **分词器**：Qwen2.5-0.5B tokenizer（离线打包，无需联网下载）
- **训练**：100000 步，Phase A→G 课程式训练（信息素沉积 → STP/LTP → 固化）
- **显存**：推理峰值约 1.3GB，4GB 显卡可跑；无显卡自动回退 CPU
- **许可**：MIT License

## 快速开始

1. 下载本仓库代码；
2. 到 [Releases](https://github.com/) 页面下载最新发布包
   `AetherMind-V4-Inference.zip`（约 1GB，内含模型权重），解压后把
   `checkpoints_v4_fixed/` 文件夹放到仓库根目录（与 `scripts/` 同级）；
   也可以直接使用解压出的完整发布包，无需再 clone 代码；
3. 安装依赖（仅首次）：

   ```bat
   pip install -r requirements.txt
   ```

4. Windows 双击 `run_inference.cmd` 进入交互对话；或命令行单次生成：

   ```bat
   python scripts\inference_v4.py --prompt "你好" --max_new 200
   ```

启动器会自动探测装有 torch 的 Python 解释器，并自动加载最新 checkpoint，
无需手动指定路径。

## 交互命令

| 命令 | 作用 |
|------|------|
| 直接输入文字 | 对话 |
| `/rag` | 开关检索直答（高置信命中参考对话时直接返回，默认开启） |
| `/kg` / `/ref` | 开关知识图谱 / 对话语料检索增强 |
| `/learn` | 开关在线演化（信息素沉积与 LTP 固化） |
| `/stats` | 查看信息素浓度、固化质量与固化轮数 |
| `/consolidate` | 手动触发一次固化 |
| `/temp 0.8` / `/max 200` | 采样温度 / 最大生成长度 |
| `/memory` | 查看对话记忆状态 |
| `/reset` | 清空对话记忆与信息素（长期固化保留） |
| `/exit` | 退出 |

## 工作原理（简述）

- **信息素热力学注意力**：注意力偏置由可演化的"信息素场" τ 产生，遵循
  Langévin 动力学（沉积 → 扩散 → 蒸发），训练中自发形成 STP（短期）→ LTP（长期）记忆；
- **双域架构**：快速域（对话/工作记忆）与慢速域（固化知识）按时间尺度分离；
- **在线演化**：推理时可持续沉积信息素并周期性固化，部署后仍能学习；
- **RAG-lite**：内置知识图谱（821 三元组）与 3737 条参考对话，高置信命中直接回答；
- **Agent 式记忆**：超长对话采用压缩 + 缓存 + 检索，不粗暴截断。

> 说明：本模型为 1 亿参数小模型，原生生成文本可能不够通顺（模型规模决定的能力上限），
> 日常使用建议保持检索直答开启。生成已在对话回合结束标记 `<eom>` 处正确停止。

## 仓库结构

```
├── run_inference.cmd          一键启动（Windows）
├── requirements.txt           Python 依赖
├── scripts/
│   ├── inference_v4.py        推理主程序
│   └── convert_train_to_inference.py  训练权重 → 推理架构转换（可选）
├── src/                       模型源码（注意力/双域/记忆/演化/训练）
├── configs/                   模型配置
├── models_local/              Qwen2.5 分词器（离线）
├── 03_dialogue/               检索数据（知识图谱 + 参考对话）
├── train_scripts/             各阶段训练启动脚本（A→G）
└── docs/                      架构报告与技术文档
```

模型权重（约 1.1GB）与训练语料（约 13GB）不在 git 仓库中：
权重见 Releases；训练语料不随发布包分发。

## 文档

- [推理发布使用说明](推理发布_使用说明.md)
- [项目结构说明](项目结构说明.md)
- `docs/` 目录下的架构报告与训练问题记录

## License

[MIT](LICENSE)
