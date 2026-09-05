#!/usr/bin/env python3
"""
Training Monitor Generator
根据 JSON 配置生成定制化的 Electron 训练监控终端。

用法: python generator.py config.json --output D:/my_monitor
"""
import json
import os
import sys
import shutil
import argparse

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SKILL_DIR, 'template')

# 面板颜色映射
PANEL_COLORS = {
    'training': 'panel-cyan',
    'physics': 'panel-orange',
    'validation': 'panel-purple',
    'gpu': 'panel-red',
    'custom': 'panel-green',
}

CHART_COLORS = ['#00e5ff', '#56d364', '#ffa657', '#b388ff', '#ff7b72', '#ffeb3b', '#1f6feb']


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def gen_metrics_regex(metrics):
    """生成 main.js 中的 RE 对象"""
    lines = []
    for key, cfg in metrics.items():
        if cfg.get('type') == 'gpu':
            continue  # GPU 不需要正则
        regex = cfg.get('regex', '')
        if not regex:
            continue
        # JSON 中的 \\ 已被 json.load 转为 \，直接写入 JS 正则字面量即可
        lines.append(f'  {key}: /{regex}/,')
    return '\n'.join(lines)


def gen_curve_keys(charts):
    """生成 curveData 初始化的键"""
    return ', '.join(f'{c}: []' for c in charts)


def gen_curve_push(charts):
    """生成 pushCurvePoint 中的 push 语句"""
    lines = []
    for c in charts:
        lines.append(f'  curveData.{c}.push(metrics.{c} || 0);')
    return '\n'.join(lines)


def gen_metric_config(metrics):
    """生成 renderer.js 中的 METRICS 数组"""
    items = []
    for key, cfg in metrics.items():
        item = {'key': key, 'label': cfg.get('label', key)}
        if cfg.get('type'):
            item['type'] = cfg['type']
        if cfg.get('format'):
            item['format'] = cfg['format']
        if cfg.get('color'):
            item['color'] = cfg['color']
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=2)


def gen_chart_config(charts):
    """生成 renderer.js 中的 CHARTS 数组"""
    items = []
    for i, c in enumerate(charts):
        items.append({'id': c, 'color': CHART_COLORS[i % len(CHART_COLORS)]})
    return json.dumps(items, ensure_ascii=False, indent=2)


def gen_panel_html(metrics, panels):
    """生成 index.html 中的面板 HTML"""
    # 按 panel 分组
    grouped = {}
    for key, cfg in metrics.items():
        panel = cfg.get('panel', 'custom')
        if panel not in grouped:
            grouped[panel] = []
        grouped[panel].append((key, cfg))

    html_parts = []
    for panel_name in panels:
        items = grouped.get(panel_name, [])
        color_class = PANEL_COLORS.get(panel_name, 'panel-green')
        title = panel_name.capitalize()
        html_parts.append(f'<div class="panel {color_class}">')
        html_parts.append(f'  <div class="panel-title">{title}</div>')
        for key, cfg in items:
            label = cfg.get('label', key)
            if key == 'step':
                html_parts.append(f'  <div class="metric-row"><span class="metric-label">{label}</span><span class="metric-val" id="m-step">0 / 0</span></div>')
                html_parts.append(f'  <div class="metric-row"><span class="metric-label">进度 Progress</span><span class="metric-val" id="m-pct">0%</span></div>')
                html_parts.append(f'  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>')
            elif cfg.get('type') == 'gpu':
                html_parts.append(f'  <div class="metric-row"><span class="metric-label">{label}</span><span class="metric-val" id="m-gpu">0/0 MB</span></div>')
                html_parts.append(f'  <div class="metric-row"><span class="metric-label">占用 Usage</span><span class="metric-val" id="m-gpupct">0%</span></div>')
                html_parts.append(f'  <div class="metric-row"><span class="metric-label">状态 Status</span><span class="metric-val val-green" id="m-gpustatus">SAFE</span></div>')
                html_parts.append(f'  <div class="gpu-bar"><div class="gpu-bar-fill" id="gpu-bar-fill"></div></div>')
            else:
                html_parts.append(f'  <div class="metric-row"><span class="metric-label">{label}</span><span class="metric-val" id="m-{key}">?</span></div>')
        html_parts.append('</div>')
    return '\n'.join(html_parts)


def gen_chart_html(charts):
    """生成 index.html 中的图表 HTML"""
    parts = []
    for c in charts:
        parts.append(f'<div class="chart-box" id="chart-{c}"><canvas></canvas><div class="chart-label">{c}</div></div>')
    return '\n'.join(parts)


def generate(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'renderer'), exist_ok=True)

    model_name = config.get('model_name', 'Model')
    window_title = config.get('window_title', f'{model_name} Monitor')
    project_root = config.get('project_root', os.getcwd())
    train_cmd = config.get('train_command', ['train.py'])
    metrics = config.get('metrics', {})
    charts = config.get('charts', ['loss', 'lr', 'gpu'])
    panels = config.get('panels', ['training', 'gpu'])

    # 确保 gpu 指标存在
    if 'gpu' not in metrics:
        metrics['gpu'] = {'type': 'gpu', 'label': 'VRAM', 'panel': 'gpu'}
    if 'gpu' not in panels:
        panels.append('gpu')
    if 'gpu' not in charts:
        charts.append('gpu')

    replacements = {
        '{{MODEL_NAME}}': model_name,
        '{{MODEL_NAME_SLUG}}': model_name.lower().replace(' ', '-').replace('.', ''),
        '{{WINDOW_TITLE}}': window_title,
        '{{PROJECT_ROOT}}': project_root.replace('\\', '/'),
        '{{TRAIN_CMD}}': json.dumps(train_cmd),
        '{{METRICS_REGEX}}': gen_metrics_regex(metrics),
        '{{CURVE_KEYS}}': gen_curve_keys(charts),
        '{{CURVE_PUSH}}': gen_curve_push(charts),
        '{{METRIC_CONFIG}}': gen_metric_config(metrics),
        '{{CHART_CONFIG}}': gen_chart_config(charts),
        '{{PANEL_HTML}}': gen_panel_html(metrics, panels),
        '{{CHART_HTML}}': gen_chart_html(charts),
    }

    # 生成文件
    files = {
        'main.js.template': 'main.js',
        'preload.js': 'preload.js',
        'package.json.template': 'package.json',
        'renderer/index.html.template': 'renderer/index.html',
        'renderer/style.css': 'renderer/style.css',
        'renderer/renderer.js.template': 'renderer/renderer.js',
    }

    for tpl_name, out_name in files.items():
        tpl_path = os.path.join(TEMPLATE_DIR, tpl_name)
        out_path = os.path.join(output_dir, out_name)
        with open(tpl_path, 'r', encoding='utf-8') as f:
            content = f.read()
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, str(value))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

    # 生成启动脚本
    start_cmd = os.path.join(output_dir, '启动监控.cmd')
    with open(start_cmd, 'w', encoding='utf-8') as f:
        f.write('@echo off\nchcp 65001 >nul\ncd /d "%~dp0"\n')
        f.write('if not exist "node_modules" (\n  echo [i] 首次运行，安装依赖...\n  npm install\n)\n')
        f.write('echo [i] 启动监控中心...\nnpm start\npause\n')

    print(f'[OK] Generated at: {output_dir}')
    print(f'     Model: {model_name}')
    print(f'     Metrics: {len(metrics)}')
    print(f'     Charts: {len(charts)}')
    print(f'     Panels: {panels}')
    print(f'     Next: cd {output_dir} && npm install && npm start')
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Training Monitor Generator')
    parser.add_argument('config', help='Path to config JSON')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    generate(config, args.output)


if __name__ == '__main__':
    main()
