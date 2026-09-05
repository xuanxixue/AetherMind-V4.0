"""
AetherMind V4 训练监控终端 (TUI)
- 启动训练子进程，实时捕获输出
- 解析关键指标 (loss/lr/tau/cons/physics)
- 独立监控GPU显存，超阈值告警
- 所有输出同时写入日志文件
- 纯终端渲染，零显存开销（不碰CUDA）
"""
import os
import sys
import re
import time
import threading
import subprocess
import queue
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = r"D:\AetherMind-Nano3"
PYTHON = r"C:\Python312\python.exe"
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "training", "train_v4.py")
LOG_DIR = os.path.join(PROJECT_ROOT, "training_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 显存阈值
MEM_WARN = 0.80   # 黄色告警
MEM_CRIT = 0.92   # 红色危险

# 指标正则
RE_LOSS = re.compile(r'loss=([\d.]+)')
RE_LR = re.compile(r'lr=([\d.eE+-]+)')
RE_PHASE = re.compile(r'ph=([A-Z])|Phase ([A-Z])')
RE_STEP = re.compile(r'\((\d+)/(\d+)\)|(\d+)/(\d+)')
RE_TAU = re.compile(r'tau=([\d.]+)')
RE_CONS = re.compile(r'cons=([\d.]+)')
RE_PHYSICS = re.compile(r'physics=([\d.]+)')
RE_DOMINANCE = re.compile(r'->\s*(\S+)')
RE_BAD = re.compile(r'bad=(\d+)')
RE_SPS = re.compile(r'([\d.]+)\s*step/s')


class Metrics:
    """训练指标状态"""
    def __init__(self):
        self.phase = "?"
        self.step = 0
        self.total = 0
        self.loss = 0.0
        self.lr = 0.0
        self.tau = 0.0
        self.cons = 0.0
        self.physics_ratio = 0.0
        self.dominance = "?"
        self.bad = 0
        self.sps = 0.0
        self.gpu_used = 0
        self.gpu_total = 4096
        self.gpu_pct = 0.0
        self.watchdog_clears = 0
        self.watchdog_emergency = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.log_lines = []
        self.max_log_lines = 200

    def add_log(self, line: str):
        self.log_lines.append(line)
        if len(self.log_lines) > self.max_log_lines:
            self.log_lines = self.log_lines[-self.max_log_lines:]
        self.last_update = time.time()


def parse_line(line: str, m: Metrics):
    """从训练输出行解析指标"""
    # loss
    mo = RE_LOSS.search(line)
    if mo:
        try:
            m.loss = float(mo.group(1))
        except ValueError:
            pass
    # lr
    mo = RE_LR.search(line)
    if mo:
        try:
            m.lr = float(mo.group(1))
        except ValueError:
            pass
    # phase
    mo = RE_PHASE.search(line)
    if mo:
        m.phase = mo.group(1) or mo.group(2) or m.phase
    # step/total
    mo = RE_STEP.search(line)
    if mo:
        try:
            if mo.group(1):
                m.step = int(mo.group(1))
                m.total = int(mo.group(2))
            elif mo.group(3):
                m.step = int(mo.group(3))
                m.total = int(mo.group(4))
        except (ValueError, IndexError):
            pass
    # tau
    mo = RE_TAU.search(line)
    if mo:
        try:
            m.tau = float(mo.group(1))
        except ValueError:
            pass
    # cons
    mo = RE_CONS.search(line)
    if mo:
        try:
            m.cons = float(mo.group(1))
        except ValueError:
            pass
    # physics ratio
    mo = RE_PHYSICS.search(line)
    if mo:
        try:
            m.physics_ratio = float(mo.group(1))
        except ValueError:
            pass
    # dominance
    if '->' in line and ('dominant' in line or 'mixed' in line or 'significant' in line):
        mo = RE_DOMINANCE.search(line)
        if mo:
            m.dominance = mo.group(1)
    # bad
    mo = RE_BAD.search(line)
    if mo:
        try:
            m.bad = int(mo.group(1))
        except ValueError:
            pass
    # sps
    mo = RE_SPS.search(line)
    if mo:
        try:
            m.sps = float(mo.group(1))
        except ValueError:
            pass


def get_gpu_memory():
    """独立查询GPU显存（不依赖PyTorch，用nvidia-smi）"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=5
        )
        raw = result.stdout
        for enc in ('utf-8', 'gbk', 'latin-1'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode('latin-1', errors='ignore')
        parts = text.strip().split(',')
        if len(parts) >= 2:
            used = int(parts[0].strip())
            total = int(parts[1].strip())
            return used, total
    except Exception:
        pass
    return 0, 4096


def gpu_monitor_thread(m: Metrics, stop_event: threading.Event):
    """后台线程：每3秒查询GPU显存"""
    while not stop_event.is_set():
        used, total = get_gpu_memory()
        m.gpu_used = used
        m.gpu_total = total
        m.gpu_pct = used / total if total > 0 else 0
        stop_event.wait(3)


def build_dashboard(m: Metrics) -> Panel:
    """构建TUI仪表盘"""
    # 显存状态颜色
    if m.gpu_pct >= MEM_CRIT:
        mem_color = "red"
        mem_status = "CRITICAL"
    elif m.gpu_pct >= MEM_WARN:
        mem_color = "yellow"
        mem_status = "WARNING"
    else:
        mem_color = "green"
        mem_status = "SAFE"

    # 进度
    pct = (m.step / m.total * 100) if m.total > 0 else 0
    elapsed = time.time() - m.start_time
    eta_str = "?"
    if m.sps > 0 and m.total > 0:
        remaining = (m.total - m.step) / m.sps
        if remaining < 3600:
            eta_str = f"{remaining/60:.0f}min"
        else:
            eta_str = f"{remaining/3600:.1f}h"

    # 物理占比颜色
    if m.physics_ratio > 0.30:
        phys_color = "green"
    elif m.physics_ratio > 0.05:
        phys_color = "yellow"
    else:
        phys_color = "red"

    # 显存条
    bar_len = 30
    filled = int(bar_len * m.gpu_pct)
    mem_bar = "█" * filled + "░" * (bar_len - filled)

    # 指标表
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column(justify="left")

    table.add_row("Phase:", f"[cyan bold]{m.phase}[/cyan bold]")
    table.add_row("Step:", f"{m.step:,} / {m.total:,}  ({pct:.1f}%)")
    table.add_row("Loss:", f"[bold]{m.loss:.4f}[/bold]")
    table.add_row("LR:", f"{m.lr:.2e}")
    table.add_row("τ浓度:", f"{m.tau:.1f}")
    table.add_row("固化量:", f"{m.cons:,.0f}")
    table.add_row("物理占比:", f"[{phys_color}]{m.physics_ratio:.3f} ({m.dominance})[/{phys_color}]")
    table.add_row("Bad batch:", f"{m.bad}")
    table.add_row("速度:", f"{m.sps:.1f} step/s  ETA {eta_str}")
    table.add_row("耗时:", f"{elapsed/3600:.1f}h")

    # GPU 面板
    gpu_text = Text()
    gpu_text.append(f"GPU显存: {m.gpu_used}/{m.gpu_total}MB ({m.gpu_pct:.0%})\n", style="bold")
    gpu_text.append(f"[{mem_color}]{mem_bar}[/{mem_color}]  ", style="bold")
    gpu_text.append(f"[{mem_color}]{mem_status}[/{mem_color}]\n")
    gpu_text.append(f"守护线程: 常规清理 {m.watchdog_clears}次 / 紧急 {m.watchdog_emergency}次")

    # 日志
    log_text = Text()
    for line in m.log_lines[-15:]:
        # 给不同类型的日志上色
        if '[VAL]' in line:
            log_text.append(line + "\n", style="cyan")
        elif 'WARNING' in line or 'WARN' in line:
            log_text.append(line + "\n", style="yellow")
        elif 'ERROR' in line or 'OOM' in line or 'CRITICAL' in line:
            log_text.append(line + "\n", style="red bold")
        elif 'Save' in line or 'DONE' in line or '完成' in line:
            log_text.append(line + "\n", style="green")
        else:
            log_text.append(line + "\n", style="white")

    # 组合
    outer = Table.grid(padding=1)
    outer.add_column()
    outer.add_row(table)
    outer.add_row(Panel(gpu_text, title="[bold]GPU 显存监控[/bold]", border_style=mem_color))
    outer.add_row(Panel(log_text, title="[bold]训练日志 (最近15行)[/bold]", border_style="blue"))

    return Panel(outer, title=f"[bold cyan]AetherMind V4 训练监控终端[/bold cyan]  "
                              f"[{datetime.now().strftime('%H:%M:%S')}]",
                 border_style="cyan")


def main():
    if len(sys.argv) < 2:
        print("用法: python monitor_train.py <训练命令或.cmd文件> [参数...]")
        print("示例1: python monitor_train.py src/training/train_v4.py --phase_g ...")
        print("示例2: python monitor_train.py train_v4_full_7day.cmd")
        sys.exit(1)

    # 支持直接运行 .cmd 文件
    if sys.argv[1].lower().endswith('.cmd') or sys.argv[1].lower().endswith('.bat'):
        cmd = ['cmd', '/c', sys.argv[1]]
    else:
        cmd = [PYTHON] + sys.argv[1:]
    log_file = os.path.join(LOG_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    print(f"[Monitor] 命令: {' '.join(cmd)}")
    print(f"[Monitor] 日志: {log_file}")
    print(f"[Monitor] 启动训练进程...\n")

    m = Metrics()
    stop_event = threading.Event()

    # 启动GPU监控线程
    gpu_thread = threading.Thread(target=gpu_monitor_thread, args=(m, stop_event),
                                  daemon=True)
    gpu_thread.start()

    # 启动训练子进程
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace',
        cwd=PROJECT_ROOT,
    )

    # 写日志文件
    log_fh = open(log_file, 'w', encoding='utf-8')

    # 输出读取线程
    def read_output():
        for line in proc.stdout:
            line = line.rstrip('\n\r')
            log_fh.write(line + '\n')
            log_fh.flush()
            parse_line(line, m)
            m.add_log(line)

    reader_thread = threading.Thread(target=read_output, daemon=True)
    reader_thread.start()

    # TUI 主循环
    console = Console()
    try:
        with Live(build_dashboard(m), console=console, refresh_per_second=4,
                  screen=True) as live:
            while proc.poll() is None:
                time.sleep(0.25)
                live.update(build_dashboard(m))
    except KeyboardInterrupt:
        print("\n[Monitor] 用户中断，终止训练进程...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        stop_event.set()
        reader_thread.join(timeout=3)
        log_fh.close()

    ret = proc.returncode
    print(f"\n[Monitor] 训练进程退出, code={ret}")
    print(f"[Monitor] 完整日志: {log_file}")
    sys.exit(ret if ret else 0)


if __name__ == '__main__':
    main()
