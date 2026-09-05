"""
AetherMind V4 训练监控中心 (纯Win32 API原生窗口)
- 零依赖：只用ctypes调用Windows API，不装任何GUI库
- 独立原生窗口，双击exe即开，不依赖cmd终端
- 实时曲线：Loss / 物理占比 / τ / GPU显存 / LR (GDI绘制)
- 全套验证指标 + 显存监控 + 训练日志
"""
import os
import sys
import re
import time
import threading
import subprocess
import ctypes
from ctypes import wintypes
from datetime import datetime

# ============================================================
# Win32 API 定义
# ============================================================
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# 常量
WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_VISIBLE = 0x10000000
WS_CHILD = 0x40000000
WS_VSCROLL = 0x00200000
ES_MULTILINE = 0x0004
ES_READONLY = 0x0800
ES_AUTOVSCROLL = 0x0040
WM_PAINT = 0x000F
WM_TIMER = 0x0113
WM_DESTROY = 0x0002
WM_SETFONT = 0x0030
WM_SETTEXT = 0x000C
EM_REPLACESEL = 0x00C2
EM_SETSEL = 0x00B1
EM_SCROLLCARET = 0x00B7
CW_USEDEFAULT = 0x80000000
DT_RIGHT = 0x00000002
DT_VCENTER = 0x00000004
DT_SINGLELINE = 0x00000020
TRANSPARENT = 1
DEFAULT_CHARSET = 1
FF_SWISS = 0x30
FW_BOLD = 700
FW_NORMAL = 400
PS_SOLID = 0
BLACK_BRUSH = 4
WHITE_BRUSH = 0
COLOR_WINDOW = 5
NULL = 0

# 颜色 (RGB)
def RGB(r, g, b):
    return r | (g << 8) | (b << 16)

C_BG = RGB(26, 26, 46)
C_PANEL = RGB(22, 33, 62)
C_PANEL2 = RGB(15, 52, 96)
C_TEXT = RGB(224, 224, 224)
C_DIM = RGB(136, 136, 136)
C_GREEN = RGB(0, 230, 118)
C_YELLOW = RGB(255, 235, 59)
C_RED = RGB(255, 82, 82)
C_CYAN = RGB(0, 229, 255)
C_BLUE = RGB(68, 138, 255)
C_ORANGE = RGB(255, 145, 0)
C_PURPLE = RGB(179, 136, 255)
C_LOG_BG = RGB(13, 17, 23)

MEM_WARN = 0.80
MEM_CRIT = 0.92
MAX_POINTS = 400

# ============================================================
# 路径配置
# ============================================================
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PYTHON = None
for c in [r"C:\Python312\python.exe", r"C:\Python311\python.exe"]:
    if os.path.exists(c):
        PYTHON = c
        break
if PYTHON is None:
    PYTHON = sys.executable

LOG_DIR = os.path.join(PROJECT_ROOT, "training_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 指标解析
# ============================================================
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
RE_TAU_SLOPE = re.compile(r'slope=([\d.]+)')
RE_KL_WARN = re.compile(r'KL=([\d.]+)')
RE_WATCHDOG_CLEAR = re.compile(r'累计清理(\d+)次')
RE_WATCHDOG_EMERG = re.compile(r'累计紧急(\d+)次')
RE_RESUME_STEP = re.compile(r'\[Resume\] 恢复成功: step=(\d+)')
RE_START_STEP = re.compile(r'\[Train\] 起始步数: (\d+)')


class Metrics:
    def __init__(self):
        self.phase = "?"; self.step = 0; self.total = 0
        self.loss = 0.0; self.lr = 0.0; self.tau = 0.0; self.cons = 0.0
        self.physics_ratio = 0.0; self.dominance = "?"
        self.bad = 0; self.sps = 0.0
        self.gpu_used = 0; self.gpu_total = 4096; self.gpu_pct = 0.0
        self.tau_slope = 0.0; self.tau_high = False
        self.nan_detected = False; self.grad_ratio = 0.0
        self.compression_kl = 0.0; self.kl_warning = False
        self.watchdog_clears = 0; self.watchdog_emergency = 0
        self.curve_loss = []; self.curve_physics = []
        self.curve_tau = []; self.curve_gpu = []; self.curve_lr = []
        self.start_time = time.time()

    def add_point(self):
        for arr, val in [(self.curve_loss, self.loss), (self.curve_physics, self.physics_ratio),
                         (self.curve_tau, self.tau), (self.curve_gpu, self.gpu_pct),
                         (self.curve_lr, self.lr)]:
            arr.append(val)
            if len(arr) > MAX_POINTS:
                del arr[0]


def parse_line(line, m):
    for pat, attr, caster in [
        (RE_LOSS, 'loss', float), (RE_LR, 'lr', float),
        (RE_TAU, 'tau', float), (RE_CONS, 'cons', float),
        (RE_PHYSICS, 'physics_ratio', float), (RE_BAD, 'bad', int),
        (RE_SPS, 'sps', float), (RE_TAU_SLOPE, 'tau_slope', float),
        (RE_KL_WARN, 'compression_kl', float),
        (RE_WATCHDOG_CLEAR, 'watchdog_clears', int),
        (RE_WATCHDOG_EMERG, 'watchdog_emergency', int),
    ]:
        mo = pat.search(line)
        if mo:
            try: setattr(m, attr, caster(mo.group(1)))
            except ValueError: pass
    mo = RE_PHASE.search(line)
    if mo: m.phase = mo.group(1) or mo.group(2) or m.phase
    mo = RE_STEP.search(line)
    if mo:
        try:
            if mo.group(1): m.step, m.total = int(mo.group(1)), int(mo.group(2))
            elif mo.group(3): m.step, m.total = int(mo.group(3)), int(mo.group(4))
        except (ValueError, IndexError): pass
    mo = RE_RESUME_STEP.search(line)
    if mo:
        # resume后立即同步步数并清空曲线，避免旧数据残留导致面板未同步
        try:
            m.step = int(mo.group(1))
            m.curve_loss.clear(); m.curve_physics.clear()
            m.curve_tau.clear(); m.curve_gpu.clear(); m.curve_lr.clear()
        except (ValueError, IndexError): pass
    mo = RE_START_STEP.search(line)
    if mo:
        try: m.step = int(mo.group(1))
        except (ValueError, IndexError): pass
    if '->' in line and any(k in line for k in ['dominant','mixed','significant']):
        mo = RE_DOMINANCE.search(line)
        if mo: m.dominance = mo.group(1)
    if 'nan' in line.lower() and 'loss' in line.lower(): m.nan_detected = True
    if 'τ浓度过高' in line: m.tau_high = True
    if '平移不变假设失效' in line: m.kl_warning = True


def get_gpu_memory():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, timeout=5)
        raw = r.stdout
        for enc in ('utf-8', 'gbk', 'latin-1'):
            try: text = raw.decode(enc); break
            except UnicodeDecodeError: continue
        else: text = raw.decode('latin-1', errors='ignore')
        parts = text.strip().split(',')
        if len(parts) >= 2:
            return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass
    return 0, 4096


# ============================================================
# GDI 绘制工具
# ============================================================
def create_font(height, bold=False):
    lf = ctypes.create_string_buffer(280)  # LOGFONT
    ctypes.memset(lf, 0, 280)
    ctypes.cast(lf, ctypes.POINTER(wintypes.LONG))[0] = height  # lfHeight
    ctypes.cast(lf, ctypes.POINTER(wintypes.LONG))[3] = FW_BOLD if bold else FW_NORMAL  # lfWeight
    ctypes.cast(lf, ctypes.POINTER(wintypes.BYTE))[23] = DEFAULT_CHARSET  # lfCharSet
    face = "Consolas"
    ctypes.memmove(ctypes.addressof(lf) + 28, face.encode('utf-16-le')[:62], 62)
    return gdi32.CreateFontIndirectW(lf)


def draw_text(hdc, x, y, text, color, font=None):
    old_color = gdi32.SetTextColor(hdc, color)
    gdi32.SetBkMode(hdc, TRANSPARENT)
    old_font = None
    if font:
        old_font = gdi32.SelectObject(hdc, font)
    user32.TextOutW(hdc, x, y, text, len(text))
    if old_font:
        gdi32.SelectObject(hdc, old_font)
    gdi32.SetTextColor(hdc, old_color)


def fill_rect(hdc, x, y, w, h, color):
    brush = gdi32.CreateSolidBrush(color)
    rect = wintypes.RECT(x, y, x + w, y + h)
    user32.FillRect(hdc, ctypes.byref(rect), brush)
    gdi32.DeleteObject(brush)


def draw_curve(hdc, x, y, w, h, data, color, title):
    """绘制一条曲线"""
    fill_rect(hdc, x, y, w, h, C_PANEL)
    # 网格
    pen = gdi32.CreatePen(PS_SOLID, 1, C_PANEL2)
    old_pen = gdi32.SelectObject(hdc, pen)
    for i in range(1, 4):
        gy = y + h * i // 4
        gdi32.MoveToEx(hdc, x, gy, None)
        gdi32.LineTo(hdc, x + w, gy)
    gdi32.SelectObject(hdc, old_pen)
    gdi32.DeleteObject(pen)
    # 标题
    draw_text(hdc, x + 4, y + 3, title, C_DIM)
    if not data or len(data) < 2:
        draw_text(hdc, x + w // 2 - 25, y + h // 2, "等待数据...", C_DIM)
        return
    vals = [v for v in data if v is not None]
    if not vals: return
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin: vmax = vmin + 1
    vrange = vmax - vmin
    n = len(data)
    pad = 4
    pw = w - pad * 2
    ph = h - 24
    # 曲线
    pen = gdi32.CreatePen(PS_SOLID, 2, color)
    old_pen = gdi32.SelectObject(hdc, pen)
    points = []
    for i, v in enumerate(data):
        px = x + pad + (i / (n - 1)) * pw if n > 1 else x + pad
        py = y + 20 + (1 - (v - vmin) / vrange) * ph
        points.append((int(px), int(py)))
    for i in range(len(points) - 1):
        gdi32.MoveToEx(hdc, points[i][0], points[i][1], None)
        gdi32.LineTo(hdc, points[i+1][0], points[i+1][1])
    gdi32.SelectObject(hdc, old_pen)
    gdi32.DeleteObject(pen)
    # 最新值
    last = data[-1]
    if last is not None:
        txt = f"{last:.4f}" if abs(last) < 100 else f"{last:.1f}"
        draw_text(hdc, x + w - 60, y + 3, txt, color)
    # min/max
    draw_text(hdc, x + 4, y + h - 14, f"min={vmin:.3f}", C_DIM)
    draw_text(hdc, x + w - 70, y + h - 14, f"max={vmax:.3f}", C_DIM)


def draw_panel(hdc, x, y, w, h, title, color):
    fill_rect(hdc, x, y, w, h, C_PANEL)
    # 边框
    pen = gdi32.CreatePen(PS_SOLID, 1, color)
    old_pen = gdi32.SelectObject(hdc, pen)
    old_brush = gdi32.SelectObject(hdc, gdi32.GetStockObject(NULL))
    gdi32.RoundRect(hdc, x, y, x + w, y + h, 6, 6)
    gdi32.SelectObject(hdc, old_pen)
    gdi32.SelectObject(hdc, old_brush)
    gdi32.DeleteObject(pen)
    # 标题
    draw_text(hdc, x + w // 2 - len(title) * 4, y + 3, title, color)


def draw_metric_row(hdc, x, y, label, value, color=C_TEXT):
    draw_text(hdc, x, y, label, C_DIM)
    draw_text(hdc, x + 140, y, value, color)


# ============================================================
# 全局状态
# ============================================================
metrics = Metrics()
hwnd_main = None
hwnd_log = None
font_normal = None
font_bold = None
font_small = None
train_proc = None
log_file = None
point_counter = 0
stop_event = threading.Event()


def gpu_monitor_loop():
    while not stop_event.is_set():
        used, total = get_gpu_memory()
        metrics.gpu_used = used
        metrics.gpu_total = total
        metrics.gpu_pct = used / total if total > 0 else 0
        stop_event.wait(2)


def append_log(text, tag="info"):
    if hwnd_log:
        # 移动到末尾
        user32.SendMessageW(hwnd_log, EM_SETSEL, -1, -1)
        user32.SendMessageW(hwnd_log, EM_REPLACESEL, 0, text + "\r\n")


def train_reader():
    global point_counter
    for line in train_proc.stdout:
        line = line.rstrip('\n\r')
        if not line:
            continue
        if log_file:
            log_file.write(line + '\n')
            log_file.flush()
        parse_line(line, metrics)
        point_counter += 1
        if point_counter % 10 == 0:
            metrics.add_point()
        append_log(line)


# ============================================================
# 窗口过程
# ============================================================
def wnd_proc(hwnd, msg, wparam, lparam):
    global hwnd_log, font_normal, font_bold, font_small

    if msg == WM_CREATE:
        # 创建日志编辑框
        hwnd_log = user32.CreateWindowExW(
            0, "EDIT", "",
            WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
            0, 0, 0, 0, hwnd, NULL, NULL, NULL
        )
        user32.SendMessageW(hwnd_log, WM_SETFONT, font_normal, 0)
        # 启动定时器 (500ms)
        user32.SetTimer(hwnd, 1, 500, None)
        return 0

    elif msg == WM_TIMER:
        user32.InvalidateRect(hwnd, None, True)
        return 0

    elif msg == WM_PAINT:
        ps = wintypes.PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        W = rect.right
        H = rect.bottom

        # 背景
        fill_rect(hdc, 0, 0, W, H, C_BG)

        # 标题栏
        fill_rect(hdc, 0, 0, W, 32, C_PANEL2)
        draw_text(hdc, 12, 8, "◆ AetherMind V4 训练监控中心", C_CYAN, font_bold)
        clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw_text(hdc, W - 160, 8, clock, C_DIM, font_normal)

        m = metrics
        left_w = 270
        right_x = left_w + 12
        right_w = W - right_x - 10

        # ===== 左侧面板 =====
        py = 42
        # 训练状态
        draw_panel(hdc, 10, py, left_w, 170, "训练状态", C_CYAN)
        ry = py + 24
        draw_metric_row(hdc, 20, ry, "阶段", m.phase, C_CYAN); ry += 18
        draw_metric_row(hdc, 20, ry, "步数", f"{m.step:,}/{m.total:,}"); ry += 18
        pct = (m.step / m.total * 100) if m.total > 0 else 0
        draw_metric_row(hdc, 20, ry, "进度", f"{pct:.1f}%"); ry += 18
        draw_metric_row(hdc, 20, ry, "Loss", f"{m.loss:.4f}", C_CYAN); ry += 18
        draw_metric_row(hdc, 20, ry, "LR", f"{m.lr:.2e}"); ry += 18
        draw_metric_row(hdc, 20, ry, "速度", f"{m.sps:.1f} step/s"); ry += 18
        eta = "?"
        if m.sps > 0 and m.total > 0:
            rem = (m.total - m.step) / m.sps
            eta = f"{rem/3600:.1f}h" if rem > 3600 else f"{rem/60:.0f}min"
        draw_metric_row(hdc, 20, ry, "ETA", eta); ry += 18
        bad_color = C_RED if m.bad > 0 else C_GREEN
        draw_metric_row(hdc, 20, ry, "坏批次", str(m.bad), bad_color)

        py += 178
        # 物理演化
        draw_panel(hdc, 10, py, left_w, 120, "物理演化", C_ORANGE)
        ry = py + 24
        draw_metric_row(hdc, 20, ry, "τ 浓度", f"{m.tau:.1f}"); ry += 18
        slope_color = C_YELLOW if m.tau_slope > 0 else C_GREEN
        draw_metric_row(hdc, 20, ry, "τ 斜率", f"{m.tau_slope:.2f}", slope_color); ry += 18
        draw_metric_row(hdc, 20, ry, "固化量", f"{m.cons:,.0f}"); ry += 18
        draw_metric_row(hdc, 20, ry, "物理占比", f"{m.physics_ratio:.3f}", C_GREEN); ry += 18
        dom_color = C_GREEN if m.physics_ratio > 0.3 else (C_YELLOW if m.physics_ratio > 0.05 else C_RED)
        draw_metric_row(hdc, 20, ry, "主导权", m.dominance, dom_color)

        py += 128
        # 验证监控
        draw_panel(hdc, 10, py, left_w, 130, "验证监控", C_PURPLE)
        ry = py + 24
        nan_color = C_RED if m.nan_detected else C_GREEN
        draw_metric_row(hdc, 20, ry, "NaN/Inf", "异常!" if m.nan_detected else "正常", nan_color); ry += 18
        th_color = C_RED if m.tau_high else C_GREEN
        draw_metric_row(hdc, 20, ry, "τ 过饱和", "过饱和!" if m.tau_high else "正常", th_color); ry += 18
        draw_metric_row(hdc, 20, ry, "梯度比", f"{m.grad_ratio:.1e}"); ry += 18
        draw_metric_row(hdc, 20, ry, "压缩KL", f"{m.compression_kl:.3f}"); ry += 18
        kl_color = C_RED if m.kl_warning else C_GREEN
        draw_metric_row(hdc, 20, ry, "迁移性", "FAIL!" if m.kl_warning else "PASS", kl_color)

        py += 138
        # GPU显存
        draw_panel(hdc, 10, py, left_w, 110, "GPU 显存", C_RED)
        ry = py + 24
        draw_metric_row(hdc, 20, ry, "显存", f"{m.gpu_used}/{m.gpu_total}MB"); ry += 18
        if m.gpu_pct >= MEM_CRIT: gpu_c, gpu_s = C_RED, "CRITICAL"
        elif m.gpu_pct >= MEM_WARN: gpu_c, gpu_s = C_YELLOW, "WARNING"
        else: gpu_c, gpu_s = C_GREEN, "SAFE"
        draw_metric_row(hdc, 20, ry, "占用率", f"{m.gpu_pct:.0%}", gpu_c); ry += 18
        draw_metric_row(hdc, 20, ry, "状态", gpu_s, gpu_c); ry += 18
        draw_metric_row(hdc, 20, ry, "守护清理", f"{m.watchdog_clears}次"); ry += 18
        em_color = C_RED if m.watchdog_emergency > 0 else C_TEXT
        draw_metric_row(hdc, 20, ry, "紧急释放", f"{m.watchdog_emergency}次", em_color)
        # 显存条
        bar_x, bar_y, bar_w, bar_h = 20, py + 96, left_w - 40, 10
        fill_rect(hdc, bar_x, bar_y, bar_w, bar_h, C_PANEL2)
        fill_w = int(bar_w * m.gpu_pct)
        fill_rect(hdc, bar_x, bar_y, fill_w, bar_h, gpu_c)

        # ===== 右侧曲线 =====
        chart_h = 95
        chart_gap = 6
        cw = (right_w - chart_gap * 2) // 3
        # 第一行
        cy = 42
        draw_curve(hdc, right_x, cy, cw, chart_h, m.curve_loss, C_CYAN, "Loss")
        draw_curve(hdc, right_x + cw + chart_gap, cy, cw, chart_h, m.curve_physics, C_GREEN, "物理占比")
        draw_curve(hdc, right_x + (cw + chart_gap) * 2, cy, cw, chart_h, m.curve_tau, C_ORANGE, "τ 浓度")
        # 第二行
        cy += chart_h + chart_gap
        cw2 = (right_w - chart_gap) // 2
        draw_curve(hdc, right_x, cy, cw2, chart_h - 15, m.curve_gpu, C_RED, "GPU 显存 %")
        draw_curve(hdc, right_x + cw2 + chart_gap, cy, cw2, chart_h - 15, m.curve_lr, C_PURPLE, "Learning Rate")

        # ===== 日志区 =====
        log_y = cy + chart_h - 15 + chart_gap
        log_h = H - log_y - 30
        fill_rect(hdc, right_x, log_y, right_w, log_h, C_LOG_BG)
        draw_text(hdc, right_x + 6, log_y + 4, " 训练日志 ", C_CYAN, font_small)
        # 移动日志编辑框到这个区域
        user32.MoveWindow(hwnd_log, right_x + 4, log_y + 22, right_w - 8, log_h - 26, True)

        # 状态栏
        fill_rect(hdc, 0, H - 24, W, 24, C_PANEL2)
        if train_proc and train_proc.poll() is not None:
            status = f"训练进程已退出 code={train_proc.returncode}"
        else:
            elapsed = time.time() - m.start_time
            status = f"运行中 | 耗时 {elapsed/3600:.1f}h | 日志: {os.path.basename(log_file.name) if log_file else ''}"
        draw_text(hdc, 10, H - 18, status, C_DIM, font_small)

        user32.EndPaint(hwnd, ctypes.byref(ps))
        return 0

    elif msg == WM_DESTROY:
        stop_event.set()
        user32.KillTimer(hwnd, 1)
        if train_proc and train_proc.poll() is None:
            train_proc.terminate()
            try: train_proc.wait(timeout=5)
            except subprocess.TimeoutExpired: train_proc.kill()
        if log_file: log_file.close()
        user32.PostQuitMessage(0)
        return 0

    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


# ============================================================
# 主入口
# ============================================================
def main():
    global font_normal, font_bold, font_small, train_proc, log_file

    # 训练命令
    if len(sys.argv) < 2:
        train_cmd = ["train_v4_full_7day.cmd"]
    else:
        train_cmd = sys.argv[1:]

    # 注册窗口类
    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, wintypes.HWND, wintypes.UINT,
                                  wintypes.WPARAM, wintypes.LPARAM)
    wnd_proc_ref = WNDPROC(wnd_proc)

    class WNDCLASSEX(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT),
                    ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                    ("hIcon", wintypes.HANDLE), ("hCursor", wintypes.HANDLE),
                    ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                    ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HANDLE)]

    hInstance = kernel32.GetModuleHandleW(None)
    wc = WNDCLASSEX()
    wc.cbSize = ctypes.sizeof(WNDCLASSEX)
    wc.lpfnWndProc = wnd_proc_ref
    wc.hInstance = hInstance
    wc.lpszClassName = "AetherMindMonitor"
    wc.hbrBackground = 0
    user32.RegisterClassExW(ctypes.byref(wc))

    # 创建字体
    font_normal = create_font(-14, False)
    font_bold = create_font(-14, True)
    font_small = create_font(-12, False)

    # 创建窗口
    hwnd = user32.CreateWindowExW(
        0, "AetherMindMonitor", "AetherMind V4 训练监控中心",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 1320, 840,
        NULL, NULL, hInstance, NULL
    )

    # 日志文件
    log_path = os.path.join(LOG_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, 'w', encoding='utf-8')

    # 启动训练
    if train_cmd[0].lower().endswith('.cmd') or train_cmd[0].lower().endswith('.bat'):
        cmd = ['cmd', '/c'] + train_cmd
    else:
        cmd = [PYTHON] + train_cmd
    append_log(f"[Monitor] 启动: {' '.join(cmd)}")
    try:
        train_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True, encoding='utf-8',
            errors='replace', cwd=PROJECT_ROOT
        )
    except Exception as e:
        append_log(f"[Monitor] 启动失败: {e}")
        train_proc = None

    if train_proc:
        t = threading.Thread(target=train_reader, daemon=True)
        t.start()

    # GPU监控线程
    t2 = threading.Thread(target=gpu_monitor_loop, daemon=True)
    t2.start()

    # 消息循环
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), NULL, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))

    sys.exit(msg.wParam)


if __name__ == '__main__':
    main()
