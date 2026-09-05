"""
GPU显存守护工具
- 训练前：自动清理占用GPU的非必要进程
- 训练中：监控显存，紧急情况下释放显存
- 白名单：保护系统关键进程、当前训练进程
"""
import os
import sys
import subprocess
import gc
import time
import threading
from typing import List, Tuple, Optional

# 进程白名单（这些进程不会被杀掉）
WHITELIST_PROCESSES = {
    # 系统关键进程
    'System', 'System Idle Process', 'Registry', 'smss.exe', 'csrss.exe',
    'wininit.exe', 'winlogon.exe', 'services.exe', 'lsass.exe', 'svchost.exe',
    'fontdrvhost.exe', 'dwm.exe', 'win32k.sys',
    # Windows桌面/资源管理器
    'explorer.exe', 'sihost.exe', 'taskhostw.exe', 'ShellExperienceHost.exe',
    'SearchIndexer.exe', 'SearchUI.exe', 'StartMenuExperienceHost.exe',
    # NVIDIA驱动相关
    'nvcontainer.exe', 'nvsphelper64.exe', 'nvdisplay.container.exe',
    'nvidia share.exe', 'nvidia web helper service.exe',
    # 当前Python解释器（训练进程本身）
    os.path.basename(sys.executable).lower(),  # python.exe
    # 命令行/终端
    'cmd.exe', 'powershell.exe', 'WindowsTerminal.exe', 'conhost.exe',
    # Trae IDE相关
    'trae.exe', 'code.exe', 'cursor.exe',
}

def find_nvidia_smi() -> Optional[str]:
    """查找nvidia-smi.exe路径"""
    possible_paths = [
        r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe',
        r'C:\Windows\System32\nvidia-smi.exe',
        'nvidia-smi',  # PATH中
    ]
    for p in possible_paths:
        try:
            subprocess.run([p, '--version'], capture_output=True, timeout=5)
            return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None

def get_gpu_processes(nvidia_smi: str) -> List[dict]:
    """获取所有占用GPU显存的进程"""
    try:
        # 用CSV格式查询，更稳定；Windows上nvidia-smi输出是GBK编码
        result = subprocess.run(
            [nvidia_smi, '--query-compute-apps=pid,process_name,used_gpu_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, timeout=10
        )
        # Windows nvidia-smi输出GBK编码，Linux是UTF-8
        raw = result.stdout
        for enc in ('utf-8', 'gbk', 'latin-1'):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode('latin-1', errors='ignore')
        
        processes = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                try:
                    pid = int(parts[0])
                    name = os.path.basename(parts[1].strip())
                    mem_mb = int(parts[2].strip())
                    processes.append({
                        'pid': pid,
                        'name': name,
                        'mem_mb': mem_mb,
                        'path': parts[1].strip(),
                    })
                except ValueError:
                    continue
        return processes
    except Exception as e:
        print(f'[GPU Guard] 查询GPU进程失败: {e}')
        return []

def get_total_gpu_memory(nvidia_smi: str) -> Tuple[int, int]:
    """返回 (总显存MB, 已用显存MB)"""
    try:
        result = subprocess.run(
            [nvidia_smi, '--query-gpu=memory.total,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, timeout=10
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
        line = text.strip().split('\n')[0]
        total, used = [int(x.strip()) for x in line.split(',')]
        return total, used
    except Exception:
        return 4096, 0  # 默认4GB

def is_protected(proc_name: str, current_pid: int, proc_pid: int) -> bool:
    """判断进程是否受保护（不能杀）"""
    name_lower = proc_name.lower()
    # 保护当前进程和父进程
    if proc_pid == current_pid or proc_pid == os.getppid():
        return True
    # 白名单匹配
    for wl in WHITELIST_PROCESSES:
        if name_lower == wl.lower():
            return True
    return False

def kill_process(pid: int) -> bool:
    """尝试结束进程"""
    try:
        subprocess.run(
            ['taskkill', '/F', '/PID', str(pid)],
            capture_output=True, timeout=5
        )
        return True
    except Exception:
        return False

def emergency_free_gpu_memory(threshold_mb: int = 300) -> int:
    """紧急释放显存：清理PyTorch缓存，必要时杀掉其他GPU进程
    返回释放的显存MB数
    """
    freed_mb = 0
    
    # 1. 先清理PyTorch缓存
    try:
        import torch
        before = torch.cuda.memory_allocated() / 1024**2
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated() / 1024**2
        freed_mb += before - after
    except ImportError:
        gc.collect()
    
    # 2. 如果nvidia-smi可用，检查是否有其他进程可以杀
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        return int(freed_mb)
    
    total, used = get_total_gpu_memory(nvidia_smi)
    free_mb = total - used
    
    if free_mb >= threshold_mb:
        return int(freed_mb)
    
    # 显存不足，尝试结束其他GPU进程
    procs = get_gpu_processes(nvidia_smi)
    current_pid = os.getpid()
    
    for proc in procs:
        if free_mb >= threshold_mb:
            break
        if is_protected(proc['name'], current_pid, proc['pid']):
            continue
        # 非保护进程，结束它
        print(f'[GPU Guard] 紧急结束进程 PID={proc["pid"]} ({proc["name"]}), '
              f'占用{proc["mem_mb"]}MB', flush=True)
        kill_process(proc['pid'])
        freed_mb += proc['mem_mb']
        free_mb += proc['mem_mb']
        time.sleep(0.5)
    
    return int(freed_mb)

def pre_train_gpu_cleanup(auto_kill: bool = True) -> int:
    """训练前GPU清理
    auto_kill: True=自动结束非保护进程；False=只列出不杀
    返回释放的显存MB数
    """
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        print('[GPU Guard] 未找到nvidia-smi，跳过GPU清理')
        return 0
    
    total, used_before = get_total_gpu_memory(nvidia_smi)
    procs = get_gpu_processes(nvidia_smi)
    current_pid = os.getpid()
    
    print(f'[GPU Guard] GPU总显存: {total}MB, 当前已用: {used_before}MB', flush=True)
    
    if not procs:
        print('[GPU Guard] 没有其他GPU进程，显存干净', flush=True)
        return 0
    
    # 分类进程
    protected_procs = []
    killable_procs = []
    for proc in procs:
        if is_protected(proc['name'], current_pid, proc['pid']):
            protected_procs.append(proc)
        else:
            killable_procs.append(proc)
    
    print(f'[GPU Guard] 当前GPU进程: {len(procs)}个 '
          f'(保护: {len(protected_procs)}个, 可清理: {len(killable_procs)}个)', flush=True)
    
    # 显示可清理的进程
    if killable_procs:
        print('[GPU Guard] 可清理的GPU进程:', flush=True)
        total_killable_mem = 0
        for proc in killable_procs:
            print(f'  PID={proc["pid"]:>6}  {proc["mem_mb"]:>6}MB  {proc["name"]}', flush=True)
            total_killable_mem += proc['mem_mb']
        print(f'[GPU Guard] 可释放显存: ~{total_killable_mem}MB', flush=True)
    
    freed_mb = 0
    if auto_kill and killable_procs:
        print('[GPU Guard] 自动清理非必要GPU进程...', flush=True)
        for proc in killable_procs:
            print(f'  结束: PID={proc["pid"]} {proc["name"]} ({proc["mem_mb"]}MB)', flush=True)
            kill_process(proc['pid'])
            freed_mb += proc['mem_mb']
        time.sleep(1)
        _, used_after = get_total_gpu_memory(nvidia_smi)
        actual_freed = used_before - used_after
        print(f'[GPU Guard] 清理完成! 实际释放: {actual_freed}MB, 当前可用: {total - used_after}MB', flush=True)
        return actual_freed
    
    return freed_mb


class GPUMemoryWatchdog:
    """训练进程内显存守护线程 — 后台轮询，超阈值自动清理，防止OOM中断训练。

    用法:
        watchdog = GPUMemoryWatchdog(threshold=0.85, interval=10)
        watchdog.start()
        # ... 训练循环 ...
        watchdog.stop()
    """

    def __init__(self, threshold: float = 0.85, critical: float = 0.95,
                 interval: float = 10.0, log_callback=None):
        self.threshold = threshold      # 超过85%触发empty_cache
        self.critical = critical        # 超过95%触发紧急清理
        self.interval = interval
        self.log_callback = log_callback  # 可选日志回调 fn(msg)
        self._thread = None
        self._stop_event = threading.Event()
        self._last_clear_time = 0
        self._clear_count = 0
        self._emergency_count = 0
        self._last_used_mb = 0
        self._last_total_mb = 4096

    def _log(self, msg: str):
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass
        print(f'[GPU Watchdog] {msg}', flush=True)

    def _check_and_clear(self):
        nvidia_smi = find_nvidia_smi()
        if nvidia_smi is None:
            return
        total, used = get_total_gpu_memory(nvidia_smi)
        self._last_used_mb = used
        self._last_total_mb = total
        if total <= 0:
            return
        ratio = used / total

        # 临界值：紧急清理（杀其他进程）
        if ratio >= self.critical:
            self._log(f'CRITICAL: {used}/{total}MB ({ratio:.0%}), 紧急释放...')
            freed = emergency_free_gpu_memory(threshold_mb=500)
            self._emergency_count += 1
            self._log(f'紧急释放完成: ~{freed}MB (累计紧急{self._emergency_count}次)')
            return

        # 阈值：常规清理（empty_cache + gc）
        if ratio >= self.threshold:
            now = time.time()
            # 限流：两次清理至少间隔5秒
            if now - self._last_clear_time < 5:
                return
            self._last_clear_time = now
            try:
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self._clear_count += 1
                self._log(f'清理缓存: {used}/{total}MB ({ratio:.0%}) '
                          f'(累计清理{self._clear_count}次)')
            except ImportError:
                gc.collect()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._check_and_clear()
            except Exception as e:
                self._log(f'watchdog error: {e}')
            self._stop_event.wait(self.interval)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='GPUWatchdog')
        self._thread.start()
        self._log(f'已启动: threshold={self.threshold:.0%} '
                  f'critical={self.critical:.0%} interval={self.interval}s')

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._log(f'已停止: 常规清理{self._clear_count}次, '
                  f'紧急清理{self._emergency_count}次')

    def get_stats(self) -> dict:
        return {
            'clear_count': self._clear_count,
            'emergency_count': self._emergency_count,
            'last_used_mb': self._last_used_mb,
            'last_total_mb': self._last_total_mb,
            'running': self._thread is not None and self._thread.is_alive(),
        }


if __name__ == '__main__':
    print('='*60)
    print('  AetherMind GPU显存守护工具 - 训练前清理')
    print('='*60)
    print()
    pre_train_gpu_cleanup(auto_kill=True)
    print()
    print('清理完成！可以开始训练。')
