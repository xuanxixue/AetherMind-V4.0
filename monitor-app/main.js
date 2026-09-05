// AetherMind V4 训练监控中心 - Electron 主进程 (多会话版)
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

app.disableHardwareAcceleration();

let mainWindow = null;
let gpuMonitorTimer = null;
let sessions = new Map(); // id -> { pty, label, buffer }
let nextSessionId = 1;
let activeSessionId = null;

const metrics = {
  phase: '?', step: 0, total: 0, loss: 0, lr: 0, tau: 0, cons: 0,
  physics_ratio: 0, dominance: '?', bad: 0, sps: 0,
  gpu_used: 0, gpu_total: 4096, gpu_pct: 0,
  tau_slope: 0, tau_high: false, nan_detected: false, grad_ratio: 0,
  compression_kl: 0, kl_warning: false,
  watchdog_clears: 0, watchdog_emergency: 0,
  running: false
};

const curveData = { loss: [], physics: [], tau: [], gpu: [], lr: [], steps: [] };
const MAX_POINTS = 400;

// 重置全部指标与曲线（每次启动新训练会话时调用，避免旧 session 残留导致"面板未同步"）
function resetMetrics() {
  Object.assign(metrics, {
    phase: '?', step: 0, total: 0, loss: 0, lr: 0, tau: 0, cons: 0,
    physics_ratio: 0, dominance: '?', bad: 0, sps: 0,
    tau_slope: 0, tau_high: false, nan_detected: false, grad_ratio: 0,
    compression_kl: 0, kl_warning: false,
    watchdog_clears: 0, watchdog_emergency: 0,
    running: true
  });
  delete metrics.eta;
  for (const k of Object.keys(curveData)) curveData[k] = [];
}

const RE = {
  loss: /loss=([\d.]+)/, lr: /lr=([\d.eE+-]+)/,
  phase: /ph=([A-Z])|Phase ([A-Z])/, step: /\(\s*(\d+)\s*\/\s*(\d+)\s*\)|Phase\s+[A-Z]\]\s+(\d+)\/(\d+)/,
  tau: /tau=([\d.]+)/, cons: /cons=([\d.]+)/,
  physics: /physics=([\d.]+)/, dominance: /->\s*(\S+)/,
  bad: /bad=(\d+)/, sps: /([\d.]+)\s*step\/s/, eta: /ETA\s+(\S+)/,
  tau_slope: /slope=([\d.]+)/, kl: /KL=([\d.]+)/,
  wd_clear: /累计清理(\d+)次/, wd_emerg: /累计紧急(\d+)次/,
  resume_step: /\[Resume\] 恢复成功: step=(\d+)/,
  start_step: /\[Train\] 起始步数: (\d+)/
};

function parseLine(line) {
  for (const [key, regex] of Object.entries(RE)) {
    const m = line.match(regex);
    if (m) {
      if (key === 'phase') metrics.phase = m[1] || m[2] || metrics.phase;
      else if (key === 'step') {
        if (m[1]) { metrics.step = parseInt(m[1]); metrics.total = parseInt(m[2]); }
        else if (m[3]) { metrics.step = parseInt(m[3]); metrics.total = parseInt(m[4]); }
      } else if (key === 'resume_step' || key === 'start_step') {
        // resume/从指定步数继续：立即同步面板步数；resume时清空曲线让x轴从新起点开始
        metrics.step = parseInt(m[1]);
        if (key === 'resume_step') for (const k of Object.keys(curveData)) curveData[k] = [];
      } else if (key === 'dominance') {
        if (line.includes('->') && (line.includes('dominant')||line.includes('mixed')||line.includes('significant')))
          metrics.dominance = m[1];
      } else if (key === 'kl') {
        if (line.includes('平移不变假设失效')) { metrics.kl_warning = true; metrics.compression_kl = parseFloat(m[1]); }
      } else if (key === 'eta') {
        metrics.eta = m[1];
      } else {
        const val = parseFloat(m[1]);
        if (!isNaN(val)) metrics[key] = val;
      }
    }
  }
  if (line.toLowerCase().includes('nan') && line.toLowerCase().includes('loss')) metrics.nan_detected = true;
  if (line.includes('τ浓度过高')) metrics.tau_high = true;
}

// 缓冲解析：在文本中找每个正则的最后一个匹配（兼容ConPTY吃掉换行符的情况）
function parseBuffer(text) {
  for (const [key, regex] of Object.entries(RE)) {
    try {
      const gre = new RegExp(regex.source, 'g');
      const matches = [...text.matchAll(gre)];
      if (matches.length === 0) continue;
      const m = matches[matches.length - 1];
      if (key === 'phase') metrics.phase = m[1] || m[2] || metrics.phase;
      else if (key === 'step') {
        if (m[1]) { metrics.step = parseInt(m[1]); metrics.total = parseInt(m[2]); }
        else if (m[3]) { metrics.step = parseInt(m[3]); metrics.total = parseInt(m[4]); }
      } else if (key === 'resume_step' || key === 'start_step') {
        // resume/从指定步数继续：立即同步面板步数；resume时清空曲线让x轴从新起点开始
        metrics.step = parseInt(m[1]);
        if (key === 'resume_step') for (const k of Object.keys(curveData)) curveData[k] = [];
      } else if (key === 'dominance') {
        if (text.includes('->') && (text.includes('dominant')||text.includes('mixed')||text.includes('significant')))
          metrics.dominance = m[1];
      } else if (key === 'kl') {
        if (text.includes('平移不变假设失效')) { metrics.kl_warning = true; metrics.compression_kl = parseFloat(m[1]); }
      } else if (key === 'eta') {
        metrics.eta = m[1];
      } else {
        const val = parseFloat(m[1]);
        if (!isNaN(val)) metrics[key] = val;
      }
    } catch(e) {}
  }
}

function pushCurvePoint() {
  curveData.loss.push(metrics.loss);
  curveData.physics.push(metrics.physics_ratio);
  curveData.tau.push(metrics.tau);
  curveData.gpu.push(metrics.gpu_pct);
  curveData.lr.push(metrics.lr);
  curveData.steps.push(metrics.step);
  for (const k of Object.keys(curveData)) {
    if (curveData[k].length > MAX_POINTS) curveData[k].shift();
  }
}

function queryGpuMemory() {
  return new Promise((resolve) => {
    exec('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits',
      { timeout: 5000, encoding: 'utf-8' },
      (err, stdout) => {
        if (err) { resolve([0, 4096]); return; }
        const parts = stdout.trim().split(',');
        if (parts.length >= 2) resolve([parseInt(parts[0].trim()), parseInt(parts[1].trim())]);
        else resolve([0, 4096]);
      }
    );
  });
}

async function gpuMonitorLoop() {
  const [used, total] = await queryGpuMemory();
  metrics.gpu_used = used;
  metrics.gpu_total = total;
  metrics.gpu_pct = total > 0 ? used / total : 0;
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('gpu-update', { used, total, pct: metrics.gpu_pct });
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400, height: 900, minWidth: 1100, minHeight: 700,
    backgroundColor: '#1a1a2e',
    title: 'AetherMind V4 Monitor',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true, nodeIntegration: false
    },
    autoHideMenuBar: true,
    show: false
  });
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.once('ready-to-show', () => mainWindow.show());
  gpuMonitorTimer = setInterval(gpuMonitorLoop, 2000);
  gpuMonitorLoop();
}

// 创建新PTY会话
function createSession(cmdArgs, label) {
  resetMetrics();  // 新会话清空旧指标/曲线，避免跨session残留导致面板不同步
  const id = nextSessionId++;
  const projectRoot = path.resolve(__dirname, '..');
  const isCmd = cmdArgs[0] && (cmdArgs[0].endsWith('.cmd') || cmdArgs[0].endsWith('.bat'));

  let shell, args;
  if (isCmd) { shell = 'cmd.exe'; args = ['/c', ...cmdArgs]; }
  else {
    shell = 'C:\\Python312\\python.exe';
    if (!fs.existsSync(shell)) shell = 'python.exe';
    args = cmdArgs;
  }

  try {
    const pty = require('node-pty');
    const proc = pty.spawn(shell, args, {
      name: 'xterm-color', cols: 120, rows: 30,
      cwd: projectRoot,
      env: {
        ...process.env,
        HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', HF_DATASETS_OFFLINE: '1',
        PYTORCH_CUDA_ALLOC_CONF: 'max_split_size_mb:128'
      }
    });

    const session = { pty: proc, label, buffer: '', pointCounter: 0 };
    sessions.set(id, session);
    metrics.running = true;

    proc.onData((data) => {
      session.buffer += data;
      if (session.buffer.length > 500000) session.buffer = session.buffer.slice(-300000);
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('pty-data', { id, data });
      }
      // 缓冲解析：ConPTY会吃掉\r/\n，直接在最近的输出缓冲中找最后一个匹配
      session.pointCounter++;
      if (session.pointCounter % 3 === 0) {
        parseBuffer(session.buffer.slice(-4000));
        pushCurvePoint();
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('metrics-update', { ...metrics, curves: { ...curveData } });
        }
      }
    });

    proc.onExit(() => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('session-exit', { id });
      }
      sessions.delete(id);
      if (sessions.size === 0) metrics.running = false;
    });

    return { ok: true, id, label };
  } catch (e) {
    return { ok: false, msg: e.message };
  }
}

// IPC: 创建会话
ipcMain.handle('session-create', (event, cmdArgs, label) => {
  return createSession(cmdArgs, label || 'Session');
});

// IPC: 关闭会话
ipcMain.handle('session-close', (event, id) => {
  const s = sessions.get(id);
  if (s) { try { s.pty.kill(); } catch (e) {} sessions.delete(id); }
  if (sessions.size === 0) metrics.running = false;
  return { ok: true };
});

// IPC: 切换会话
ipcMain.handle('session-switch', (event, id) => {
  activeSessionId = id;
  const s = sessions.get(id);
  return { ok: true, buffer: s ? s.buffer : '' };
});

// IPC: 终端输入
ipcMain.handle('pty-input', (event, id, data) => {
  const s = sessions.get(id);
  if (s) { s.pty.write(data); return { ok: true }; }
  return { ok: false };
});

// IPC: resize
ipcMain.handle('pty-resize', (event, id, cols, rows) => {
  const s = sessions.get(id);
  if (s) { try { s.pty.resize(cols, rows); } catch (e) {} }
});

// IPC: 下载日志
ipcMain.handle('log-download', async (event, id) => {
  const s = sessions.get(id);
  if (!s || !s.buffer) return { ok: false, msg: 'No log data' };
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Terminal Log',
    defaultPath: `aethermind_log_${Date.now()}.txt`,
    filters: [{ name: 'Text', extensions: ['txt'] }]
  });
  if (result.canceled || !result.filePath) return { ok: false, msg: 'Cancelled' };
  try {
    fs.writeFileSync(result.filePath, s.buffer, 'utf-8');
    return { ok: true, path: result.filePath };
  } catch (e) {
    return { ok: false, msg: e.message };
  }
});

// IPC: 获取指标
ipcMain.handle('get-metrics', () => ({ ...metrics, curves: { ...curveData } }));

// IPC: 列出会话
ipcMain.handle('session-list', () => {
  return Array.from(sessions.entries()).map(([id, s]) => ({ id, label: s.label }));
});

app.whenReady().then(createWindow);
app.on('window-all-closed', () => {
  if (gpuMonitorTimer) clearInterval(gpuMonitorTimer);
  for (const [id, s] of sessions) { try { s.pty.kill(); } catch (e) {} }
  sessions.clear();
  if (process.platform !== 'darwin') app.quit();
});
