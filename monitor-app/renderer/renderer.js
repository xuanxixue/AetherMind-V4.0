// AetherMind V4 Monitor - 渲染进程 (多会话版)
const Terminal = window.Terminal;
const FitAddonClass = (window.FitAddon && window.FitAddon.FitAddon) || window.FitAddon;

// 会话管理
let tabs = []; // {id, label, term, fitAddon, paneEl}
let activeTabId = null;
let tabCounter = 0;

const termTheme = {
  background: '#0d1117', foreground: '#e0e0e0', cursor: '#00e5ff',
  black: '#000', red: '#ff5252', green: '#00e676', yellow: '#ffeb3b',
  blue: '#448aff', magenta: '#b388ff', cyan: '#00e5ff', white: '#e0e0e0',
  brightBlack: '#666', brightRed: '#ff7070', brightGreen: '#00ff88',
  brightYellow: '#fff176', brightBlue: '#82b1ff', brightMagenta: '#d1c4e9',
  brightCyan: '#84ffff', brightWhite: '#fff'
};

// === 创建新标签页 ===
function createTab(label) {
  const id = ++tabCounter;
  const paneId = `pane-${id}`;

  // 创建终端容器
  const container = document.getElementById('terminal-container');
  const pane = document.createElement('div');
  pane.className = 'terminal-pane';
  pane.id = paneId;
  container.appendChild(pane);

  // 创建xterm (启用文本选择复制)
  const term = new Terminal({
    cursorBlink: true, fontSize: 12,
    fontFamily: 'Consolas, "Cascadia Code", "Microsoft YaHei", monospace',
    theme: termTheme, scrollback: 5000, convertEol: true,
    allowTransparency: false
  });
  const fitAddon = new FitAddonClass();
  term.loadAddon(fitAddon);
  term.open(pane);

  // 终端输入 -> 发送到对应会话
  term.onData((data) => {
    if (activeTabId === id) window.api.ptyInput(id, data);
  });

  // 创建标签按钮
  const tabsEl = document.getElementById('tabs');
  const tabEl = document.createElement('div');
  tabEl.className = 'tab';
  tabEl.dataset.id = id;
  tabEl.innerHTML = `<span class="tab-label">${label}</span><span class="tab-close">x</span>`;
  tabEl.addEventListener('click', (e) => {
    if (e.target.classList.contains('tab-close')) {
      closeTab(id);
    } else {
      switchTab(id);
    }
  });
  tabsEl.appendChild(tabEl);

  const tab = { id, label, term, fitAddon, paneEl: pane, tabEl };
  tabs.push(tab);

  // 延迟fit
  requestAnimationFrame(() => {
    fitAddon.fit();
    window.api.ptyResize(id, term.cols, term.rows);
  });

  switchTab(id);
  return tab;
}

function switchTab(id) {
  activeTabId = id;
  for (const t of tabs) {
    t.paneEl.classList.toggle('active', t.id === id);
    t.tabEl.classList.toggle('active', t.id === id);
  }
  const tab = tabs.find(t => t.id === id);
  if (tab) {
    requestAnimationFrame(() => {
      tab.fitAddon.fit();
      window.api.ptyResize(id, tab.term.cols, tab.term.rows);
    });
  }
}

function closeTab(id) {
  const idx = tabs.findIndex(t => t.id === id);
  if (idx < 0) return;
  const tab = tabs[idx];
  window.api.sessionClose(id);
  tab.tabEl.remove();
  tab.paneEl.remove();
  tabs.splice(idx, 1);
  if (activeTabId === id) {
    if (tabs.length > 0) switchTab(tabs[Math.max(0, idx - 1)].id);
    else activeTabId = null;
  }
}

function getActiveTab() {
  return tabs.find(t => t.id === activeTabId);
}

// === IPC 监听 ===
window.api.onPtyData(({ id, data }) => {
  const tab = tabs.find(t => t.id === id);
  if (tab) tab.term.write(data);
});

window.api.onMetricsUpdate((data) => {
  updateMetrics(data);
  if (data.curves) drawAllCharts(data.curves);
});

window.api.onGpuUpdate((gpu) => {
  document.getElementById('m-gpu').textContent = `${gpu.used}/${gpu.total} MB`;
  const pct = (gpu.pct * 100).toFixed(1);
  document.getElementById('m-gpupct').textContent = pct + '%';
  const fill = document.getElementById('gpu-bar-fill');
  fill.style.width = pct + '%';
  if (gpu.pct > 0.95) { fill.style.background = '#ff5252'; setVal('m-gpustatus', '危险 OOM!', 'red'); }
  else if (gpu.pct > 0.85) { fill.style.background = '#ffeb3b'; setVal('m-gpustatus', '警告 WARN', 'yellow'); }
  else { fill.style.background = '#00e676'; setVal('m-gpustatus', '安全 SAFE', 'green'); }
});

window.api.onSessionExit(({ id }) => {
  const tab = tabs.find(t => t.id === id);
  if (tab) {
    tab.term.write('\r\n\x1b[31m[进程已退出 / Process exited]\x1b[0m\r\n');
    tab.tabEl.querySelector('.tab-label').textContent = tab.label + ' (已结束)';
  }
  setStatus('会话已退出 / Session exited');
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
});

// === 指标更新 ===
function setVal(id, text, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = 'metric-val' + (cls ? ' val-' + cls : '');
}

function updateMetrics(m) {
  setVal('m-phase', m.phase || '?');
  setVal('m-step', `${m.step} / ${m.total}`);
  const pct = m.total > 0 ? ((m.step / m.total) * 100).toFixed(1) : 0;
  setVal('m-pct', pct + '%');
  document.getElementById('progress-fill').style.width = pct + '%';
  setVal('m-loss', m.loss ? m.loss.toFixed(4) : '0.0000', 'cyan');
  setVal('m-lr', m.lr ? m.lr.toExponential(2) : '0.0');
  setVal('m-sps', (m.sps || 0).toFixed(1) + ' step/s');
  setVal('m-bad', m.bad || 0, (m.bad > 0 ? 'red' : 'green'));

  if (m.sps > 0 && m.total > 0 && m.step > 0) {
    const remaining = (m.total - m.step) / m.sps;
    const h = Math.floor(remaining / 3600);
    const min = Math.floor((remaining % 3600) / 60);
    setVal('m-eta', `${h}h${min}m`);
  }

  setVal('m-tau', (m.tau || 0).toFixed(2));
  setVal('m-tau-slope', (m.tau_slope || 0).toFixed(4));
  setVal('m-cons', (m.cons || 0).toFixed(0));
  setVal('m-physics', (m.physics_ratio || 0).toFixed(3), 'green');
  setVal('m-dom', m.dominance || '?');

  setVal('m-nan', m.nan_detected ? '异常 FAIL' : '正常 OK', m.nan_detected ? 'red' : 'green');
  setVal('m-tauhigh', m.tau_high ? '警告 WARN' : '正常 OK', m.tau_high ? 'yellow' : 'green');
  setVal('m-grad', (m.grad_ratio || 0).toExponential(1));
  setVal('m-kl', (m.compression_kl || 0).toFixed(3));
  setVal('m-klstatus', m.kl_warning ? '失败 FAIL' : '通过 PASS', m.kl_warning ? 'red' : 'green');

  setVal('m-wdclear', m.watchdog_clears || 0);
  setVal('m-wdemerg', m.watchdog_emergency || 0);

  if (m.running) setStatus(`运行中 Running - Phase ${m.phase} | ${m.step}/${m.total} | loss=${(m.loss||0).toFixed(4)}`);
}

// === Canvas 曲线 ===
const charts = {};
const chartColors = {
  'chart-loss': '#00e5ff', 'chart-physics': '#00e676',
  'chart-tau': '#ff9100', 'chart-gpu': '#ff5252', 'chart-lr': '#b388ff'
};

function initCharts() {
  for (const id of Object.keys(chartColors)) {
    const canvas = document.getElementById(id);
    charts[id] = { canvas, ctx: canvas.getContext('2d') };
  }
  resizeCharts();
}

function resizeCharts() {
  for (const id of Object.keys(charts)) {
    const { canvas } = charts[id];
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
  }
}

function drawChart(id, data, color, steps) {
  const { canvas, ctx } = charts[id];
  if (!canvas || !ctx) return;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  if (!data || data.length < 2) return;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pad = h * 0.1, usableH = h - pad * 2;
  // x轴: 优先用step坐标（resume后从新起点对齐）, 否则用点序号
  const useStep = steps && steps.length === data.length;
  const sMin = useStep ? steps[0] : 0;
  const sSpan = useStep ? (steps[steps.length - 1] - sMin || 1) : (data.length - 1);
  const xPos = (i) => useStep ? ((steps[i] - sMin) / sSpan) * w : (i / sSpan) * w;
  ctx.beginPath(); ctx.moveTo(0, h);
  for (let i = 0; i < data.length; i++) {
    const x = xPos(i);
    const y = h - pad - ((data[i] - min) / range) * usableH;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '40'); grad.addColorStop(1, color + '00');
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = xPos(i);
    const y = h - pad - ((data[i] - min) / range) * usableH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color; ctx.lineWidth = 2 * window.devicePixelRatio; ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = `${10 * window.devicePixelRatio}px Consolas`;
  ctx.textAlign = 'right';
  ctx.fillText(data[data.length - 1].toFixed(3), w - 4 * window.devicePixelRatio, 12 * window.devicePixelRatio);
}

function drawAllCharts(curves) {
  const st = curves.steps || [];
  drawChart('chart-loss', curves.loss || [], chartColors['chart-loss'], st);
  drawChart('chart-physics', curves.physics || [], chartColors['chart-physics'], st);
  drawChart('chart-tau', curves.tau || [], chartColors['chart-tau'], st);
  drawChart('chart-gpu', curves.gpu || [], chartColors['chart-gpu'], st);
  drawChart('chart-lr', curves.lr || [], chartColors['chart-lr'], st);
}

// === 按钮事件 ===
document.getElementById('btn-start').addEventListener('click', async () => {
  setStatus('正在启动训练 / Launching training...');
  const tab = createTab('训练 Training');
  const result = await window.api.sessionCreate(['train_v4_full_7day.cmd'], 'Training');
  if (result.ok) {
    tab.sessionId = result.id;
    setStatus('训练已启动 / Training started');
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
  } else {
    tab.term.write(`\x1b[31m[ERROR] ${result.msg}\x1b[0m\r\n`);
    setStatus('启动失败 / Failed: ' + result.msg);
  }
});

document.getElementById('btn-stop').addEventListener('click', async () => {
  const tab = getActiveTab();
  if (tab && tab.sessionId) {
    if (confirm('确认停止当前训练？/ Stop current session?')) {
      await window.api.sessionClose(tab.sessionId);
      setStatus('已停止 / Stopped');
    }
  }
});

document.getElementById('btn-clear').addEventListener('click', () => {
  const tab = getActiveTab();
  if (tab) tab.term.clear();
  setStatus('终端已清空 / Terminal cleared');
});

// 复制选中的终端文本
document.getElementById('btn-copy').addEventListener('click', async () => {
  const tab = getActiveTab();
  if (!tab) return;
  const selected = tab.term.getSelection();
  if (selected) {
    try {
      await navigator.clipboard.writeText(selected);
      setStatus('已复制到剪贴板 / Copied to clipboard');
    } catch (e) {
      // fallback: 用textarea复制
      const ta = document.createElement('textarea');
      ta.value = selected;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setStatus('已复制 / Copied');
    }
  } else {
    setStatus('未选中文本 / No text selected (鼠标框选终端内容)');
  }
});

document.getElementById('btn-download').addEventListener('click', async () => {
  const tab = getActiveTab();
  if (!tab || !tab.sessionId) { setStatus('无活动会话 / No active session'); return; }
  const result = await window.api.logDownload(tab.sessionId);
  if (result.ok) setStatus('日志已保存 / Log saved: ' + result.path);
  else setStatus(result.msg || '保存已取消 / Save cancelled');
});

document.getElementById('btn-new-tab').addEventListener('click', () => {
  createTab('终端 Terminal ' + (tabCounter + 1));
});

document.getElementById('btn-send').addEventListener('click', sendCmd);
document.getElementById('cmd-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendCmd();
});

function sendCmd() {
  const input = document.getElementById('cmd-input');
  const cmd = input.value.trim();
  if (!cmd) return;
  const tab = getActiveTab();
  if (cmd === 'stop') {
    if (tab && tab.sessionId) window.api.sessionClose(tab.sessionId);
  } else if (cmd === 'start') {
    document.getElementById('btn-start').click();
  } else if (cmd === 'clear') {
    if (tab) tab.term.clear();
  } else if (tab && tab.sessionId) {
    window.api.ptyInput(tab.sessionId, cmd + '\r');
  }
  input.value = '';
}

// === 时钟 ===
function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2, '0');
  const m = String(now.getMinutes()).padStart(2, '0');
  const s = String(now.getSeconds()).padStart(2, '0');
  document.getElementById('clock').textContent = `${h}:${m}:${s}`;
}
setInterval(updateClock, 1000);
updateClock();

function setStatus(text) { document.getElementById('status-text').textContent = text; }

// === resize ===
window.addEventListener('resize', () => {
  resizeCharts();
  for (const t of tabs) {
    t.fitAddon.fit();
    if (t.sessionId) window.api.ptyResize(t.sessionId, t.term.cols, t.term.rows);
  }
});

// === 初始化 ===
initCharts();
createTab('终端 Terminal');
setStatus('就绪 - 点击"启动训练"或输入命令 / Ready - click Start or type commands');

window.api.getMetrics().then(m => { if (m) updateMetrics(m); });
