const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('api', {
  sessionCreate: (cmdArgs, label) => ipcRenderer.invoke('session-create', cmdArgs, label),
  sessionClose: (id) => ipcRenderer.invoke('session-close', id),
  sessionSwitch: (id) => ipcRenderer.invoke('session-switch', id),
  ptyInput: (id, data) => ipcRenderer.invoke('pty-input', id, data),
  ptyResize: (id, cols, rows) => ipcRenderer.invoke('pty-resize', id, cols, rows),
  logDownload: (id) => ipcRenderer.invoke('log-download', id),
  getMetrics: () => ipcRenderer.invoke('get-metrics'),
  onPtyData: (cb) => ipcRenderer.on('pty-data', (_, d) => cb(d)),
  onMetricsUpdate: (cb) => ipcRenderer.on('metrics-update', (_, d) => cb(d)),
  onGpuUpdate: (cb) => ipcRenderer.on('gpu-update', (_, d) => cb(d)),
  onSessionExit: (cb) => ipcRenderer.on('session-exit', (_, d) => cb(d)),
});
