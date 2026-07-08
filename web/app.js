/* ─── Media Agent WebUI — Application Logic ─── */

/* ─── SVG Icons (inline, no dependencies) ─── */
const ICONS = {
  send: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>',
  attach: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>',
  download: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
  gallery: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
  settings: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
  clear: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
  close: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
  refresh: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
  upload: '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
  check: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  alert: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  play: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
};

/* ─── State ─── */
const state = {
  agentReady: false,
  processing: false,
  clientId: 'web_' + Math.random().toString(36).slice(2, 10),
  ws: null,
  wsReconnectTimer: null,
  currentTaskMsg: null,
  previewCount: 0,
  params: { steps: 20, cfg: 7.0, width: 512, height: 512, seed: -1 },
  uploadedFile: null,
  uploadedFilePath: null,
  galleryItems: [],
  settingsOpen: false,
  galleryOpen: false,
  currentModalSrc: null,
  initAttempted: false,
  agentInfo: null,
};

/* ─── DOM Refs ─── */
const $ = (id) => document.getElementById(id);
const chat = $('chat');
const welcome = $('welcome');
const input = $('input');
const sendBtn = $('sendBtn');
const statusDot = $('statusDot');
const statusText = $('statusText');
const dropZone = $('drop-zone');
const fileInput = $('fileInput');
const uploadPreview = $('upload-preview');
const previewImg = $('previewImg');
const uploadName = $('uploadName');
const galleryGrid = $('gallery-grid');
const galleryBtn = $('galleryBtn');
const settingsBtn = $('settingsBtn');
const settingsPanel = $('settings-panel');
const galleryPanel = $('gallery-panel');
const modal = $('modal');
const modalImg = $('modalImg');
const toasts = $('toasts');

/* ─── Init ─── */
async function init() {
  updateStatus('offline', 'Checking agent…');
  await checkAgentStatus();
  connectWebSocket();
  setupEventListeners();
  loadConversation();
  loadGallery();
}

/* ─── Status Display ─── */
function updateStatus(state, text) {
  const dot = statusDot;
  dot.className = 'status-dot';
  if (state === 'online') dot.classList.add('online');
  if (state === 'error') dot.classList.add('error');
  statusText.textContent = text;
}

/* ─── Agent Status ─── */
async function checkAgentStatus() {
  try {
    const r = await fetch('/agent/status');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    state.agentReady = !!data.ready;
    state.agentInfo = data;
    if (data.ready) {
      updateStatus('online', 'Agent ready');
      loadAgentInfo(data);
    } else if (!state.initAttempted) {
      updateStatus('offline', 'Not initialized');
    }
  } catch (e) {
    updateStatus('error', 'Server unreachable');
  }
}

async function initAgent() {
  if (state.processing) return;
  updateStatus('offline', 'Initializing…');
  state.initAttempted = true;
  try {
    const r = await fetch('/agent/init', { method: 'POST' });
    const data = await r.json();
    if (data.status === 'initialized' || data.ready) {
      state.agentReady = true;
      updateStatus('online', 'Agent ready');
      addSystemMsg('Agent initialized successfully.');
      // Re-fetch status to get full info
      const sr = await fetch('/agent/status');
      if (sr.ok) {
        state.agentInfo = await sr.json();
        loadAgentInfo(state.agentInfo);
      }
    } else {
      const errMsg = data.error || data.message || 'unknown error';
      updateStatus('error', 'Init failed');
      showToast('error', 'Agent init failed: ' + errMsg);
    }
  } catch (e) {
    updateStatus('error', 'Init error');
    showToast('error', 'Agent init error: ' + e.message);
  }
}

/* ─── Agent Info Panel ─── */
function loadAgentInfo(data) {
  if (!data) return;

  // Skills
  const skillsEl = $('skills-list');
  if (skillsEl && data.skills) {
    skillsEl.innerHTML = data.skills.map(s =>
      `<span class="skill-tag">${escapeHtml(s)}</span>`
    ).join('');
  }

  // Tools
  const toolsEl = $('tools-list');
  if (toolsEl && data.tools) {
    toolsEl.innerHTML = data.tools.map(t =>
      `<span class="tool-tag">${escapeHtml(t)}</span>`
    ).join('');
  }

  // Workflows
  const wfEl = $('workflows-list');
  if (wfEl && data.workflows) {
    wfEl.innerHTML = data.workflows.map(w => {
      const name = w.split('/').pop()?.replace(/\.jsonld$/, '') || w;
      return `<div class="workflow-item" onclick="showToast('info','${escapeHtml(w)}')">
        <div class="wf-name">${escapeHtml(name.replace(/_/g, ' '))}</div>
        <div class="wf-desc">${escapeHtml(w)}</div>
      </div>`;
    }).join('');
  }
}

/* ─── WebSocket ─── */
function connectWebSocket() {
  if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) return;

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${location.host}/ws?client_id=${state.clientId}`;

  try {
    state.ws = new WebSocket(url);
  } catch (e) {
    scheduleWSReconnect();
    return;
  }

  state.ws.onopen = () => {
    if (state.wsReconnectTimer) {
      clearTimeout(state.wsReconnectTimer);
      state.wsReconnectTimer = null;
    }
  };

  state.ws.onmessage = (e) => {
    if (e.data instanceof Blob) {
      const reader = new FileReader();
      reader.onload = () => updatePreview(reader.result);
      reader.readAsDataURL(e.data);
      return;
    }
    try {
      const evt = JSON.parse(e.data);
      handleWSEvent(evt);
    } catch (_) { /* ignore non-JSON messages */ }
  };

  state.ws.onclose = () => {
    state.ws = null;
    if (!state.processing) scheduleWSReconnect();
  };

  state.ws.onerror = () => {
    if (state.ws) { state.ws.close(); state.ws = null; }
    if (!state.processing) scheduleWSReconnect();
  };
}

function scheduleWSReconnect() {
  if (state.wsReconnectTimer) return;
  state.wsReconnectTimer = setTimeout(() => {
    state.wsReconnectTimer = null;
    connectWebSocket();
  }, 3000);
}

/* ─── WebSocket Event Handler ─── */
function handleWSEvent(evt) {
  const type = evt.type || evt.event || '';
  if (type === 'ExecutionStart' || evt.ExecutionStart) {
    addProgressMsg('Generation started…');
  } else if (type === 'Progress' || evt.Progress) {
    const p = evt.Progress || evt;
    const step = p.step ?? p.value ?? 0;
    const total = p.total ?? p.max ?? 100;
    const pct = Math.round((step / total) * 100);
    updateProgress(pct, `Sampling: step ${step}/${total}`);
  } else if (type === 'Preview' || evt.Preview) {
    // Preview handled via binary blob messages
  } else if (type === 'ExecutionSuccess' || evt.ExecutionSuccess) {
    updateProgress(100, 'Complete!');
    completeAgentStage('acting', true);
    setTimeout(finalizeProgress, 600);
  } else if (type === 'ExecutionError' || evt.ExecutionError) {
    const err = evt.ExecutionError || evt;
    updateProgressError(err.message || err.error || 'Generation failed');
  } else if (type === 'Executing' || evt.Executing) {
    const ex = evt.Executing || evt;
    updateProgressLabel(ex.node ? 'Running: ' + ex.node : 'Processing…');
  } else if (type === 'AgentPhaseStart' || evt.AgentPhaseStart) {
    const d = evt.AgentPhaseStart || evt;
    updateAgentStage(d.phase, d.description);
  } else if (type === 'AgentPhaseComplete' || evt.AgentPhaseComplete) {
    const d = evt.AgentPhaseComplete || evt;
    completeAgentStage(d.phase, d.success);
  } else if (type === 'AgentThought' || evt.AgentThought) {
    const d = evt.AgentThought || evt;
    appendAgentThought(d.thought, d.action);
  } else if (type === 'AgentToolCall' || evt.AgentToolCall) {
    const d = evt.AgentToolCall || evt;
    appendToolCall(d.tool_name, d.status, d.result_summary);
  }
}

/* ─── Message Display ─── */
function addMsg(role, html) {
  removeTyping();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'agent' && !html) {
    div.classList.add('typing');
    div.innerHTML = `<div class="avatar">AI</div><div class="bubble"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  } else if (role === 'user') {
    div.innerHTML = `<div class="avatar">You</div><div class="bubble">${escapeHtml(html)}</div>`;
  } else {
    div.innerHTML = `<div class="avatar">AI</div><div class="bubble">${html}</div>`;
  }
  chat.appendChild(div);
  scrollDown();
  return div;
}

function addSystemMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg system';
  div.innerHTML = `<div class="bubble">${text}</div>`;
  chat.appendChild(div);
  scrollDown();
}

function removeTyping() {
  const typing = chat.querySelector('.msg.agent.typing');
  if (typing) typing.remove();
}

function scrollDown() {
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

/* ─── Progress Messages ─── */
function addProgressMsg(text) {
  removeTyping();
  state.previewCount = 0;
  const div = document.createElement('div');
  div.className = 'msg agent';
  div.dataset.role = 'progress';
  div.innerHTML = `<div class="avatar">AI</div><div class="bubble">
    <div class="pdca-stages">
      <div class="stage" data-phase="planning">
        <span class="stage-icon">1</span><span class="stage-label">Planning</span>
      </div>
      <div class="stage-connector"></div>
      <div class="stage" data-phase="doing">
        <span class="stage-icon">2</span><span class="stage-label">Doing</span>
      </div>
      <div class="stage-connector"></div>
      <div class="stage" data-phase="checking">
        <span class="stage-icon">3</span><span class="stage-label">Checking</span>
      </div>
      <div class="stage-connector"></div>
      <div class="stage" data-phase="acting">
        <span class="stage-icon">4</span><span class="stage-label">Acting</span>
      </div>
    </div>
    <div class="progress-msg">${escapeHtml(text)}</div>
    <div class="progress-bar"><div class="fill" style="width:0%"></div></div>
    <div class="progress-label" id="progLabel"></div>
    <div class="agent-thoughts"></div>
    <div class="tool-calls"></div>
  </div>`;
  chat.appendChild(div);
  state.currentTaskMsg = div;
  scrollDown();
}

function updateAgentStage(phase, description) {
  if (!state.currentTaskMsg) return;
  const stages = state.currentTaskMsg.querySelectorAll('.stage');
  stages.forEach(s => {
    if (s.dataset.phase === phase) {
      s.classList.add('active');
      s.classList.remove('completed', 'failed');
    }
  });
  if (description) {
    updateProgressLabel(description);
  }
}

function completeAgentStage(phase, success) {
  if (!state.currentTaskMsg) return;
  const stage = state.currentTaskMsg.querySelector(`.stage[data-phase="${phase}"]`);
  if (stage) {
    stage.classList.remove('active');
    stage.classList.add(success ? 'completed' : 'failed');
  }
}

function appendAgentThought(thought, action) {
  if (!state.currentTaskMsg || !thought) return;
  const container = state.currentTaskMsg.querySelector('.agent-thoughts');
  if (!container) return;
  const div = document.createElement('div');
  div.className = 'agent-thought';
  const actionLabel = action ? `<span class="thought-action">${escapeHtml(action)}</span> ` : '';
  div.innerHTML = `${actionLabel}<span class="thought-text">${escapeHtml(thought)}</span>`;
  container.appendChild(div);
  scrollDown();
}

function appendToolCall(toolName, status, summary) {
  if (!state.currentTaskMsg || !toolName) return;
  const container = state.currentTaskMsg.querySelector('.tool-calls');
  if (!container) return;
  const div = document.createElement('div');
  div.className = 'tool-call ' + (status || '');
  const statusIcon = status === 'completed' ? '✓' : status === 'failed' ? '✗' : '⏳';
  div.innerHTML = `<span class="tool-status">${statusIcon}</span>
    <span class="tool-name">${escapeHtml(toolName)}</span>
    ${summary ? `<span class="tool-summary">${escapeHtml(summary)}</span>` : ''}`;
  container.appendChild(div);
  scrollDown();
}

function updateProgress(pct, label) {
  const fill = state.currentTaskMsg?.querySelector('.fill');
  if (fill) fill.style.width = Math.min(pct, 100) + '%';
  const pl = state.currentTaskMsg?.querySelector('#progLabel');
  if (pl && label) pl.textContent = label;
}

function updateProgressLabel(text) {
  const pl = state.currentTaskMsg?.querySelector('#progLabel');
  if (pl && text) pl.textContent = text;
}

function updatePreview(dataUrl) {
  if (!state.currentTaskMsg) return;
  const bubble = state.currentTaskMsg.querySelector('.bubble');
  if (!bubble) return;
  let img = bubble.querySelector(`img[src="${dataUrl}"]`);
  if (!img) {
    img = document.createElement('img');
    img.className = 'gen-media-wrap';
    img.style.cssText = 'display:block;max-width:100%;margin-top:6px;border-radius:6px;cursor:pointer';
    img.src = dataUrl;
    img.onclick = () => openModal(dataUrl);
    bubble.appendChild(img);
    state.previewCount++;
    scrollDown();
  }
}

function updateProgressError(msg) {
  if (!state.currentTaskMsg) return;
  const bubble = state.currentTaskMsg.querySelector('.bubble');
  if (bubble) {
    const div = document.createElement('div');
    div.className = 'error-text';
    div.textContent = msg;
    bubble.appendChild(div);
    showToast('error', msg);
  }
}

function finalizeProgress() {
  state.currentTaskMsg = null;
}

/* ─── Send Message ─── */
async function sendMessage() {
  const text = input.value.trim();
  if (!text || state.processing) return;
  await sendToAgent(text);
}

function sendQuickPrompt(text) {
  if (!text || state.processing) return;
  sendToAgent(text);
}

async function ensureAgentReady() {
  if (state.agentReady) return true;
  try {
    const r = await fetch('/agent/status');
    if (r.ok) {
      const data = await r.json();
      if (data.ready) {
        state.agentReady = true;
        updateStatus('online', 'Agent ready');
        if (state.agentInfo) loadAgentInfo(data);
        return true;
      }
    }
  } catch (_) {}
  updateStatus('offline', 'Auto-initializing…');
  state.initAttempted = true;
  try {
    const r = await fetch('/agent/init', { method: 'POST' });
    if (!r.ok) {
      const errData = await r.json().catch(() => null);
      const errMsg = errData?.error || errData?.message || 'HTTP ' + r.status;
      updateStatus('error', 'Init failed');
      showToast('error', 'Agent init failed: ' + errMsg);
      return false;
    }
    const data = await r.json();
    if (data.status === 'initialized' || data.ready) {
      state.agentReady = true;
      updateStatus('online', 'Agent ready');
      addSystemMsg('Agent auto-initialized.');
      try {
        const sr = await fetch('/agent/status');
        if (sr.ok) {
          state.agentInfo = await sr.json();
          loadAgentInfo(state.agentInfo);
        }
      } catch (_) {}
      return true;
    }
    updateStatus('error', 'Init failed');
    return false;
  } catch (e) {
    updateStatus('error', 'Init error');
    showToast('error', 'Agent init error: ' + e.message);
    return false;
  }
}

async function sendToAgent(text) {
  state.processing = true;
  sendBtn.disabled = true;

  // Hide welcome, add user message
  welcome.classList.add('hidden');
  addMsg('user', text);
  addMsg('agent', ''); // typing indicator

  const ready = await ensureAgentReady();
  if (!ready) {
    removeTyping();
    addSystemMsg('Agent is not ready. Click Init in settings or check server configuration.');
    state.processing = false;
    sendBtn.disabled = !input.value.trim();
    clearUpload();
    return;
  }

  // Build payload
  const payload = {
    message: text,
    client_id: state.clientId,
    params: state.params
  };

  // Attach uploaded image if present
  if (state.uploadedFilePath) {
    payload.image_path = state.uploadedFilePath;
  }

  try {
    const r = await fetch('/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!r.ok) {
      let errMsg = 'Server returned ' + r.status;
      try {
        const errData = await r.json();
        errMsg = errData.message || errData.error || errMsg;
      } catch (_) {
        const errText = await r.text().catch(() => '');
        if (errText) errMsg = errText.slice(0, 200);
      }
      removeTyping();
      showToast('error', errMsg);
      addSystemMsg('Error: ' + errMsg);
      state.processing = false;
      sendBtn.disabled = !input.value.trim();
      clearUpload();
      return;
    }

    const data = await r.json().catch(() => null);
    removeTyping();

    if (!data) {
      addSystemMsg('Error: Empty response from server');
    } else if (data.error) {
      addSystemMsg('Error: ' + (data.message || data.error));
    } else {
      showAgentResult(data);
      // Refresh gallery after generation
      loadGallery();
    }
  } catch (err) {
    removeTyping();
    addSystemMsg('Network error: ' + err.message);
    showToast('error', 'Network error');
  } finally {
    state.processing = false;
    sendBtn.disabled = !input.value.trim();
    clearUpload();
    saveConversation();
  }
}

/* ─── Display Agent Result ─── */
function showAgentResult(data) {
  const summary = data.summary || '';
  let outputHtml = '';
  const output = data.output;

  if (output) {
    if (typeof output === 'string') {
      // Could be a file path
      if (output.match(/\.(png|jpg|jpeg|gif|mp4|webm)$/i)) {
        outputHtml = renderMedia(output);
      } else {
        outputHtml = '<pre>' + escapeHtml(output) + '</pre>';
      }
    } else if (Array.isArray(output)) {
      const media = output.filter(item =>
        typeof item === 'string' && item.match(/\.(png|jpg|jpeg|gif|mp4|webm)$/i)
      );
      if (media.length > 0) {
        outputHtml = '<div class="media-grid">' + media.map(m => renderMedia(m)).join('') + '</div>';
      }
      const nonMedia = output.filter(item => typeof item !== 'string' || !item.match(/\.(png|jpg|jpeg|gif|mp4|webm)$/i));
      if (nonMedia.length > 0) {
        outputHtml += nonMedia.map(item =>
          '<pre>' + escapeHtml(typeof item === 'string' ? item : JSON.stringify(item, null, 2)) + '</pre>'
        ).join('');
      }
    } else if (typeof output === 'object') {
      const images = findImagePaths(output);
      if (images.length > 0) {
        if (images.length === 1) {
          outputHtml = renderMedia(images[0]);
        } else {
          outputHtml = '<div class="media-grid">' + images.map(m => renderMedia(m)).join('') + '</div>';
        }
      }
      // Show remaining output as details
      const rest = { ...output };
      ['images','files','image','video','media','result','output'].forEach(k => delete rest[k]);
      const restStr = JSON.stringify(rest, null, 2);
      if (restStr !== '{}') {
        outputHtml += '<pre>' + escapeHtml(restStr) + '</pre>';
      }
    }
  }

  // Fallback: try loading from history
  if (!outputHtml) {
    addSystemMsg(summary || 'Task completed.');
    loadRecentOutputs();
    return;
  }

  const summaryHtml = summary ? '<p>' + escapeHtml(summary) + '</p>' : '';
  addMsg('agent', summaryHtml + outputHtml);
}

function renderMedia(path) {
  const isVideo = !!path.match(/\.(mp4|webm)$/i);
  const filename = path.split('/').pop() || path;
  const subfolder = path.split('/').slice(0, -1).join('/');
  const url = '/view?filename=' + encodeURIComponent(filename)
    + (subfolder ? '&subfolder=' + encodeURIComponent(subfolder) : '');
  const dlUrl = url + '&download=1';

  if (isVideo) {
    return `<div class="gen-media-wrap">
      <video controls preload="metadata" style="max-width:100%;display:block">
        <source src="${url}" type="video/mp4">
      </video>
      <button class="dl-btn" onclick="downloadFile('${dlUrl}','${filename}')" title="Download">
        ${ICONS.download}
      </button>
    </div>`;
  }
  return `<div class="gen-media-wrap">
    <img src="${url}" onclick="openModal('${url}')" alt="Generated image" loading="lazy">
    <button class="dl-btn" onclick="downloadFile('${dlUrl}','${filename}')" title="Download">
      ${ICONS.download}
    </button>
  </div>`;
}

function findImagePaths(obj, depth = 0) {
  if (depth > 4 || !obj || typeof obj !== 'object') return [];
  const results = [];
  for (const val of Object.values(obj)) {
    if (typeof val === 'string' && val.match(/\.(png|jpg|jpeg|gif|mp4|webm)$/i)) {
      results.push(val);
    } else if (typeof val === 'object' && val) {
      results.push(...findImagePaths(val, depth + 1));
    }
  }
  return results;
}

/* ─── Download Support ─── */
async function downloadFile(url, filename) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename || 'download';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    showToast('success', 'Downloaded ' + filename);
  } catch (e) {
    showToast('error', 'Download failed: ' + e.message);
  }
}

function downloadCurrent() {
  if (state.currentModalSrc) {
    const filename = state.currentModalSrc.split('filename=')[1]?.split('&')[0] || 'image.png';
    downloadFile(state.currentModalSrc, decodeURIComponent(filename));
  }
}

/* ─── Image Upload ─── */
function openFilePicker() {
  fileInput.click();
}

function handleFileSelect(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('warning', 'Please select an image file');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('warning', 'Image too large (max 10MB)');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadName.textContent = file.name;
    uploadPreview.classList.remove('hidden');
    uploadPreview.style.display = 'flex';
    state.uploadedFile = file;
  };
  reader.readAsDataURL(file);

  // Upload to server
  uploadFile(file);
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('image', file);

  try {
    const r = await fetch('/upload/image', {
      method: 'POST',
      body: formData
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    state.uploadedFilePath = data.filename || data.name || data.image || file.name;
    showToast('success', 'Image uploaded');
  } catch (e) {
    showToast('error', 'Upload failed: ' + e.message);
    clearUpload();
  }
}

function clearUpload() {
  state.uploadedFile = null;
  state.uploadedFilePath = null;
  uploadPreview.style.display = 'none';
  uploadPreview.classList.add('hidden');
  previewImg.src = '';
  uploadName.textContent = '';
  fileInput.value = '';
}

/* ─── Event Listeners ─── */
function setupEventListeners() {
  // Input
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener('input', () => {
    sendBtn.disabled = !input.value.trim() || state.processing;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 100) + 'px';
  });

  // File input
  fileInput.addEventListener('change', (e) => {
    if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
  });

  // Drag and drop
  document.addEventListener('dragenter', (e) => {
    e.preventDefault();
    if (e.dataTransfer?.types.includes('Files')) {
      dropZone.classList.add('drag-over');
    }
  });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('dragleave', (e) => {
    if (!e.relatedTarget || e.relatedTarget === document) {
      dropZone.classList.remove('drag-over');
    }
  });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer?.files?.[0]) handleFileSelect(e.dataTransfer.files[0]);
  });

  // Escape closes modals and panels
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeModal();
      if (state.settingsOpen) toggleSettings();
      if (state.galleryOpen) toggleGallery();
    }
  });
}

/* ─── Panel Toggles ─── */
function toggleSettings() {
  state.settingsOpen = !state.settingsOpen;
  settingsPanel.classList.toggle('open', state.settingsOpen);
  settingsBtn.classList.toggle('active', state.settingsOpen);
  if (state.settingsOpen && state.galleryOpen) toggleGallery();
}

function toggleGallery() {
  state.galleryOpen = !state.galleryOpen;
  galleryPanel.classList.toggle('open', state.galleryOpen);
  galleryBtn.classList.toggle('active', state.galleryOpen);
  if (state.galleryOpen && state.settingsOpen) toggleSettings();
  if (state.galleryOpen) loadGallery();
}

/* ─── Gallery ─── */
async function loadGallery() {
  try {
    const r = await fetch('/history');
    if (!r.ok) return;
    const data = await r.json();
    const history = data.history || data;
    const entries = Array.isArray(history) ? history : Object.values(history);

    if (!entries.length) {
      galleryGrid.innerHTML = '<div class="gallery-empty">No generated images yet</div>';
      return;
    }

    const allOutputs = [];
    for (const entry of entries) {
      const outputs = entry.outputs || entry.result || {};
      for (const val of Object.values(outputs)) {
        if (typeof val === 'string' && val.match(/\.(png|jpg|jpeg|gif|mp4|webm)$/i)) {
          allOutputs.push(val);
        } else if (Array.isArray(val)) {
          val.forEach(v => {
            if (typeof v === 'string' && v.match(/\.(png|jpg|jpeg|gif|mp4|webm)$/i)) allOutputs.push(v);
          });
        }
      }
    }

    // Reverse so newest first
    allOutputs.reverse();
    state.galleryItems = allOutputs;

    if (!allOutputs.length) {
      galleryGrid.innerHTML = '<div class="gallery-empty">No generated images yet</div>';
      return;
    }

    galleryGrid.innerHTML = allOutputs.map(path => {
      const isVideo = !!path.match(/\.(mp4|webm)$/i);
      const filename = path.split('/').pop() || path;
      const subfolder = path.split('/').slice(0, -1).join('/');
      const url = '/view?filename=' + encodeURIComponent(filename)
        + (subfolder ? '&subfolder=' + encodeURIComponent(subfolder) : '');

      return `<div class="gallery-item" onclick="${isVideo ? '' : "openModal('" + url + "')"}">
        ${isVideo
          ? `<video preload="metadata"><source src="${url}" type="video/mp4"></video><div class="gallery-overlay">${ICONS.play}</div>`
          : `<img src="${url}" alt="Generated" loading="lazy">`
        }
      </div>`;
    }).join('');
  } catch (_) {
    galleryGrid.innerHTML = '<div class="gallery-empty">Failed to load gallery</div>';
  }
}

/* ─── Load Recent Outputs (fallback) ─── */
async function loadRecentOutputs() {
  try {
    const r = await fetch('/history?max_items=1');
    if (!r.ok) return;
    const data = await r.json();
    const history = data.history || data;
    const entries = Array.isArray(history) ? history : Object.values(history);
    if (!entries.length) return;

    const latest = entries[entries.length - 1];
    const outputs = latest.outputs || latest.result || {};
    const media = Object.values(outputs).flat().filter(v =>
      typeof v === 'string' && v.match(/\.(png|jpg|jpeg|gif|mp4|webm)$/i)
    );
    if (media.length > 0) {
      const html = '<div class="media-grid">' + media.map(m => renderMedia(m)).join('') + '</div>';
      addMsg('agent', html);
    }
  } catch (_) {}
}

/* ─── Clear & Reset ─── */
function clearChat() {
  if (chat.children.length === 0) return;
  if (!confirm('Clear all messages?')) return;
  chat.innerHTML = '';
  welcome.classList.remove('hidden');
  state.currentTaskMsg = null;
  // Save empty conversation
  localStorage.removeItem('media_agent_conversation');
}

/* ─── Conversation Persistence ─── */
function saveConversation() {
  try {
    const msgs = [];
    chat.querySelectorAll('.msg:not(.system)').forEach(el => {
      const role = el.classList.contains('user') ? 'user' : 'agent';
      const text = el.querySelector('.bubble')?.textContent?.trim() || '';
      if (text) msgs.push({ role, text: text.slice(0, 500) });
    });
    // Only save last 20 messages
    const recent = msgs.slice(-20);
    localStorage.setItem('media_agent_conversation', JSON.stringify(recent));
  } catch (_) {}
}

function loadConversation() {
  try {
    const saved = localStorage.getItem('media_agent_conversation');
    if (!saved) return;
    const msgs = JSON.parse(saved);
    if (!msgs.length) return;
    welcome.classList.add('hidden');
    addSystemMsg('Restored ' + msgs.length + ' messages');
    msgs.forEach(m => {
      if (m.role === 'user') {
        addMsg('user', m.text);
      }
    });
  } catch (_) {}
}

/* ─── Image Modal ─── */
function openModal(src) {
  state.currentModalSrc = src;
  modalImg.src = src;
  modal.classList.add('open');
}

function closeModal() {
  modal.classList.remove('open');
  state.currentModalSrc = null;
}

/* ─── Toast Notifications ─── */
function showToast(type, message) {
  const iconMap = { success: ICONS.check, error: ICONS.alert, warning: ICONS.alert, info: ICONS.alert };
  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.innerHTML = `<span class="toast-icon">${iconMap[type] || ICONS.alert}</span>${escapeHtml(message)}`;
  toasts.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('out');
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

/* ─── Utility ─── */
function escapeHtml(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

/* ─── Parameter Controls (wired from settings panel) ─── */
function updateParam(key, value) {
  state.params[key] = key === 'cfg' || key === 'steps' ? parseFloat(value) : parseInt(value, 10);
  // Update display
  const display = document.getElementById('param-' + key);
  if (display) display.textContent = value;
}

/* ─── Start ─── */
document.addEventListener('DOMContentLoaded', init);
