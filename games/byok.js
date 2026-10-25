// 静态托管版：自带 DeepSeek Key（BYOK）。把页面原本发往本地 server.py 的 /api/* 请求
// 改成浏览器直连 DeepSeek；Key 只保存在本浏览器 localStorage，只发送给 api.deepseek.com。
(function () {
  const BASE = 'https://api.deepseek.com';
  const KEY_LS = 'deepseek_api_key', GAMES_LS = 'arcade_games_v1';
  const getKey = () => localStorage.getItem(KEY_LS) || '';
  const json = (status, obj) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } });
  const loadGames = () => { try { return JSON.parse(localStorage.getItem(GAMES_LS)) || {}; } catch { return {}; } };
  const origFetch = window.fetch.bind(window);

  async function chat(init) {
    const key = getKey();
    if (!key) { setTimeout(openDialog, 0); return json(401, { error: '还没有配置 DeepSeek API Key：点右上角「🔑 API Key」填入你自己的 Key', code: 'auth' }); }
    let body; try { body = JSON.parse(init.body || '{}'); } catch { return json(400, { error: '请求体不是合法 JSON' }); }
    const payload = { model: body.model || 'deepseek-chat', messages: body.messages || [], stream: true };
    for (const k of ['temperature', 'response_format', 'max_tokens', 'reasoning_effort']) if (k in body) payload[k] = body[k];
    let resp;
    try {
      resp = await origFetch(BASE + '/chat/completions', { method: 'POST', signal: init.signal,
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + key }, body: JSON.stringify(payload) });
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      return json(502, { error: '请求 DeepSeek 失败: ' + e.message });
    }
    if (!resp.ok) {
      const detail = (await resp.text().catch(() => '')).slice(0, 500);
      if (resp.status === 402) return json(402, { error: 'DeepSeek 账户余额不足，充值后点「继续」接着下', code: 'balance' });
      if (resp.status === 401) return json(401, { error: 'DeepSeek API Key 无效，请点右上角「🔑 API Key」检查', code: 'auth' });
      return json(502, { error: `DeepSeek HTTP ${resp.status}: ${detail}` });
    }
    // SSE → 逐行 JSON：{"r"} 思考增量 / {"c"} 回答增量 / {"done":true}
    const reader = resp.body.getReader(), dec = new TextDecoder(), enc = new TextEncoder();
    let buf = '';
    const stream = new ReadableStream({
      async pull(ctl) {
        const emit = o => ctl.enqueue(enc.encode(JSON.stringify(o) + '\n'));
        try {
          const { value, done } = await reader.read();
          if (done) { emit({ done: true }); ctl.close(); return; }
          buf += dec.decode(value, { stream: true });
          let i;
          while ((i = buf.indexOf('\n')) >= 0) {
            const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
            if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim();
            if (data === '[DONE]') { emit({ done: true }); ctl.close(); reader.cancel().catch(() => {}); return; }
            const ch = JSON.parse(data);
            const d = (ch.choices && ch.choices[0] && ch.choices[0].delta) || {};
            if (d.reasoning_content) emit({ r: d.reasoning_content });
            if (d.content) emit({ c: d.content });
          }
        } catch (e) { if (e.name === 'AbortError') { ctl.error(e); return; } emit({ error: '读取 DeepSeek 流失败: ' + e.message }); ctl.close(); }
      },
      cancel() { reader.cancel().catch(() => {}); },
    });
    return new Response(stream, { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } });
  }

  async function balance() {
    const key = getKey();
    if (!key) return json(200, { configured: false });
    try {
      const r = await origFetch(BASE + '/user/balance', { headers: { Authorization: 'Bearer ' + key } });
      if (!r.ok) return json(200, { configured: true, error: 'HTTP ' + r.status, auth: r.status === 401 });
      const d = await r.json(), infos = d.balance_infos || [];
      const info = infos.find(i => i.currency === 'CNY') || infos[0] || {};
      return json(200, { configured: true, available: d.is_available !== false, currency: info.currency || 'CNY', total: parseFloat(info.total_balance || 0) });
    } catch (e) { return json(200, { configured: true, error: String(e.message || e) }); }
  }

  window.fetch = function (input, init = {}) {
    const url = typeof input === 'string' ? input : input.url;
    const path = new URL(url, location.href).pathname;
    const method = (init.method || 'GET').toUpperCase();
    if (path === '/api/chat' && method === 'POST') return chat(init);
    if (path === '/api/balance') return balance();
    if (path === '/api/games') {
      const g = loadGames();
      if (method === 'POST') {
        try {
          const rec = JSON.parse(init.body); if (!/^[0-9A-Za-z_-]{1,64}$/.test(String(rec.id))) return Promise.resolve(json(400, { error: '对局 id 不合法' }));
          g[rec.id] = rec;
          const ids = Object.keys(g).sort().reverse(); ids.slice(30).forEach(id => delete g[id]); // 最多保留 30 局
          localStorage.setItem(GAMES_LS, JSON.stringify(g));
          return Promise.resolve(json(200, { ok: true }));
        } catch (e) { return Promise.resolve(json(500, { error: '本地保存失败: ' + e.message })); }
      }
      const list = Object.values(g).sort((a, b) => String(b.id).localeCompare(String(a.id))).map(x => ({
        type: x.type, id: x.id, startedAt: x.startedAt, updatedAt: x.updatedAt, redModel: x.redModel, blackModel: x.blackModel,
        whiteModel: x.whiteModel, result: x.result, plies: (x.moves || []).length }));
      return Promise.resolve(json(200, list));
    }
    if (path.startsWith('/api/games/')) {
      const rec = loadGames()[decodeURIComponent(path.slice(11))];
      return Promise.resolve(rec ? json(200, rec) : json(404, { error: '找不到这局记录' }));
    }
    return origFetch(input, init);
  };

  // ---------- 设置对话框 ----------
  let dlg;
  function openDialog() {
    if (!dlg) {
      dlg = document.createElement('dialog');
      dlg.id = 'byokDialog';
      dlg.innerHTML = `<form method="dialog" class="byok-box">
        <h3>DeepSeek API Key</h3>
        <p>AI 对弈需要你自己的 <a href="https://platform.deepseek.com/api_keys" target="_blank" rel="noopener">DeepSeek API Key</a>。双人对弈不需要。</p>
        <input id="byokInput" type="password" autocomplete="off" spellcheck="false" placeholder="sk-...">
        <p class="byok-note">🔒 Key 只保存在你这个浏览器（localStorage），只会发送给 api.deepseek.com，不会经过本网站服务器。</p>
        <p class="byok-status" id="byokStatus"></p>
        <div class="byok-btns"><button type="button" id="byokClear" class="ghost">清除 Key</button><span style="flex:1"></span>
        <button value="cancel" class="ghost">取消</button><button type="button" id="byokSave">保存</button></div></form>`;
      document.body.appendChild(dlg);
      const inp = dlg.querySelector('#byokInput'), st = dlg.querySelector('#byokStatus');
      dlg.querySelector('#byokSave').onclick = () => {
        const v = inp.value.trim(); if (!v) { st.textContent = '请输入 Key'; return; }
        localStorage.setItem(KEY_LS, v); inp.value = ''; dlg.close(); refresh();
      };
      dlg.querySelector('#byokClear').onclick = () => { localStorage.removeItem(KEY_LS); inp.value = ''; refresh(); st.textContent = '已清除'; };
    }
    dlg.querySelector('#byokStatus').textContent = getKey() ? '已保存 Key：' + getKey().slice(0, 5) + '••••' + getKey().slice(-4) : '尚未设置';
    dlg.showModal();
  }
  function refresh() {
    const b = document.getElementById('byokBtn'); if (b) b.textContent = getKey() ? '🔑 Key 已设置' : '🔑 API Key';
    if (window.Models && Models.checkBalance) Models.checkBalance();
  }
  window.BYOK = { open: openDialog, getKey };
  document.addEventListener('DOMContentLoaded', () => {
    const st = document.createElement('style');
    st.textContent = `#byokBtn{margin-left:auto;margin-right:8px}
      #byokDialog{border:0;border-radius:14px;padding:0;max-width:440px;width:92vw;background:#1b1830;color:#eee;box-shadow:0 20px 60px #0008}
      #byokDialog::backdrop{background:#0009}.byok-box{padding:20px 22px}.byok-box h3{margin:0 0 8px}
      .byok-box a{color:#7cd4ff}.byok-box input{width:100%;box-sizing:border-box;padding:9px 10px;border-radius:8px;border:1px solid #555;background:#0e0c1c;color:#fff;font-family:monospace}
      .byok-note{font-size:12px;opacity:.8}.byok-status{font-size:12px;color:#9fe39f;min-height:1em}
      .byok-btns{display:flex;gap:8px}.byok-btns button{padding:7px 14px;border-radius:8px;border:0;background:#33e6ff;color:#111;font-weight:600;cursor:pointer}
      .byok-btns button.ghost{background:transparent;color:#ddd;border:1px solid #555}`;
    document.head.appendChild(st);
    const home = document.querySelector('header.topbar .home-btn');
    if (home) {
      const b = document.createElement('button'); b.id = 'byokBtn'; b.type = 'button'; b.className = 'home-btn'; b.onclick = openDialog;
      home.before(b); refresh();
    }
  });
})();
