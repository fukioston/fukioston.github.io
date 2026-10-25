// 棋类游戏共用的 DeepSeek 模型配置：下拉选项、思考强度、请求参数。
(function (root) {
  const MODELS = [
    { id: 'deepseek-v4-pro', label: 'V4 Pro · 最强' },
    { id: 'deepseek-flash', label: 'V4.1 Flash · 快' },
    { id: 'deepseek-reasoner', label: 'Flash · 旧接口，深度思考' },
    { id: 'deepseek-chat', label: 'Flash · 不思考，最快' },
  ];
  const EFFORTS = [
    { id: 'none', label: '不思考' },
    { id: 'low', label: '思考：快' },
    { id: 'high', label: '思考：标准' },
    { id: 'max', label: '思考：最深' },
  ];
  const DEFAULT_MODEL = 'deepseek-v4-pro', DEFAULT_EFFORT = 'high';

  const hasEffort = m => /v4|flash/i.test(m);              // 支持 reasoning_effort 的新模型
  const thinks = (m, effort) => (hasEffort(m) && effort !== 'none') || /reasoner/i.test(m);  // 这次请求会不会思考
  const jsonMode = m => !/reasoner/i.test(m);               // reasoner 不支持 json_object

  const options = () => MODELS.map(m => `<option value="${m.id}">${m.label}</option>`).join('') + '<option value="__custom">其他模型…</option>';
  const effortOptions = () => EFFORTS.map(e => `<option value="${e.id}">${e.label}</option>`).join('');
  const effortLabel = id => (EFFORTS.find(e => e.id === id) || {}).label?.replace('思考：', '') || id;  // 快 / 标准 / 最深 / 不思考

  // 生成 /api/chat 的请求体
  function body(model, messages, { temperature, effort } = {}) {
    const b = { model, messages };
    if (thinks(model, effort)) b.max_tokens = effort === 'max' ? 64000 : 32000;
    else { b.max_tokens = 800; if (!isNaN(temperature)) b.temperature = temperature; }
    if (jsonMode(model)) b.response_format = { type: 'json_object' };
    if (hasEffort(model) && effort) b.reasoning_effort = effort;
    return b;
  }
  // 棋谱里显示的模型名，带上思考强度
  const display = (model, effort) => hasEffort(model) && effort ? `${model} · ${effortLabel(effort)}` : model;

  // 把一方的模型下拉框、思考强度、温度输入框接好；onChange 在选择变化后调用
  function wire(side, $, onChange) {
    const sel = $(side + 'Model'), eff = $(side + 'Effort'), temp = $(side + 'Temp');
    sel.innerHTML = options();
    sel.value = DEFAULT_MODEL;
    eff.innerHTML = effortOptions();
    eff.value = DEFAULT_EFFORT;
    const sync = () => {
      eff.hidden = !hasEffort(sel.value);
      temp.hidden = thinks(sel.value, eff.value);  // 温度只对不思考的请求有效
    };
    sel.dataset.prev = sel.value;
    sel.addEventListener('change', () => {
      if (sel.value === '__custom') {
        const name = (prompt('输入 DeepSeek 模型名', '') || '').trim();
        if (name) set(side, $, name); else sel.value = sel.dataset.prev;
      } else sel.dataset.prev = sel.value;
      sync(); onChange && onChange();
    });
    eff.addEventListener('change', () => { sync(); onChange && onChange(); });
    sync();
  }
  function set(side, $, name) {
    const sel = $(side + 'Model');
    if (![...sel.options].some(o => o.value === name)) sel.add(new Option(name, name), sel.options[sel.options.length - 1]);
    sel.value = name;
    sel.dataset.prev = name;
    $(side + 'Effort').hidden = !hasEffort(name);
    $(side + 'Temp').hidden = thinks(name, $(side + 'Effort').value);
  }

  // ---------- 账户余额：不足时在顶栏下方提示 ----------
  const LOW = 5, OUT = 1; // 元
  let lastCheck = 0, dismissed = false;
  async function checkBalance() {
    lastCheck = Date.now();
    let b;
    try { b = await (await fetch('/api/balance', { cache: 'no-store' })).json(); } catch { return null; }
    renderBalance(b);
    return b;
  }
  // 对局中每隔几分钟顺手查一次
  const maybeCheckBalance = () => { if (Date.now() - lastCheck > 180000) checkBalance(); };
  function renderBalance(b) {
    const money = b.total != null ? (b.currency === 'USD' ? '$' : '¥') + b.total.toFixed(2) : '';
    const line = document.getElementById('balanceLine');
    if (line) line.textContent = !b.configured ? '未配置 API Key' : b.error ? '查询失败' : money + (b.available === false ? '（不可用）' : '');
    let msg = '', level = 'err', link = '<a href="https://platform.deepseek.com/top_up" target="_blank" rel="noopener">去充值 →</a>';
    if (!b.configured) { msg = '还没有配置 DeepSeek API Key：点右上角「🔑 API Key」填入你自己的 Key，AI 才能下棋（双人对弈不需要）。'; link = ''; }
    else if (b.auth) { msg = 'DeepSeek API Key 无效，请点右上角「🔑 API Key」检查。'; link = '<a href="https://platform.deepseek.com/api_keys" target="_blank" rel="noopener">管理 Key →</a>'; }
    else if (b.error) msg = '';
    else if (b.available === false || b.total < OUT) msg = `DeepSeek 余额不足（剩 ${money}），AI 随时可能无法走棋。`;
    else if (b.total < LOW) { msg = `DeepSeek 余额偏低（剩 ${money}），记得充值。`; level = 'low'; }
    let el = document.getElementById('quotaWarn');
    if (!msg || (dismissed && level === 'low')) { if (el) el.hidden = true; return; }
    if (!el) {
      el = document.createElement('div');
      el.id = 'quotaWarn';
      (document.querySelector('header.topbar') || document.body.firstChild).after(el);
    }
    el.className = 'quota-warn ' + level;
    el.hidden = false;
    el.innerHTML = `<span>⚠ ${msg}</span>${link}<button type="button" title="先不提醒">×</button>`;
    el.querySelector('button').onclick = () => { dismissed = true; el.hidden = true; };
  }
  // 请求失败时判断是不是余额/Key 问题（服务端会带上 code）
  const isAccountError = e => e && (e.code === 'balance' || e.code === 'auth');

  root.Models = { MODELS, EFFORTS, DEFAULT_MODEL, hasEffort, thinks, body, display, wire, set,
                  checkBalance, maybeCheckBalance, isAccountError };
})(this);
