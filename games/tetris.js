// 俄罗斯方块：现代规则（7 袋随机、SRS 旋转与踢墙、暂存、5 块预览、落点阴影、锁定延迟、T-Spin、连击、背靠背）。
(function () {
  const $ = id => document.getElementById(id);
  const cv = $('cv'), ctx = cv.getContext('2d');
  const COLS = 10, ROWS = 20, HID = 2, CELL = 30;
  const W = 660, H = 640, BX = 180, BY = 20;   // 画布逻辑尺寸、棋盘左上角
  const LOCK_DELAY = .5, MAX_RESETS = 15, DAS = .16, ARR = .035, CLEAR_T = .32;

  const COLORS = { I: '#33e6ff', J: '#4d7cff', L: '#ff9a2e', O: '#ffd93a', S: '#4ee26b', T: '#b45cff', Z: '#ff4d6d' };
  const LIGHT = {};
  for (const [k, c] of Object.entries(COLORS)) LIGHT[k] = mix(c, '#ffffff', .45);
  function mix(a, b, t) {
    const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    const [x, y] = [p(a), p(b)];
    return '#' + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join('');
  }

  // 方块形状（y 向下），JLSTZ 在 3×3 内旋转，I 在 4×4 内，O 不转
  const BASE = {
    I: [[0, 1], [1, 1], [2, 1], [3, 1]], J: [[0, 0], [0, 1], [1, 1], [2, 1]], L: [[2, 0], [0, 1], [1, 1], [2, 1]],
    O: [[1, 0], [2, 0], [1, 1], [2, 1]], S: [[1, 0], [2, 0], [0, 1], [1, 1]], T: [[1, 0], [0, 1], [1, 1], [2, 1]],
    Z: [[0, 0], [1, 0], [1, 1], [2, 1]],
  };
  const SHAPES = {};
  for (const [k, cells] of Object.entries(BASE)) {
    const n = k === 'I' ? 4 : 3;
    SHAPES[k] = [cells];
    for (let r = 1; r < 4; r++) SHAPES[k].push(k === 'O' ? cells : SHAPES[k][r - 1].map(([x, y]) => [n - 1 - y, x]));
  }
  // SRS 踢墙表（标准写法 y 向上，使用时取反）
  const KICK = {
    '01': [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]], '10': [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
    '12': [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]], '21': [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
    '23': [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]], '32': [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
    '30': [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]], '03': [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  };
  const KICK_I = {
    '01': [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]], '10': [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
    '12': [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]], '21': [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
    '23': [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]], '32': [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
    '30': [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]], '03': [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
  };

  let state = 'menu', G = null;
  let best = 0;
  try { best = +localStorage.getItem('tetris-best') || 0; } catch {}

  function newGame() {
    G = {
      board: Array.from({ length: ROWS + HID }, () => Array(COLS).fill(null)),
      bag: [], queue: [], cur: null, hold: null, canHold: true,
      score: 0, lines: 0, level: 1, combo: -1, b2b: false,
      fallT: 0, lockT: 0, resets: 0, lowest: 0, lastRotate: false, kick: 0,
      clearing: null, pops: [], parts: [], shake: 0, flash: 0,
    };
    refill();
    spawn(G.queue.shift());
  }
  function refill() {
    while (G.queue.length < 6) {
      if (!G.bag.length) { G.bag = 'IJLOSTZ'.split(''); for (let i = 6; i > 0; i--) { const j = Math.random() * (i + 1) | 0; [G.bag[i], G.bag[j]] = [G.bag[j], G.bag[i]]; } }
      G.queue.push(G.bag.pop());
    }
  }
  function spawn(type) {
    refill();
    G.cur = { type, rot: 0, x: 3, y: 0 };
    G.fallT = 0; G.lockT = 0; G.resets = 0; G.lowest = 0; G.lastRotate = false;
    if (collide(G.cur, 0, 0)) gameOver();
    else if (!collide(G.cur, 0, 1)) G.cur.y++; // 出生后立刻下落一格进入可见区
  }
  const cells = (p, rot = p.rot) => SHAPES[p.type][rot];
  function collide(p, dx, dy, rot = p.rot) {
    for (const [cx, cy] of cells(p, rot)) {
      const x = p.x + cx + dx, y = p.y + cy + dy;
      if (x < 0 || x >= COLS || y >= ROWS + HID) return true;
      if (y >= 0 && G.board[y][x]) return true;
    }
    return false;
  }
  const grounded = () => collide(G.cur, 0, 1);
  function afterMove() {
    if (grounded() && G.resets < MAX_RESETS) { G.lockT = 0; G.resets++; }
  }
  function move(dx) {
    if (!G.cur || G.clearing || collide(G.cur, dx, 0)) return false;
    G.cur.x += dx; G.lastRotate = false; afterMove();
    FX.sound('move');
    return true;
  }
  function rotate(dir) {
    const p = G.cur;
    if (!p || G.clearing || p.type === 'O') return;
    const to = (p.rot + dir + 4) % 4, table = (p.type === 'I' ? KICK_I : KICK)['' + p.rot + to];
    for (let i = 0; i < table.length; i++) {
      const [kx, ky] = table[i];
      if (!collide(p, kx, -ky, to)) {
        p.x += kx; p.y -= ky; p.rot = to;
        G.lastRotate = true; G.kick = i;
        afterMove();
        FX.sound('move');
        return;
      }
    }
  }
  function hardDrop() {
    if (!G.cur || G.clearing) return;
    let n = 0;
    while (!collide(G.cur, 0, 1)) { G.cur.y++; n++; }
    if (n) G.lastRotate = false;
    G.score += n * 2;
    // 硬降拖尾
    for (const [cx, cy] of cells(G.cur)) for (let k = 0; k < 3; k++) {
      G.parts.push({ x: BX + (G.cur.x + cx + Math.random()) * CELL, y: BY + (G.cur.y + cy - HID) * CELL - Math.random() * n * CELL * .5,
                     vx: 0, vy: 60 + Math.random() * 120, life: .35, max: .35, c: COLORS[G.cur.type], s: 2 + Math.random() * 2 });
    }
    G.shake = Math.min(.25, .06 + n * .01);
    lock();
  }
  function holdPiece() {
    if (!G.cur || G.clearing || !G.canHold) return;
    const t = G.cur.type;
    if (G.hold) spawn(G.hold); else spawn(G.queue.shift());
    G.hold = t; G.canHold = false;
    FX.sound('move');
  }

  // T-Spin：T 块最后一步是旋转，且中心四角至少占了三个
  function isTSpin() {
    const p = G.cur;
    if (p.type !== 'T' || !G.lastRotate) return false;
    const cx = p.x + 1, cy = p.y + 1;
    let n = 0;
    for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      const x = cx + dx, y = cy + dy;
      if (x < 0 || x >= COLS || y >= ROWS + HID || (y >= 0 && G.board[y][x])) n++;
    }
    return n >= 3;
  }
  function lock() {
    const p = G.cur, tspin = isTSpin();
    let allHidden = true;
    for (const [cx, cy] of cells(p)) {
      const y = p.y + cy;
      if (y >= 0) G.board[y][p.x + cx] = p.type;
      if (y >= HID) allHidden = false;
    }
    G.cur = null;
    if (allHidden) { gameOver(); return; }
    const full = [];
    for (let y = 0; y < ROWS + HID; y++) if (G.board[y].every(Boolean)) full.push(y);
    score(full.length, tspin);
    if (full.length) { G.clearing = { rows: full, t: 0 }; burstRows(full); }
    else { FX.sound(tspin ? 'check' : 'capture'); G.canHold = true; spawn(G.queue.shift()); }
  }
  function score(n, tspin) {
    const L = G.level;
    let pts = tspin ? [400, 800, 1200, 1600][n] : [0, 100, 300, 500, 800][n];
    const hard = n === 4 || (tspin && n > 0);
    let label = tspin ? ['T-SPIN', 'T-SPIN SINGLE', 'T-SPIN DOUBLE', 'T-SPIN TRIPLE'][n] : ['', '', 'DOUBLE', 'TRIPLE', 'TETRIS!'][n];
    if (n > 0) {
      if (hard && G.b2b) { pts *= 1.5; label = 'B2B ' + label; }
      G.b2b = hard ? true : false;
      G.combo++;
    } else G.combo = -1;
    G.score += Math.round(pts * L) + (G.combo > 0 ? 50 * G.combo * L : 0);
    if (label) pop(label, n === 4 || tspin ? (tspin ? COLORS.T : COLORS.I) : '#fff', n === 4 || tspin ? 1.3 : 1);
    if (G.combo > 0) pop(`连击 ×${G.combo}`, '#ffd93a', .8, 34);
    if (n === 4) { G.flash = .5; G.shake = .4; }
    G.lines += n;
    const lv = Math.floor(G.lines / 10) + 1;
    if (lv > G.level) { G.level = lv; setTimeout(() => { pop(`LEVEL ${lv}`, '#7dff6a', 1.1, -40); FX.sound('big'); }, 350); }
    if (n) FX.sound(n === 4 || tspin ? 'mate' : 'capture');
  }
  function finishClear() {
    const rows = G.clearing.rows;
    G.board = G.board.filter((_, y) => !rows.includes(y));
    while (G.board.length < ROWS + HID) G.board.unshift(Array(COLS).fill(null));
    G.clearing = null;
    if (G.board.every(r => r.every(c => !c))) { G.score += 2000 * G.level; pop('PERFECT CLEAR', '#ff4dd2', 1.2, 40); FX.sound('mate'); }
    G.canHold = true;
    spawn(G.queue.shift());
  }
  function burstRows(rows) {
    for (const y of rows) for (let x = 0; x < COLS; x++) {
      const c = COLORS[G.board[y][x]] || '#fff';
      for (let k = 0; k < 2; k++) G.parts.push({ x: BX + (x + .5) * CELL, y: BY + (y - HID + .5) * CELL, vx: (Math.random() - .5) * 420, vy: -120 - Math.random() * 260,
                                                 life: .7, max: .7, c, s: 3 + Math.random() * 3, g: 700 });
    }
  }
  function pop(text, color, scale = 1, dy = 0) { G.pops.push({ text, color, scale, t: 0, dy }); }

  function gameOver() {
    state = 'over';
    G.cur = null;
    FX.sound('hit');
    const isNew = G.score > best;
    if (isNew) { best = G.score; try { localStorage.setItem('tetris-best', best); } catch {} }
    $('overScore').textContent = G.score.toLocaleString();
    $('overLines').textContent = G.lines;
    $('overLevel').textContent = G.level;
    $('overBest').textContent = best.toLocaleString();
    $('overNew').hidden = !isNew;
    setTimeout(() => { $('overOv').hidden = false; }, 500);
  }

  // ---------- 输入 ----------
  const held = { left: false, right: false, soft: false };
  let dasDir = 0, dasT = 0, arrT = 0;
  function press(k) {
    if (state !== 'play') return;
    if (k === 'left' || k === 'right') { const d = k === 'left' ? -1 : 1; held[k] = true; dasDir = d; dasT = 0; arrT = 0; move(d); }
    else if (k === 'soft') held.soft = true;
    else if (k === 'rotate') rotate(1);
    else if (k === 'ccw') rotate(-1);
    else if (k === 'hard') hardDrop();
    else if (k === 'hold') holdPiece();
  }
  function release(k) {
    if (k in held) held[k] = false;
    if ((k === 'left' && dasDir === -1) || (k === 'right' && dasDir === 1)) dasDir = held.left ? -1 : held.right ? 1 : 0;
  }
  const KEYMAP = { ArrowLeft: 'left', ArrowRight: 'right', ArrowDown: 'soft', ArrowUp: 'rotate', x: 'rotate', X: 'rotate', z: 'ccw', Z: 'ccw',
                   Control: 'ccw', ' ': 'hard', c: 'hold', C: 'hold', Shift: 'hold' };
  addEventListener('keydown', e => {
    const k = KEYMAP[e.key];
    if (k || e.key === ' ') e.preventDefault();
    if ((e.key === 'Enter' || e.key === ' ') && (state === 'menu' || state === 'over')) { start(); return; }
    if (e.key === 'p' || e.key === 'P' || e.key === 'Escape') { togglePause(); return; }
    if (k && !e.repeat) press(k);
  });
  addEventListener('keyup', e => { const k = KEYMAP[e.key]; if (k) release(k); });
  addEventListener('blur', () => { held.left = held.right = held.soft = false; dasDir = 0; if (state === 'play') togglePause(); });
  // 触屏：点按旋转，左右拖动移动，下滑软降，快速下滑硬降
  {
    let sx = 0, sy = 0, lx = 0, ly = 0, t0 = 0, moved = false, active = false;
    cv.addEventListener('pointerdown', e => { active = true; sx = lx = e.clientX; sy = ly = e.clientY; t0 = performance.now(); moved = false; });
    cv.addEventListener('pointermove', e => {
      if (!active || state !== 'play') return;
      const step = cv.getBoundingClientRect().width / W * CELL;
      while (e.clientX - lx > step) { move(1); lx += step; moved = true; }
      while (lx - e.clientX > step) { move(-1); lx -= step; moved = true; }
      if (e.clientY - ly > step) { if (G.cur && !collide(G.cur, 0, 1)) { G.cur.y++; G.score++; } ly += step; moved = true; }
    });
    cv.addEventListener('pointerup', e => {
      if (!active) return;
      active = false;
      const dt = performance.now() - t0, dy = e.clientY - sy;
      if (state !== 'play') return;
      if (dy > 60 && dt < 250) hardDrop();
      else if (!moved && dt < 300) rotate(1);
    });
    for (const b of document.querySelectorAll('.pad button')) {
      const k = b.dataset.k;
      b.addEventListener('pointerdown', e => { e.preventDefault(); press(k); });
      b.addEventListener('pointerup', () => release(k));
      b.addEventListener('pointerleave', () => release(k));
    }
  }

  // ---------- 主循环 ----------
  function update(dt) {
    if (state !== 'play') return;
    if (G.clearing) {
      G.clearing.t += dt;
      if (G.clearing.t >= CLEAR_T) finishClear();
      return;
    }
    if (!G.cur) return;
    // 自动重复移动
    if (dasDir) {
      dasT += dt;
      if (dasT > DAS) { arrT += dt; while (arrT >= ARR) { arrT -= ARR; if (!move(dasDir)) break; } }
    }
    const grav = Math.pow(Math.max(.05, .8 - (G.level - 1) * .007), G.level - 1);
    const interval = held.soft ? Math.min(grav, .03) : grav;
    G.fallT += dt;
    while (G.fallT >= interval) {
      G.fallT -= interval;
      if (!collide(G.cur, 0, 1)) {
        G.cur.y++; G.lastRotate = false;
        if (held.soft) G.score++;
        if (G.cur.y > G.lowest) { G.lowest = G.cur.y; G.resets = 0; }
      } else { G.fallT = 0; break; }
    }
    if (grounded()) { G.lockT += dt; if (G.lockT >= LOCK_DELAY) lock(); }
    else G.lockT = 0;
  }

  // ---------- 绘制 ----------
  function rr(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }
  function block(x, y, type, a = 1, size = CELL) {
    ctx.globalAlpha = a;
    const g = ctx.createLinearGradient(x, y, x + size, y + size);
    g.addColorStop(0, LIGHT[type]); g.addColorStop(1, COLORS[type]);
    ctx.fillStyle = g; rr(x + 1, y + 1, size - 2, size - 2, size * .16); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,.32)'; rr(x + size * .16, y + size * .12, size * .68, size * .18, size * .08); ctx.fill();
    ctx.fillStyle = 'rgba(0,0,0,.18)'; rr(x + size * .16, y + size * .78, size * .68, size * .1, size * .05); ctx.fill();
    ctx.globalAlpha = 1;
  }
  function miniPiece(type, cx, cy, size) {
    const cs = SHAPES[type][0];
    const xs = cs.map(c => c[0]), ys = cs.map(c => c[1]);
    const w = (Math.max(...xs) - Math.min(...xs) + 1) * size, h = (Math.max(...ys) - Math.min(...ys) + 1) * size;
    for (const [x, y] of cs) block(cx - w / 2 + (x - Math.min(...xs)) * size, cy - h / 2 + (y - Math.min(...ys)) * size, type, 1, size);
  }
  function panel(x, y, w, h, title) {
    ctx.fillStyle = 'rgba(255,255,255,.04)'; rr(x, y, w, h, 14); ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,.08)'; ctx.lineWidth = 1; ctx.stroke();
    if (title) { ctx.fillStyle = '#8f99c4'; ctx.font = '800 13px -apple-system, sans-serif'; ctx.textAlign = 'left'; ctx.fillText(title, x + 14, y + 22); }
  }
  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, '#11143a'); bg.addColorStop(1, '#090a1c');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    if (!G) return;
    ctx.save();
    if (G.shake > 0) ctx.translate((Math.random() - .5) * G.shake * 24, (Math.random() - .5) * G.shake * 24);

    // 左侧：暂存 + 数据
    panel(14, BY, 150, 120, 'HOLD');
    if (G.hold) { ctx.globalAlpha = G.canHold ? 1 : .35; miniPiece(G.hold, 89, BY + 72, 22); ctx.globalAlpha = 1; }
    panel(14, BY + 136, 150, 300);
    const stat = (label, val, y, col = '#fff') => {
      ctx.textAlign = 'left'; ctx.fillStyle = '#8f99c4'; ctx.font = '800 12px -apple-system, sans-serif'; ctx.fillText(label, 30, y);
      ctx.fillStyle = col; ctx.font = '900 24px -apple-system, sans-serif'; ctx.fillText(val, 30, y + 28);
    };
    stat('SCORE', G.score.toLocaleString(), BY + 166, '#33e6ff');
    stat('LEVEL', G.level, BY + 230);
    stat('LINES', G.lines, BY + 294);
    stat('BEST', Math.max(best, G.score).toLocaleString(), BY + 358, '#ffd93a');
    if (G.b2b) { ctx.fillStyle = '#b45cff'; ctx.font = '900 12px -apple-system, sans-serif'; ctx.fillText('BACK-TO-BACK', 30, BY + 412); }

    // 右侧：预览
    panel(BX + COLS * CELL + 16, BY, 150, 420, 'NEXT');
    G.queue.slice(0, 5).forEach((q, i) => miniPiece(q, BX + COLS * CELL + 91, BY + 72 + i * 72, i ? 18 : 24));

    // 棋盘
    const bw = COLS * CELL, bh = ROWS * CELL;
    ctx.fillStyle = 'rgba(0,0,0,.35)'; rr(BX - 6, BY - 6, bw + 12, bh + 12, 12); ctx.fill();
    ctx.strokeStyle = 'rgba(51,230,255,.35)'; ctx.lineWidth = 2; ctx.shadowColor = '#33e6ff'; ctx.shadowBlur = 16; ctx.stroke(); ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,.04)'; ctx.lineWidth = 1;
    for (let x = 1; x < COLS; x++) { ctx.beginPath(); ctx.moveTo(BX + x * CELL + .5, BY); ctx.lineTo(BX + x * CELL + .5, BY + bh); ctx.stroke(); }
    for (let y = 1; y < ROWS; y++) { ctx.beginPath(); ctx.moveTo(BX, BY + y * CELL + .5); ctx.lineTo(BX + bw, BY + y * CELL + .5); ctx.stroke(); }
    for (let y = HID; y < ROWS + HID; y++) for (let x = 0; x < COLS; x++) {
      const c = G.board[y][x];
      if (c) block(BX + x * CELL, BY + (y - HID) * CELL, c);
    }
    // 消行闪光
    if (G.clearing) {
      const k = G.clearing.t / CLEAR_T;
      for (const y of G.clearing.rows) {
        ctx.fillStyle = `rgba(255,255,255,${(1 - k) * .9})`;
        ctx.fillRect(BX + bw / 2 * k, BY + (y - HID) * CELL, bw * (1 - k), CELL);
      }
    }
    // 落点阴影与当前方块
    if (G.cur && state !== 'over') {
      let gy = 0;
      while (!collide(G.cur, 0, gy + 1)) gy++;
      for (const [cx, cy] of cells(G.cur)) {
        const y = G.cur.y + cy + gy - HID;
        if (y < 0) continue;
        ctx.strokeStyle = COLORS[G.cur.type]; ctx.globalAlpha = .55; ctx.lineWidth = 2;
        rr(BX + (G.cur.x + cx) * CELL + 3, BY + y * CELL + 3, CELL - 6, CELL - 6, 5); ctx.stroke();
        ctx.globalAlpha = .1; ctx.fillStyle = COLORS[G.cur.type]; ctx.fill(); ctx.globalAlpha = 1;
      }
      ctx.shadowColor = COLORS[G.cur.type]; ctx.shadowBlur = 14;
      const lockFade = grounded() ? .75 + .25 * Math.cos(G.lockT / LOCK_DELAY * Math.PI * 4) : 1;
      for (const [cx, cy] of cells(G.cur)) {
        const y = G.cur.y + cy - HID;
        if (y >= 0) block(BX + (G.cur.x + cx) * CELL, BY + y * CELL, G.cur.type, lockFade);
      }
      ctx.shadowBlur = 0;
    }
    // 粒子
    for (const p of G.parts) {
      ctx.globalAlpha = Math.max(0, p.life / p.max);
      ctx.fillStyle = p.c; ctx.fillRect(p.x - p.s / 2, p.y - p.s / 2, p.s, p.s);
    }
    ctx.globalAlpha = 1;
    // 文字
    for (const p of G.pops) {
      const k = p.t / 1.3;
      ctx.globalAlpha = k < .1 ? k / .1 : k > .7 ? (1 - k) / .3 : 1;
      const s = p.scale * (k < .12 ? 1.6 - k * 5 : 1);
      ctx.font = `900 ${Math.round(34 * s)}px "Arial Rounded MT Bold", -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      const x = BX + bw / 2, y = BY + bh * .38 + p.dy - k * 40;
      ctx.lineWidth = 6; ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.strokeText(p.text, x, y);
      ctx.fillStyle = p.color; ctx.shadowColor = p.color; ctx.shadowBlur = 18; ctx.fillText(p.text, x, y); ctx.shadowBlur = 0;
    }
    ctx.globalAlpha = 1;
    ctx.restore();
    if (G.flash > 0) { ctx.fillStyle = `rgba(120,240,255,${G.flash * .5})`; ctx.fillRect(0, 0, W, H); }
  }

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(.05, (now - last) / 1000);
    last = now;
    update(dt);
    if (G) {
      for (const p of G.parts) { p.life -= dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vy += (p.g || 0) * dt; }
      G.parts = G.parts.filter(p => p.life > 0);
      for (const p of G.pops) p.t += dt;
      G.pops = G.pops.filter(p => p.t < 1.3);
      if (G.shake > 0) G.shake = Math.max(0, G.shake - dt);
      if (G.flash > 0) G.flash = Math.max(0, G.flash - dt * 1.5);
    }
    draw(now / 1000);
  }

  function resize() {
    const dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W * dpr; cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function start() {
    newGame();
    state = 'play';
    held.left = held.right = held.soft = false; dasDir = 0;
    $('menuOv').hidden = $('overOv').hidden = $('pauseOv').hidden = true;
    FX.sound('big');
  }
  function togglePause() {
    if (state === 'play') { state = 'pause'; $('pauseOv').hidden = false; }
    else if (state === 'pause') { state = 'play'; $('pauseOv').hidden = true; last = performance.now(); }
  }
  $('btnStart').onclick = start;
  $('btnAgain').onclick = start;
  $('btnRestart2').onclick = start;
  $('btnResume').onclick = togglePause;
  $('menuBest').textContent = best ? `最高分 ${best.toLocaleString()}` : '消行越多、越连续，分数越高';
  FX.muteButton($('btnMute'));
  resize();
  newGame(); G.cur = null; // 菜单背景显示空棋盘
  requestAnimationFrame(t => { last = t; frame(t); });

  window.__tetris = { get G() { return G; }, get state() { return state; }, start, press, release, update, SHAPES };
})();
