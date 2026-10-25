// 贪吃蛇：经典模式 + 大作战（三条电脑蛇，BFS 找食物、泛洪填充判断安全）。渲染时在两格之间插值，移动更顺滑。
(function () {
  const $ = id => document.getElementById(id);
  const cv = $('cv'), ctx = cv.getContext('2d');
  const GW = 40, GH = 25, CELL = 24, W = GW * CELL, H = GH * CELL;
  const DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
  const AI_STYLE = [
    { name: '红莓', color: '#ff4d6d', speed: .145 },
    { name: '蓝鲸', color: '#4d9bff', speed: .155 },
    { name: '紫薯', color: '#b45cff', speed: .165 },
  ];
  const PLAYER_COLOR = '#4ee26b';

  let mode = 'battle', state = 'menu', S = null;
  const bestKey = () => 'snake-best-' + mode;
  const getBest = () => { try { return +localStorage.getItem(bestKey()) || 0; } catch { return 0; } };

  // ---------- 工具 ----------
  const key = (x, y) => y * GW + x;
  const inside = (x, y) => x >= 0 && x < GW && y >= 0 && y < GH;
  const rnd = n => Math.random() * n | 0;
  function occupied() {
    const set = new Set();
    for (const s of S.snakes) if (s.alive) for (const [x, y] of s.body) set.add(key(x, y));
    return set;
  }
  function emptyCell(margin = 1) {
    const occ = occupied(), foods = new Set(S.foods.map(f => key(f.x, f.y)));
    for (let i = 0; i < 400; i++) {
      const x = margin + rnd(GW - margin * 2), y = margin + rnd(GH - margin * 2);
      if (!occ.has(key(x, y)) && !foods.has(key(x, y))) return [x, y];
    }
    return null;
  }

  // ---------- 开局 ----------
  function makeSnake(x, y, dir, len, style, ai) {
    const [dx, dy] = DIRS[dir];
    const body = [];
    for (let i = 0; i < len; i++) body.push([x - dx * i, y - dy * i]);
    return { body, prev: body.map(c => c.slice()), dir, queue: [], grow: 0, alive: true, ai, color: style.color, name: style.name,
             interval: style.speed, acc: 0, deadT: 0, kills: 0, boostN: 0 };
  }
  function newGame() {
    S = { snakes: [], foods: [], parts: [], pops: [], score: 0, kills: 0, t: 0, shake: 0, reason: '' };
    S.player = makeSnake(8, 12, 'right', 4, { name: '你', color: PLAYER_COLOR, speed: .11 }, false);
    S.snakes.push(S.player);
    if (mode === 'battle') {
      const spots = [[31, 5, 'left'], [31, 19, 'left'], [20, 3, 'down']];
      AI_STYLE.forEach((st, i) => S.snakes.push(makeSnake(spots[i][0], spots[i][1], spots[i][2], 5, st, true)));
    }
    fillFood();
  }
  function fillFood() {
    const want = mode === 'battle' ? 14 : 1;
    while (S.foods.filter(f => !f.bean).length < want) {
      const c = emptyCell();
      if (!c) break;
      const gold = Math.random() < (mode === 'battle' ? .08 : .12);
      S.foods.push({ x: c[0], y: c[1], gold, t: 0, life: gold ? 9 : Infinity, color: gold ? '#ffd23a' : ['#ff7a59', '#ff4dd2', '#7af7ff', '#ffe066'][rnd(4)] });
    }
  }

  // ---------- 电脑蛇 ----------
  function aiChoose(s) {
    const occ = occupied();
    const [hx, hy] = s.body[0];
    // 尾巴下一步会挪开，不算障碍
    if (!s.grow) occ.delete(key(...s.body[s.body.length - 1]));
    const heads = S.snakes.filter(o => o !== s && o.alive).map(o => o.body[0]);
    const danger = new Set();
    for (const [x, y] of heads) for (const [dx, dy] of Object.values(DIRS)) danger.add(key(x + dx, y + dy));
    const opts = Object.entries(DIRS).filter(([d]) => !isOpposite(d, s.dir)).map(([d, [dx, dy]]) => ({ d, x: hx + dx, y: hy + dy }))
      .filter(o => inside(o.x, o.y) && !occ.has(key(o.x, o.y)));
    if (!opts.length) return s.dir;
    // BFS 到最近的食物
    const foods = new Map(S.foods.map(f => [key(f.x, f.y), f]));
    const firstStep = new Map(), q = [];
    for (const o of opts) { const k = key(o.x, o.y); firstStep.set(k, o.d); q.push([o.x, o.y]); }
    let target = null;
    for (let i = 0; i < q.length && i < 260; i++) { // 只看得见附近的食物，给玩家留点机会
      const [x, y] = q[i], k = key(x, y);
      if (foods.has(k)) { target = firstStep.get(k); break; }
      for (const [dx, dy] of Object.values(DIRS)) {
        const nx = x + dx, ny = y + dy, nk = key(nx, ny);
        if (!inside(nx, ny) || occ.has(nk) || firstStep.has(nk)) continue;
        firstStep.set(nk, firstStep.get(k)); q.push([nx, ny]);
      }
    }
    // 每个方向的可活动空间（泛洪），太小就是死路
    const space = o => {
      const seen = new Set([key(o.x, o.y)]), st = [[o.x, o.y]];
      while (st.length && seen.size < 200) {
        const [x, y] = st.pop();
        for (const [dx, dy] of Object.values(DIRS)) {
          const nx = x + dx, ny = y + dy, nk = key(nx, ny);
          if (inside(nx, ny) && !occ.has(nk) && !seen.has(nk)) { seen.add(nk); st.push([nx, ny]); }
        }
      }
      return seen.size;
    };
    const need = Math.min(200, s.body.length + 4);
    let bestO = null, bestV = -1e9;
    for (const o of opts) {
      const sp = space(o);
      let v = Math.min(sp, need) * 10;
      if (sp < need) v -= 500;
      if (o.d === target && s.body.length < 45) v += 60; // 长到 45 节就不再追食物，免得把场地挤满
      if (danger.has(key(o.x, o.y))) v -= 300;
      if (o.d === s.dir) v += 3;
      v += Math.random() * 4;
      if (v > bestV) { bestV = v; bestO = o; }
    }
    return bestO.d;
  }
  const isOpposite = (a, b) => DIRS[a][0] === -DIRS[b][0] && DIRS[a][1] === -DIRS[b][1];

  // ---------- 移动与碰撞 ----------
  function step(s) {
    if (s.ai) s.dir = aiChoose(s);
    else if (s.queue.length) s.dir = s.queue.shift();
    const [dx, dy] = DIRS[s.dir];
    const [hx, hy] = s.body[0];
    const nx = hx + dx, ny = hy + dy;
    s.prev = s.body.map(c => c.slice());
    // 撞墙
    if (!inside(nx, ny)) return kill(s, '撞墙了', null);
    // 撞身体（包括自己）；对方尾巴这一步若会挪开则不算
    for (const o of S.snakes) {
      if (!o.alive) continue;
      const len = o.body.length - (o === s && !s.grow ? 1 : 0); // 自己的尾巴这一步会挪开
      for (let i = 0; i < len; i++) {
        const [bx, by] = o.body[i];
        if (bx === nx && by === ny) {
          if (o === s) return kill(s, '咬到了自己', null);
          if (i === 0) { kill(o, `和${s.name === '你' ? '你' : s.name}迎头相撞`, s); return kill(s, `和${o.name}迎头相撞`, o); }
          return kill(s, `撞到了${o.name}的身体`, o);
        }
      }
    }
    s.body.unshift([nx, ny]);
    if (s.grow > 0) s.grow--; else s.body.pop();
    // 吃东西
    const fi = S.foods.findIndex(f => f.x === nx && f.y === ny);
    if (fi >= 0) {
      const f = S.foods[fi];
      S.foods.splice(fi, 1);
      s.grow += f.gold ? 3 : 1;
      if (!s.ai) {
        const pts = f.gold ? 50 : f.bean ? 5 : 10;
        S.score += pts;
        if (f.gold) { pop('+50', '#ffd23a', nx, ny); FX.sound('power'); } else FX.sound('coin');
        burst(nx, ny, f.color, f.gold ? 16 : 6);
        if (mode === 'classic') s.interval = Math.max(.055, .11 - (s.body.length - 4) * .0018);
      }
      fillFood();
    }
  }
  function kill(s, reason, by) {
    if (!s.alive) return;
    s.alive = false; s.deadT = 0;
    for (const [x, y] of s.body) burst(x, y, s.color, 2);
    if (s === S.player) {
      S.reason = reason;
      S.shake = .5;
      FX.sound('hit');
      setTimeout(gameOver, 900);
      return;
    }
    // 电脑蛇死后，身体变成豆子
    s.body.forEach(([x, y], i) => { if (i % 2 === 0 && inside(x, y)) S.foods.push({ x, y, bean: true, t: 0, life: 14, color: s.color }); });
    if (by === S.player) {
      S.player.kills++; S.kills++; S.score += 100;
      pop('击杀 +100', '#ff4dd2', s.body[0][0], s.body[0][1]);
      FX.sound('mate');
      S.shake = .3;
    } else FX.sound('capture');
  }
  function respawn(s) {
    for (let tries = 0; tries < 50; tries++) {
      const c = emptyCell(4);
      if (!c) return;
      const dir = ['left', 'right', 'up', 'down'][rnd(4)];
      const [dx, dy] = DIRS[dir];
      const occ = occupied();
      let ok = true;
      for (let i = -3; i < 8; i++) if (occ.has(key(c[0] + dx * i, c[1] + dy * i))) ok = false;
      if (!ok) continue;
      const style = AI_STYLE.find(a => a.name === s.name);
      Object.assign(s, makeSnake(c[0], c[1], dir, 4, style, true));
      return;
    }
  }

  // ---------- 特效 ----------
  function burst(x, y, color, n) {
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2, v = 60 + Math.random() * 160;
      S.parts.push({ x: (x + .5) * CELL, y: (y + .5) * CELL, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: .6, c: color, s: 2 + Math.random() * 3 });
    }
  }
  function pop(text, color, x, y) { S.pops.push({ text, color, x: (x + .5) * CELL, y: (y + .5) * CELL, t: 0 }); }

  // ---------- 输入 ----------
  function turn(d) {
    if (state !== 'play' || !S.player.alive) return;
    const q = S.player.queue, lastDir = q.length ? q[q.length - 1] : S.player.dir;
    if (d === lastDir || isOpposite(d, lastDir) || q.length >= 3) return;
    q.push(d);
  }
  let boost = false;
  const KEYS = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right', w: 'up', s: 'down', a: 'left', d: 'right', W: 'up', S: 'down', A: 'left', D: 'right' };
  addEventListener('keydown', e => {
    if (KEYS[e.key] || e.key === ' ') e.preventDefault();
    if ((e.key === 'Enter' || e.key === ' ') && (state === 'menu' || state === 'over')) { start(); return; }
    if (e.key === 'p' || e.key === 'P' || e.key === 'Escape') { togglePause(); return; }
    if (e.key === ' ') boost = true;
    if (KEYS[e.key]) turn(KEYS[e.key]);
  });
  addEventListener('keyup', e => { if (e.key === ' ') boost = false; });
  addEventListener('blur', () => { boost = false; if (state === 'play') togglePause(); });
  {
    let sx = 0, sy = 0, active = false;
    cv.addEventListener('pointerdown', e => { active = true; sx = e.clientX; sy = e.clientY; });
    cv.addEventListener('pointermove', e => {
      if (!active) return;
      const dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
      turn(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
      sx = e.clientX; sy = e.clientY;
    });
    addEventListener('pointerup', () => { active = false; });
    for (const b of document.querySelectorAll('.pad button')) b.addEventListener('pointerdown', e => { e.preventDefault(); turn(b.dataset.d); });
  }

  // ---------- 主循环 ----------
  function update(dt) {
    if (state !== 'play') return;
    S.t += dt;
    for (const s of S.snakes) {
      if (!s.alive) {
        if (s.ai && mode === 'battle' && (s.deadT += dt) > 3) respawn(s);
        continue;
      }
      // 玩家按空格加速：大作战里每加速 8 步掉一节尾巴
      const fast = !s.ai && boost && s.body.length > 4;
      s.acc += dt * (fast ? 1.8 : 1);
      while (s.acc >= s.interval && s.alive) {
        s.acc -= s.interval;
        step(s);
        if (fast && mode === 'battle' && s.alive && ++s.boostN % 8 === 0 && s.body.length > 4) {
          const [tx, ty] = s.body.pop();
          S.foods.push({ x: tx, y: ty, bean: true, t: 0, life: 10, color: s.color });
        }
      }
    }
    for (const f of S.foods) f.t += dt;
    const before = S.foods.length;
    S.foods = S.foods.filter(f => f.t < f.life);
    if (S.foods.length !== before) fillFood();
  }

  // ---------- 绘制 ----------
  function drawSnake(s, t) {
    const k = s.alive ? Math.min(1, s.acc / s.interval) : 1;
    const pts = s.body.map((c, i) => {
      const p = s.prev[i] || s.prev[s.prev.length - 1] || c;
      return [(p[0] + (c[0] - p[0]) * k + .5) * CELL, (p[1] + (c[1] - p[1]) * k + .5) * CELL];
    });
    const alpha = s.alive ? 1 : Math.max(0, 1 - s.deadT * 2.5);
    if (alpha <= 0) return;
    ctx.globalAlpha = alpha;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.shadowColor = s.color; ctx.shadowBlur = s === S.player ? 18 : 10;
    ctx.strokeStyle = s.color; ctx.lineWidth = CELL * .78;
    ctx.beginPath();
    pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.stroke();
    ctx.shadowBlur = 0;
    // 背部花纹
    ctx.strokeStyle = 'rgba(255,255,255,.28)'; ctx.lineWidth = CELL * .22;
    ctx.setLineDash([CELL * .25, CELL * .55]);
    ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
    ctx.setLineDash([]);
    // 头和眼睛
    const [hx, hy] = pts[0], [dx, dy] = DIRS[s.dir];
    ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(hx, hy, CELL * .48, 0, 7); ctx.fill();
    for (const side of [-1, 1]) {
      const ex = hx + dx * CELL * .14 - dy * side * CELL * .2, ey = hy + dy * CELL * .14 + dx * side * CELL * .2;
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(ex, ey, CELL * .15, 0, 7); ctx.fill();
      ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(ex + dx * CELL * .05, ey + dy * CELL * .05, CELL * .08, 0, 7); ctx.fill();
    }
    if (s !== S.player && s.alive) {
      ctx.font = '800 11px -apple-system, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = 'rgba(255,255,255,.75)';
      ctx.fillText(s.name, hx, hy - CELL * .8);
    }
    ctx.globalAlpha = 1;
  }
  function draw(t) {
    ctx.save();
    if (S && S.shake > 0) ctx.translate((Math.random() - .5) * S.shake * 18, (Math.random() - .5) * S.shake * 18);
    const g = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W * .7);
    g.addColorStop(0, '#0f1f2a'); g.addColorStop(1, '#070b14');
    ctx.fillStyle = g; ctx.fillRect(-20, -20, W + 40, H + 40);
    ctx.fillStyle = 'rgba(255,255,255,.025)';
    for (let y = 0; y < GH; y++) for (let x = (y % 2); x < GW; x += 2) ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
    ctx.strokeStyle = 'rgba(78,226,107,.5)'; ctx.lineWidth = 3; ctx.shadowColor = '#4ee26b'; ctx.shadowBlur = 14;
    ctx.strokeRect(1.5, 1.5, W - 3, H - 3); ctx.shadowBlur = 0;
    if (!S) { ctx.restore(); return; }
    // 食物
    for (const f of S.foods) {
      const cx = (f.x + .5) * CELL, cy = (f.y + .5) * CELL;
      const pulse = 1 + Math.sin(t * 6 + f.x) * .12;
      const blink = f.life !== Infinity && f.life - f.t < 2 ? (Math.sin(t * 20) > 0 ? 1 : .3) : 1;
      ctx.globalAlpha = blink;
      ctx.shadowColor = f.color; ctx.shadowBlur = f.gold ? 22 : 12;
      ctx.fillStyle = f.color;
      ctx.beginPath(); ctx.arc(cx, cy, CELL * (f.gold ? .42 : f.bean ? .22 : .3) * pulse, 0, 7); ctx.fill();
      if (f.gold) { ctx.shadowBlur = 0; ctx.fillStyle = '#7a4a00'; ctx.font = '900 13px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', cx, cy + 1); ctx.textBaseline = 'alphabetic'; }
      ctx.shadowBlur = 0; ctx.globalAlpha = 1;
    }
    for (const s of S.snakes) if (s !== S.player) drawSnake(s, t);
    drawSnake(S.player, t);
    for (const p of S.parts) { ctx.globalAlpha = Math.max(0, p.life / .6); ctx.fillStyle = p.c; ctx.fillRect(p.x - p.s / 2, p.y - p.s / 2, p.s, p.s); }
    ctx.globalAlpha = 1;
    for (const p of S.pops) {
      ctx.globalAlpha = Math.max(0, 1 - p.t);
      ctx.font = '900 20px -apple-system, sans-serif'; ctx.textAlign = 'center';
      ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.strokeText(p.text, p.x, p.y - p.t * 40);
      ctx.fillStyle = p.color; ctx.fillText(p.text, p.x, p.y - p.t * 40);
    }
    ctx.globalAlpha = 1;
    ctx.restore();
    // 记分牌
    if (state !== 'menu') {
      ctx.textAlign = 'left'; ctx.font = '900 22px -apple-system, sans-serif'; ctx.fillStyle = '#fff';
      ctx.fillText(S.score.toLocaleString(), 16, 34);
      ctx.font = '700 12px -apple-system, sans-serif'; ctx.fillStyle = '#8fb8a0';
      ctx.fillText(`长度 ${S.player.body.length}${mode === 'battle' ? ` · 击杀 ${S.kills}` : ''}`, 16, 52);
      if (mode === 'battle') {
        const rows = S.snakes.map(s => ({ n: s.name, l: s.alive ? s.body.length : 0, c: s.color, me: s === S.player })).sort((a, b) => b.l - a.l);
        ctx.fillStyle = 'rgba(0,0,0,.35)'; ctx.beginPath(); ctx.roundRect(W - 150, 12, 136, 20 + rows.length * 20, 10); ctx.fill();
        rows.forEach((r, i) => {
          ctx.fillStyle = r.c; ctx.beginPath(); ctx.arc(W - 136, 32 + i * 20 - 4, 5, 0, 7); ctx.fill();
          ctx.fillStyle = r.me ? '#fff' : '#c8cff0'; ctx.font = (r.me ? '900' : '700') + ' 12px -apple-system, sans-serif';
          ctx.fillText(r.n, W - 124, 32 + i * 20);
          ctx.textAlign = 'right'; ctx.fillText(r.l ? r.l : '复活中', W - 24, 32 + i * 20); ctx.textAlign = 'left';
        });
      }
    }
  }

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(.05, (now - last) / 1000);
    last = now;
    update(dt);
    if (S) {
      for (const p of S.parts) { p.life -= dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= .96; p.vy *= .96; }
      S.parts = S.parts.filter(p => p.life > 0);
      for (const p of S.pops) p.t += dt;
      S.pops = S.pops.filter(p => p.t < 1);
      if (S.shake > 0) S.shake = Math.max(0, S.shake - dt);
      if (!S.player.alive) S.player.deadT += dt;
    }
    draw(now / 1000);
  }

  // ---------- 流程 ----------
  function start() {
    newGame();
    state = 'play'; boost = false;
    $('menuOv').hidden = $('overOv').hidden = $('pauseOv').hidden = true;
    FX.sound('big');
  }
  function gameOver() {
    if (state !== 'play') return;
    state = 'over';
    const best = getBest(), isNew = S.score > best;
    if (isNew) try { localStorage.setItem(bestKey(), S.score); } catch {}
    $('overReason').textContent = S.reason;
    $('overScore').textContent = S.score.toLocaleString();
    $('overLen').textContent = S.player.body.length;
    $('overKills').textContent = S.kills;
    $('overKills').parentElement.hidden = mode !== 'battle';
    $('overBest').textContent = Math.max(best, S.score).toLocaleString();
    $('overNew').hidden = !isNew;
    $('overOv').hidden = false;
    FX.sound(isNew ? 'mate' : 'big');
  }
  function toMenu() {
    state = 'menu';
    $('overOv').hidden = $('pauseOv').hidden = true; $('menuOv').hidden = false;
    paintMenu();
  }
  function paintMenu() {
    document.querySelectorAll('.mode').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
    const b = getBest();
    $('menuBest').textContent = b ? `${mode === 'battle' ? '大作战' : '经典'}最高分 ${b.toLocaleString()}` : '';
  }
  function togglePause() {
    if (state === 'play') { state = 'pause'; $('pauseOv').hidden = false; }
    else if (state === 'pause') { state = 'play'; $('pauseOv').hidden = true; last = performance.now(); }
  }
  $('modes').addEventListener('click', e => {
    const b = e.target.closest('[data-mode]');
    if (!b) return;
    mode = b.dataset.mode;
    try { localStorage.setItem('snake-mode', mode); } catch {}
    paintMenu(); newGame(); FX.sound('move');
  });
  try { mode = localStorage.getItem('snake-mode') || 'battle'; } catch {}
  $('btnStart').onclick = start;
  $('btnAgain').onclick = start;
  $('btnMenu').onclick = toMenu;
  $('btnMenu2').onclick = toMenu;
  $('btnResume').onclick = togglePause;
  FX.muteButton($('btnMute'));
  const dpr = Math.min(2, devicePixelRatio || 1);
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  newGame();
  paintMenu();
  requestAnimationFrame(t => { last = t; frame(t); });

  window.__snake = { get S() { return S; }, get state() { return state; }, start, turn, update, setMode: m => { mode = m; paintMenu(); } };
})();
