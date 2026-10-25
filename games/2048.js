// 2048：方块用 DOM + CSS 过渡做滑动，合并时先滑到目标格再换成新方块并弹一下；支持撤销一步。
(function () {
  const $ = id => document.getElementById(id);
  const board = $('board');
  const N = 4, SLIDE_MS = 110;
  const STYLE = {
    2: ['#2b3058', '#c8cff0'], 4: ['#353c74', '#dfe4ff'], 8: ['#ff9a2e', '#2a1400'], 16: ['#ff7a45', '#2a0d00'],
    32: ['#ff5a5f', '#fff'], 64: ['#ff3b6b', '#fff'], 128: ['#ffd23a', '#3a2a00', 18], 256: ['#ffc61a', '#3a2a00', 22],
    512: ['#ffb400', '#3a2400', 26], 1024: ['#7dff6a', '#073a12', 30], 2048: ['#33e6ff', '#032a3a', 40],
  };
  const SUPER = ['#b45cff', '#fff', 40];

  let grid, score, best = 0, won, prev, nextId = 1, over;
  try { best = +localStorage.getItem('2048-best') || 0; } catch {}

  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const c = document.createElement('div');
    c.className = 'cell';
    c.style.cssText = `left:calc(var(--gap) + ${x} * (var(--size) + var(--gap)));top:calc(var(--gap) + ${y} * (var(--size) + var(--gap)))`;
    board.prepend(c);
  }

  function makeTile(v, x, y, cls) {
    const el = document.createElement('div');
    const t = { id: nextId++, v, x, y, el };
    el.className = 'tile' + (cls ? ' ' + cls : '');
    paint(t);
    return t;
  }
  function paint(t) {
    const [bg, fg, glow] = STYLE[t.v] || SUPER;
    const el = t.el;
    el.textContent = t.v;
    el.dataset.d = String(t.v).length;
    el.style.setProperty('--x', t.x); el.style.setProperty('--y', t.y);
    el.style.background = `linear-gradient(160deg, ${bg}, color-mix(in srgb, ${bg} 75%, #000))`;
    el.style.color = fg;
    el.style.boxShadow = glow ? `0 0 ${glow}px color-mix(in srgb, ${bg} 70%, transparent), inset 0 2px 0 rgba(255,255,255,.35)` : 'inset 0 2px 0 rgba(255,255,255,.08)';
  }
  const place = t => { t.el.style.setProperty('--x', t.x); t.el.style.setProperty('--y', t.y); };

  function addRandom() {
    const empty = [];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (!grid[y][x]) empty.push([x, y]);
    if (!empty.length) return;
    const [x, y] = empty[Math.random() * empty.length | 0];
    const t = makeTile(Math.random() < .9 ? 2 : 4, x, y, 'new');
    grid[y][x] = t;
    board.appendChild(t.el);
  }
  function snapshot() { return { vals: grid.map(r => r.map(t => t ? t.v : 0)), score, won }; }
  function load(snap) {
    board.querySelectorAll('.tile').forEach(e => e.remove());
    grid = snap.vals.map((r, y) => r.map((v, x) => { if (!v) return null; const t = makeTile(v, x, y); board.appendChild(t.el); return t; }));
    score = snap.score; won = snap.won;
  }
  function newGame() {
    board.querySelectorAll('.tile').forEach(e => e.remove());
    grid = Array.from({ length: N }, () => Array(N).fill(null));
    score = 0; won = false; prev = null; over = false;
    addRandom(); addRandom();
    $('winOv').hidden = $('overOv').hidden = true;
    hud();
  }

  const VEC = { left: [-1, 0], right: [1, 0], up: [0, -1], down: [0, 1] };
  function move(dir) {
    if (over || !$('winOv').hidden) return;
    const [vx, vy] = VEC[dir];
    const xs = [0, 1, 2, 3], ys = [0, 1, 2, 3];
    if (vx === 1) xs.reverse();
    if (vy === 1) ys.reverse();
    const snap = snapshot(), merged = new Set(), removals = [], born = [];
    let moved = false, gained = 0, biggest = 0;
    for (const y of ys) for (const x of xs) {
      const t = grid[y][x];
      if (!t) continue;
      let cx = x, cy = y, target = null;
      for (;;) {
        const tx = cx + vx, ty = cy + vy;
        if (tx < 0 || tx >= N || ty < 0 || ty >= N) break;
        const o = grid[ty][tx];
        if (!o) { cx = tx; cy = ty; continue; }
        if (o.v === t.v && !merged.has(o)) target = o;
        break;
      }
      grid[y][x] = null;
      if (target) {
        t.x = target.x; t.y = target.y; place(t);
        const nt = makeTile(t.v * 2, target.x, target.y, 'merged');
        grid[target.y][target.x] = nt;
        merged.add(nt); removals.push(t, target); born.push(nt);
        gained += nt.v; biggest = Math.max(biggest, nt.v);
        moved = true;
      } else {
        grid[cy][cx] = t;
        if (cx !== x || cy !== y) { t.x = cx; t.y = cy; place(t); moved = true; }
      }
    }
    if (!moved) return;
    prev = snap;
    score += gained;
    setTimeout(() => { removals.forEach(t => t.el.remove()); born.forEach(t => board.appendChild(t.el)); }, SLIDE_MS);
    addRandom();
    if (gained) delta(gained);
    FX.sound(biggest >= 512 ? 'power' : biggest >= 64 ? 'capture' : gained ? 'coin' : 'move');
    hud();
    if (!won && born.some(t => t.v === 2048)) { won = true; setTimeout(() => { $('winOv').hidden = false; FX.sound('mate'); }, 300); }
    else if (!canMove()) setTimeout(gameOver, 300);
  }
  function canMove() {
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const t = grid[y][x];
      if (!t) return true;
      if (x < N - 1 && grid[y][x + 1] && grid[y][x + 1].v === t.v) return true;
      if (y < N - 1 && grid[y + 1][x] && grid[y + 1][x].v === t.v) return true;
    }
    return false;
  }
  function gameOver() {
    over = true;
    const max = Math.max(...grid.flat().map(t => t ? t.v : 0));
    const isNew = score >= best && score > 0;
    $('overScore').textContent = score.toLocaleString();
    $('overNew').hidden = !isNew;
    $('overMax').textContent = `最大方块 ${max}`;
    $('overOv').hidden = false;
    FX.sound('hit');
  }
  function undo() {
    if (!prev) return;
    load(prev); prev = null; over = false;
    $('overOv').hidden = true; $('winOv').hidden = true;
    hud(); FX.sound('move');
  }
  function hud() {
    if (score > best) { best = score; try { localStorage.setItem('2048-best', best); } catch {} }
    $('score').textContent = score.toLocaleString();
    $('best').textContent = best.toLocaleString();
    $('btnUndo').disabled = !prev;
    $('btnUndo2').disabled = !prev;
  }
  function delta(n) {
    const el = document.createElement('div');
    el.className = 'g-delta'; el.textContent = '+' + n;
    $('score').parentElement.appendChild(el);
    setTimeout(() => el.remove(), 800);
  }

  const KEYS = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down', a: 'left', d: 'right', w: 'up', s: 'down', A: 'left', D: 'right', W: 'up', S: 'down' };
  addEventListener('keydown', e => {
    if (KEYS[e.key]) { e.preventDefault(); move(KEYS[e.key]); }
    else if ((e.key === 'z' || e.key === 'Z') && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undo(); }
  });
  {
    let sx = 0, sy = 0, active = false;
    board.addEventListener('pointerdown', e => { if (e.target.closest('.ov')) return; active = true; sx = e.clientX; sy = e.clientY; });
    addEventListener('pointerup', e => {
      if (!active) return;
      active = false;
      const dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 30) return;
      move(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
    });
  }
  $('btnNew').onclick = newGame; $('btnNew2').onclick = newGame; $('btnAgain').onclick = newGame;
  $('btnUndo').onclick = undo; $('btnUndo2').onclick = undo;
  $('btnKeep').onclick = () => { $('winOv').hidden = true; };
  FX.muteButton($('btnMute'));
  newGame();

  window.__2048 = { move, undo, newGame, get grid() { return grid.map(r => r.map(t => t ? t.v : 0)); }, get score() { return score; }, canMove,
                    set(vals) { load({ vals, score, won }); prev = null; hud(); } };
})();
