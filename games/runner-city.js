// 世界级都市天际线：地标建筑 + 普通楼宇，预先画到离屏画布，跑动时按视差平移、首尾相接循环。
(function (root) {
  const SKY_H = 340;   // 天际线画布高度（逻辑像素），底边是建筑地基
  const SCALE = 2;     // 离屏画布放大倍数，保证高清屏清晰
  const P = {
    top: '#22104a', bottom: '#0a0420', side: '#07021a', edge: 'rgba(255,43,214,.75)', accent: '#33f0ff',
    win: ['#ffe6a0', '#33f0ff', '#ff9ad5', '#ffffff', '#ffd36b'],
  };

  function mulberry(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function seedOf(s) { let h = 7; for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) | 0; return h; }

  function bodyFill(g, top, base) {
    const gr = g.createLinearGradient(0, top, 0, base);
    gr.addColorStop(0, P.top); gr.addColorStop(1, P.bottom);
    return gr;
  }
  // 用路径画楼体：渐变填充 + 霓虹描边
  function shape(g, pts, top, base, edge = P.edge) {
    g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) g.lineTo(p[0], p[1]);
    g.closePath();
    g.fillStyle = bodyFill(g, top, base); g.fill();
    g.save(); g.shadowColor = edge; g.shadowBlur = 6; g.strokeStyle = edge; g.lineWidth = 1; g.stroke(); g.restore();
  }
  // 在当前裁剪区域内铺窗户
  function windows(g, x, y, w, h, rnd, { cw = 3, ch = 4, gx = 2, gy = 3, density = .5, colors = P.win } = {}) {
    for (let wy = y; wy < y + h - ch; wy += ch + gy) {
      const rowOn = rnd() > .15;
      for (let wx = x; wx < x + w - cw; wx += cw + gx) {
        if (!rowOn || rnd() > density) continue;
        g.globalAlpha = .45 + rnd() * .55;
        g.fillStyle = colors[Math.floor(rnd() * colors.length)];
        g.fillRect(wx, wy, cw, ch);
      }
    }
    g.globalAlpha = 1;
  }
  function clipPath(g, pts) { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (const p of pts.slice(1)) g.lineTo(p[0], p[1]); g.closePath(); g.clip(); }
  function glowLine(g, color, w, blur, fn) { g.save(); g.strokeStyle = color; g.lineWidth = w; g.shadowColor = color; g.shadowBlur = blur; g.beginPath(); fn(); g.stroke(); g.restore(); }
  function glowFill(g, color, blur, fn) { g.save(); g.fillStyle = color; g.shadowColor = color; g.shadowBlur = blur; g.beginPath(); fn(); g.fill(); g.restore(); }

  // ---------- 普通楼宇（带侧面，做出体积感） ----------
  function generic(g, x, b, w, h, rnd) {
    const side = Math.max(4, w * .22), top = b - h;
    const kind = Math.floor(rnd() * 5);
    // 侧面
    g.fillStyle = P.side;
    g.beginPath(); g.moveTo(x + w, top + 3); g.lineTo(x + w + side, top + 6); g.lineTo(x + w + side, b); g.lineTo(x + w, b); g.closePath(); g.fill();
    let pts;
    if (kind === 1) { const s = w * .18; pts = [[x, b], [x, top + 26], [x + s, top + 26], [x + s, top], [x + w - s, top], [x + w - s, top + 26], [x + w, top + 26], [x + w, b]]; }
    else if (kind === 2) pts = [[x, b], [x, top + 14], [x + w / 2, top], [x + w, top + 14], [x + w, b]];
    else if (kind === 3) pts = [[x, b], [x, top + 18], [x + w, top], [x + w, b]];
    else pts = [[x, b], [x, top], [x + w, top], [x + w, b]];
    shape(g, pts, top, b);
    g.save(); clipPath(g, pts); windows(g, x + 3, top + 6, w - 5, h - 8, rnd, { density: .35 + rnd() * .35 }); g.restore();
    const lights = [];
    if (kind === 4 || (kind === 0 && rnd() > .5)) { // 天线
      g.strokeStyle = '#3a2a6a'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x + w / 2, top); g.lineTo(x + w / 2, top - 18); g.stroke();
      lights.push([x + w / 2, top - 19]);
    }
    if (kind === 0 && rnd() > .5) { g.fillStyle = P.side; g.fillRect(x + w * .2, top - 6, w * .3, 6); }
    // 楼顶霓虹招牌
    if (rnd() > .7 && w > 26) glowFill(g, P.win[Math.floor(rnd() * 3) + 1], 8, () => g.rect(x + 3, top + 6, w - 6, 3));
    return lights;
  }

  // ---------- 地标 ----------
  const L = {};
  L.ICC = { w: 50, draw(g, x, b, rnd) { // 香港 环球贸易广场
    const h = 300, pts = [[x, b], [x, b - h + 22], [x + 9, b - h], [x + 41, b - h], [x + 50, b - h + 22], [x + 50, b]];
    shape(g, pts, b - h, b);
    g.save(); clipPath(g, pts);
    for (let i = 0; i < 10; i++) { // 外墙 LED 灯光秀
      const gx = x + 3 + i * 4.6, gr = g.createLinearGradient(0, b - h, 0, b);
      gr.addColorStop(0, `hsla(${190 + i * 14},100%,65%,.75)`); gr.addColorStop(1, 'hsla(300,100%,60%,.08)');
      g.fillStyle = gr; g.fillRect(gx, b - h + 6, 1.4, h - 10);
    }
    windows(g, x + 2, b - h + 8, 46, h - 10, rnd, { cw: 2, ch: 2, gx: 2, gy: 3, density: .35 });
    g.restore();
    return [[x + 25, b - h - 2]];
  } };
  L.IFC = { w: 44, draw(g, x, b, rnd) { // 香港 国际金融中心二期
    const pts = [[x, b], [x, b - 190], [x + 4, b - 192], [x + 4, b - 222], [x + 8, b - 224], [x + 8, b - 236], [x + 36, b - 236], [x + 36, b - 224], [x + 40, b - 222], [x + 40, b - 192], [x + 44, b - 190], [x + 44, b]];
    shape(g, pts, b - 236, b);
    g.save(); clipPath(g, pts); windows(g, x + 2, b - 232, 40, 230, rnd, { cw: 2, ch: 3, gx: 2, gy: 3, density: .5 });
    for (let i = 0; i < 6; i++) glowFill(g, '#e8f4ff', 6, () => g.rect(x + 2, b - 200 + i * 30, 40, 1));
    g.restore();
    for (let i = 0; i < 5; i++) { // 楼顶「爪」
      const fx = x + 9 + i * 6, fh = i % 2 ? 14 : 20;
      g.fillStyle = P.top; g.fillRect(fx, b - 236 - fh, 3, fh);
      glowFill(g, '#e8f4ff', 8, () => g.rect(fx, b - 236 - fh, 3, 2));
    }
    return [[x + 22, b - 258]];
  } };
  L.BOC = { w: 52, draw(g, x, b) { // 香港 中银大厦
    const pts = [[x, b], [x, b - 150], [x + 20, b - 228], [x + 28, b - 188], [x + 38, b - 166], [x + 52, b - 112], [x + 52, b]];
    shape(g, pts, b - 228, b);
    g.save(); clipPath(g, pts);
    g.strokeStyle = 'rgba(235,245,255,.85)'; g.lineWidth = 1.3; g.shadowColor = '#cfe8ff'; g.shadowBlur = 6;
    g.beginPath();
    for (let k = -8; k < 10; k++) { const s = k * 26; g.moveTo(x + s, b); g.lineTo(x + s + 260, b - 260); g.moveTo(x + s + 52, b); g.lineTo(x + s + 52 - 260, b - 260); }
    for (const yy of [b - 2, b - 56, b - 112, b - 166]) { g.moveTo(x, yy); g.lineTo(x + 52, yy); }
    g.stroke();
    g.restore();
    g.strokeStyle = '#cfe8ff'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x + 20, b - 228); g.lineTo(x + 20, b - 262); g.moveTo(x + 28, b - 188); g.lineTo(x + 28, b - 212); g.stroke();
    return [[x + 20, b - 263], [x + 28, b - 213]];
  } };
  L.CENTER = { w: 36, draw(g, x, b, rnd) { // 香港 中环中心（彩虹灯带）
    const pts = [[x, b], [x, b - 196], [x + 18, b - 214], [x + 36, b - 196], [x + 36, b]];
    shape(g, pts, b - 214, b);
    g.save(); clipPath(g, pts);
    for (let i = 0; i < 24; i++) { g.fillStyle = `hsla(${(i * 22) % 360},100%,62%,.8)`; g.fillRect(x, b - 200 + i * 8, 36, 1.6); }
    g.restore();
    g.strokeStyle = '#a090ff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x + 18, b - 214); g.lineTo(x + 18, b - 240); g.stroke();
    return [[x + 18, b - 241]];
  } };
  L.MBS = { w: 150, draw(g, x, b, rnd) { // 新加坡 滨海湾金沙
    for (let i = 0; i < 3; i++) {
      const tx = x + 12 + i * 44, pts = [[tx - 4, b], [tx, b - 60], [tx, b - 150], [tx + 26, b - 150], [tx + 26, b - 60], [tx + 30, b]];
      shape(g, pts, b - 150, b);
      g.save(); clipPath(g, pts); windows(g, tx + 1, b - 146, 25, 144, rnd, { cw: 2, ch: 2, gx: 2, gy: 2, density: .55, colors: ['#ffe6a0', '#ffffff', '#ffd36b'] }); g.restore();
      g.fillStyle = P.bottom; g.beginPath(); g.moveTo(tx + 6, b); g.lineTo(tx + 13, b - 40); g.lineTo(tx + 20, b); g.closePath(); g.fill();
    }
    // 空中花园
    const sy = b - 160;
    g.fillStyle = P.top;
    g.beginPath(); g.moveTo(x + 2, sy + 4); g.lineTo(x + 146, sy); g.quadraticCurveTo(x + 166, sy + 2, x + 172, sy + 8); g.lineTo(x + 146, sy + 10); g.lineTo(x + 2, sy + 12); g.closePath(); g.fill();
    glowLine(g, P.accent, 1.6, 10, () => { g.moveTo(x + 2, sy + 4); g.lineTo(x + 146, sy); g.quadraticCurveTo(x + 166, sy + 2, x + 172, sy + 8); });
    g.fillStyle = '#3ee06b'; for (let i = 0; i < 22; i++) { g.beginPath(); g.arc(x + 10 + i * 6.4, sy + 1 - i * .02, 1.6, 0, Math.PI * 2); g.fill(); }
    return [[x + 25, sy - 2], [x + 125, sy - 4]];
  } };
  L.ARTSCI = { w: 60, draw(g, x, b) { // 新加坡 艺术科学博物馆（莲花）
    const cx = x + 30, by = b - 12;
    g.fillStyle = P.bottom; g.fillRect(cx - 8, by, 16, 12);
    for (let i = 0; i < 9; i++) {
      const a = -1.2 + i * .3, len = 36 + Math.cos(a) * 14;
      const tx = cx + Math.sin(a) * len, ty = by - Math.cos(a) * len * .9;
      const gr = g.createLinearGradient(cx, by, tx, ty);
      gr.addColorStop(0, '#2a1a50'); gr.addColorStop(1, '#f2ecff');
      g.fillStyle = gr;
      g.beginPath(); g.moveTo(cx - 3, by); g.quadraticCurveTo((cx + tx) / 2 - 6, (by + ty) / 2, tx, ty); g.quadraticCurveTo((cx + tx) / 2 + 6, (by + ty) / 2, cx + 3, by); g.fill();
    }
    glowFill(g, '#ff9ad5', 12, () => g.arc(cx, by - 2, 4, 0, Math.PI * 2));
    return [];
  } };
  L.FLYER = { w: 150, draw(g, x, b) { // 新加坡 摩天观景轮
    const cx = x + 75, cy = b - 96, R = 68;
    g.strokeStyle = '#2a1a50'; g.lineWidth = 4;
    g.beginPath(); g.moveTo(cx - 40, b); g.lineTo(cx, cy); g.lineTo(cx + 40, b); g.moveTo(cx - 20, b); g.lineTo(cx, cy); g.stroke();
    g.strokeStyle = 'rgba(200,220,255,.35)'; g.lineWidth = .8;
    g.beginPath();
    for (let i = 0; i < 28; i++) { const a = i / 28 * Math.PI * 2; g.moveTo(cx, cy); g.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R); }
    g.stroke();
    glowLine(g, P.accent, 2, 12, () => g.arc(cx, cy, R, 0, Math.PI * 2));
    glowLine(g, '#ff9ad5', 1, 8, () => g.arc(cx, cy, R - 6, 0, Math.PI * 2));
    for (let i = 0; i < 28; i++) { const a = i / 28 * Math.PI * 2; glowFill(g, '#ffffff', 6, () => g.arc(cx + Math.cos(a) * (R + 3), cy + Math.sin(a) * (R + 3), 2, 0, Math.PI * 2)); }
    return [];
  } };
  L.T101 = { w: 64, draw(g, x, b, rnd) { // 台北 101
    const cx = x + 32;
    const pod = [[x, b], [x + 2, b - 34], [x + 62, b - 34], [x + 64, b]];
    shape(g, pod, b - 34, b);
    g.save(); clipPath(g, pod); windows(g, x + 3, b - 32, 58, 30, rnd, { density: .4 }); g.restore();
    glowFill(g, '#ffd36b', 10, () => g.arc(cx, b - 40, 4, 0, Math.PI * 2));
    for (let i = 0; i < 8; i++) {
      const y0 = b - 44 - i * 27, y1 = y0 - 27;
      const seg = [[cx - 19, y0], [cx + 19, y0], [cx + 25, y1], [cx - 25, y1]];
      shape(g, seg, y1, y0);
      g.save(); clipPath(g, seg);
      windows(g, cx - 24, y1 + 3, 48, 24, rnd, { cw: 2, ch: 2, gx: 2, gy: 2, density: .3, colors: ['#9dffb0', '#e8fff0'] });
      const lg = g.createLinearGradient(0, y1, 0, y0);
      lg.addColorStop(0, `hsla(${140 + i * 8},90%,60%,.5)`); lg.addColorStop(1, 'hsla(160,90%,60%,0)');
      g.fillStyle = lg; g.fillRect(cx - 26, y1, 52, 27);
      g.restore();
      glowLine(g, '#7dffb0', 1.4, 8, () => { g.moveTo(cx - 25, y1); g.lineTo(cx + 25, y1); });
    }
    const topY = b - 44 - 8 * 27;
    shape(g, [[cx - 12, topY], [cx + 12, topY], [cx + 12, topY - 14], [cx - 12, topY - 14]], topY - 14, topY);
    shape(g, [[cx - 6, topY - 14], [cx + 6, topY - 14], [cx + 6, topY - 26], [cx - 6, topY - 26]], topY - 26, topY - 14);
    g.strokeStyle = '#c8d6ff'; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, topY - 26); g.lineTo(cx, topY - 70); g.stroke();
    return [[cx, topY - 71]];
  } };
  L.PEARL = { w: 64, draw(g, x, b) { // 上海 东方明珠
    const cx = x + 32;
    g.strokeStyle = '#3a2a6a'; g.lineWidth = 6;
    g.beginPath(); g.moveTo(x + 4, b); g.lineTo(cx - 6, b - 104); g.moveTo(x + 60, b); g.lineTo(cx + 6, b - 104); g.moveTo(cx, b); g.lineTo(cx, b - 104); g.stroke();
    g.fillStyle = P.top; g.fillRect(cx - 5, b - 250, 10, 150);
    const ball = (y, r) => {
      const gr = g.createRadialGradient(cx - r * .3, y - r * .3, r * .1, cx, y, r);
      gr.addColorStop(0, '#ffd0f0'); gr.addColorStop(.5, '#ff4fb8'); gr.addColorStop(1, '#6a0f5a');
      glowFill(g, '#ff4fb8', 18, () => g.arc(cx, y, r, 0, Math.PI * 2));
      g.fillStyle = gr; g.beginPath(); g.arc(cx, y, r, 0, Math.PI * 2); g.fill();
      g.fillStyle = 'rgba(255,255,255,.8)';
      for (let i = 0; i < 10; i++) { const a = i / 10 * Math.PI * 2; g.fillRect(cx + Math.cos(a) * r * .8 - .5, y + Math.sin(a) * r * .5 - .5, 1.2, 1.2); }
    };
    ball(b - 112, 25);
    for (let i = 0; i < 4; i++) glowFill(g, '#ff9ad5', 6, () => g.arc(cx, b - 150 - i * 18, 4, 0, Math.PI * 2));
    ball(b - 228, 16);
    ball(b - 262, 7);
    g.strokeStyle = '#d8c8ff'; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, b - 268); g.lineTo(cx, b - 312); g.stroke();
    return [[cx, b - 313]];
  } };
  L.JINMAO = { w: 40, draw(g, x, b, rnd) { // 上海 金茂大厦
    const cx = x + 20;
    let y = b, w = 34;
    const tiers = [80, 40, 30, 24, 18, 14, 12, 10];
    for (const h of tiers) {
      const pts = [[cx - w / 2, y], [cx - w / 2 + 1, y - h], [cx + w / 2 - 1, y - h], [cx + w / 2, y]];
      shape(g, pts, y - h, y, 'rgba(255,211,107,.8)');
      g.save(); clipPath(g, pts); windows(g, cx - w / 2 + 2, y - h + 2, w - 4, h - 3, rnd, { cw: 2, ch: 2, gx: 2, gy: 2, density: .45, colors: ['#ffd36b', '#fff1c4'] }); g.restore();
      y -= h; w -= 3;
    }
    g.strokeStyle = '#ffd36b'; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, y); g.lineTo(cx, y - 30); g.stroke();
    return [[cx, y - 31]];
  } };
  L.SWFC = { w: 42, draw(g, x, b, rnd) { // 上海 环球金融中心（开瓶器）
    const h = 290, pts = [[x, b], [x + 10, b - h], [x + 36, b - h + 8], [x + 42, b]];
    shape(g, pts, b - h, b);
    g.save(); clipPath(g, pts); windows(g, x + 2, b - h + 4, 40, h - 6, rnd, { cw: 2, ch: 2, gx: 2, gy: 3, density: .4 }); g.restore();
    g.save(); g.globalCompositeOperation = 'destination-out';
    g.beginPath(); g.moveTo(x + 14, b - h + 18); g.lineTo(x + 32, b - h + 22); g.lineTo(x + 30, b - h + 36); g.lineTo(x + 15, b - h + 34); g.closePath(); g.fill(); g.restore();
    glowLine(g, P.accent, 1.2, 8, () => { g.moveTo(x + 14, b - h + 18); g.lineTo(x + 32, b - h + 22); g.lineTo(x + 30, b - h + 36); g.lineTo(x + 15, b - h + 34); g.closePath(); });
    return [[x + 23, b - h - 1]];
  } };
  L.SHT = { w: 48, draw(g, x, b, rnd) { // 上海中心（螺旋上升）
    const h = 320;
    g.beginPath(); g.moveTo(x, b);
    g.bezierCurveTo(x + 2, b - 120, x + 6, b - 240, x + 14, b - h);
    g.lineTo(x + 30, b - h + 6); g.lineTo(x + 32, b - h + 16);
    g.bezierCurveTo(x + 40, b - 220, x + 46, b - 110, x + 48, b);
    g.closePath();
    g.fillStyle = bodyFill(g, b - h, b); g.fill();
    g.save(); g.shadowColor = P.edge; g.shadowBlur = 6; g.strokeStyle = P.edge; g.lineWidth = 1; g.stroke(); g.restore();
    g.save(); g.clip();
    windows(g, x + 2, b - h + 10, 46, h - 12, rnd, { cw: 2, ch: 2, gx: 2, gy: 3, density: .38 });
    g.strokeStyle = 'rgba(160,220,255,.7)'; g.lineWidth = 1.4; g.shadowColor = '#a0dcff'; g.shadowBlur = 6;
    g.beginPath();
    for (let i = 0; i < 9; i++) { const y0 = b - i * 36; g.moveTo(x, y0); g.bezierCurveTo(x + 16, y0 - 12, x + 34, y0 - 22, x + 50, y0 - 40); }
    g.stroke();
    g.restore();
    return [[x + 22, b - h - 1]];
  } };
  L.TOKYO = { w: 70, draw(g, x, b) { // 东京塔
    const cx = x + 35, h = 196;
    const out = [[x, b], [cx - 8, b - 110], [cx - 4, b - 150], [cx - 2, b - h], [cx + 2, b - h], [cx + 4, b - 150], [cx + 8, b - 110], [x + 70, b], [x + 56, b], [cx, b - 60], [x + 14, b]];
    g.beginPath(); g.moveTo(out[0][0], out[0][1]); for (const p of out.slice(1)) g.lineTo(p[0], p[1]); g.closePath();
    const gr = g.createLinearGradient(0, b - h, 0, b);
    gr.addColorStop(0, '#ffffff'); gr.addColorStop(.15, '#ff6a2b'); gr.addColorStop(.4, '#ffffff'); gr.addColorStop(.45, '#ff5a1f'); gr.addColorStop(1, '#c2410c');
    g.save(); g.shadowColor = '#ff7b2a'; g.shadowBlur = 16; g.fillStyle = gr; g.fill(); g.restore();
    g.save(); g.clip(); g.strokeStyle = 'rgba(80,20,0,.55)'; g.lineWidth = .8; g.beginPath();
    for (let y = b; y > b - h; y -= 10) { g.moveTo(x, y); g.lineTo(x + 70, y - 10); g.moveTo(x + 70, y); g.lineTo(x, y - 10); }
    g.stroke(); g.restore();
    g.fillStyle = '#ffe6c0'; g.fillRect(cx - 12, b - 112, 24, 7); g.fillRect(cx - 6, b - 152, 12, 5);
    return [[cx, b - h - 1]];
  } };
  L.SKYTREE = { w: 50, draw(g, x, b) { // 东京晴空塔
    const cx = x + 25, h = 324;
    const pts = [[x, b], [cx - 7, b - 210], [cx - 5, b - 280], [cx + 5, b - 280], [cx + 7, b - 210], [x + 50, b]];
    shape(g, pts, b - h, b, 'rgba(120,200,255,.8)');
    g.save(); clipPath(g, pts);
    g.strokeStyle = 'rgba(140,210,255,.75)'; g.lineWidth = .9; g.shadowColor = '#8cd2ff'; g.shadowBlur = 5; g.beginPath();
    for (let y = b; y > b - 290; y -= 12) { g.moveTo(x - 10, y); g.lineTo(x + 60, y - 14); g.moveTo(x + 60, y); g.lineTo(x - 10, y - 14); }
    g.stroke(); g.restore();
    for (const [y, w, hh] of [[b - 214, 26, 12], [b - 262, 18, 7]]) glowFill(g, '#b48cff', 12, () => g.rect(cx - w / 2, y, w, hh));
    g.strokeStyle = '#d8e8ff'; g.lineWidth = 2; g.beginPath(); g.moveTo(cx, b - 280); g.lineTo(cx, b - h); g.stroke();
    return [[cx, b - h - 1]];
  } };
  L.BURJ = { w: 66, draw(g, x, b, rnd) { // 迪拜 哈利法塔
    const cx = x + 33;
    const tiers = [[66, 0], [58, 60], [50, 110], [42, 150], [34, 186], [26, 216], [20, 242], [14, 264], [9, 284], [5, 300]];
    const pts = [[x, b]];
    for (let i = 0; i < tiers.length; i++) { const [w, hy] = tiers[i], nh = i + 1 < tiers.length ? tiers[i + 1][1] : 312; pts.push([cx - w / 2, b - hy], [cx - w / 2, b - nh]); }
    pts.push([cx - 1, b - 312], [cx, b - 336], [cx + 1, b - 312]);
    for (let i = tiers.length - 1; i >= 0; i--) { const [w, hy] = tiers[i], nh = i + 1 < tiers.length ? tiers[i + 1][1] : 312; pts.push([cx + w / 2, b - nh], [cx + w / 2, b - hy]); }
    shape(g, pts, b - 336, b, 'rgba(180,220,255,.85)');
    g.save(); clipPath(g, pts);
    for (let y = b - 4; y > b - 300; y -= 5) { g.fillStyle = `rgba(200,230,255,${.15 + rnd() * .35})`; g.fillRect(cx - 33, y, 66, 1); }
    g.fillStyle = 'rgba(160,210,255,.25)'; g.fillRect(cx - 1.5, b - 312, 3, 312);
    g.restore();
    return [[cx, b - 337]];
  } };
  L.ARAB = { w: 56, draw(g, x, b) { // 迪拜 帆船酒店
    g.beginPath(); g.moveTo(x, b); g.lineTo(x + 2, b - 186); g.quadraticCurveTo(x + 56, b - 120, x + 54, b); g.closePath();
    const gr = g.createLinearGradient(x, 0, x + 56, 0);
    gr.addColorStop(0, '#2a1a60'); gr.addColorStop(1, '#8a6cff');
    g.save(); g.shadowColor = '#b48cff'; g.shadowBlur = 14; g.fillStyle = gr; g.fill(); g.restore();
    g.strokeStyle = 'rgba(255,255,255,.75)'; g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(x + 2, b - 186); g.quadraticCurveTo(x + 56, b - 120, x + 54, b);
    for (let i = 1; i < 7; i++) { g.moveTo(x + 2, b - 186 + i * 26); g.lineTo(x + 8 + i * 6.5, b - 186 + i * 26 + 4); }
    g.stroke();
    g.fillStyle = '#ffffff'; g.fillRect(x - 1, b - 200, 3, 18);
    glowFill(g, '#ffd36b', 8, () => g.ellipse(x + 34, b - 132, 9, 2.5, 0, 0, Math.PI * 2));
    return [[x, b - 201]];
  } };

  const CITIES = {
    香港: { harbor: true, gen: [70, 210], items: ['g', 'g', 'ICC', 'g', 'g', 'g', 'IFC', 'g', 'CENTER', 'g', 'BOC', 'g', 'g', 'g'] },
    新加坡: { harbor: true, gen: [60, 170], items: ['g', 'g', 'g', 'MBS', 'g', 'ARTSCI', 'FLYER', 'g', 'g', 'g', 'g'] },
    台北: { harbor: false, gen: [40, 120], items: ['g', 'g', 'g', 'g', 'T101', 'g', 'g', 'g', 'g', 'g'] },
    上海: { harbor: true, gen: [60, 180], items: ['g', 'PEARL', 'g', 'g', 'JINMAO', 'SWFC', 'SHT', 'g', 'g', 'g'] },
    东京: { harbor: false, gen: [50, 170], items: ['g', 'g', 'TOKYO', 'g', 'g', 'g', 'SKYTREE', 'g', 'g', 'g'] },
    迪拜: { harbor: true, gen: [60, 200], items: ['g', 'g', 'BURJ', 'g', 'g', 'ARAB', 'g', 'g', 'g'] },
  };
  const ORDER = ['香港', '新加坡', '台北', '上海', '东京', '迪拜'];
  const cache = {};

  function build(name) {
    if (cache[name]) return cache[name];
    const spec = CITIES[name], rnd = mulberry(seedOf(name));
    const items = [];
    let x = 30;
    for (const tk of spec.items) {
      const w = tk === 'g' ? 22 + rnd() * 30 : L[tk].w;
      items.push({ tk, x, w, h: spec.gen[0] + rnd() * (spec.gen[1] - spec.gen[0]) });
      x += w + 6 + rnd() * 12;
    }
    const CW = Math.max(980, Math.ceil(x + 30));
    const c = document.createElement('canvas');
    c.width = CW * SCALE; c.height = SKY_H * SCALE;
    const g = c.getContext('2d');
    g.scale(SCALE, SCALE);
    const b = SKY_H;
    // 远景楼群（雾化）
    g.globalAlpha = .55;
    for (let bx = -10; bx < CW; bx += 14 + rnd() * 16) {
      const w = 16 + rnd() * 22, h = 30 + rnd() * spec.gen[0] * 1.2;
      g.fillStyle = '#1a0d3a'; g.fillRect(bx, b - h, w, h);
      for (let k = 0; k < 6; k++) { g.fillStyle = P.win[Math.floor(rnd() * 3)]; g.globalAlpha = .25; g.fillRect(bx + 2 + rnd() * (w - 4), b - h + 4 + rnd() * (h - 8), 2, 2); }
      g.globalAlpha = .55;
    }
    g.globalAlpha = 1;
    const haze = g.createLinearGradient(0, b - 160, 0, b);
    haze.addColorStop(0, 'rgba(255,63,142,0)'); haze.addColorStop(1, 'rgba(255,63,142,.18)');
    g.fillStyle = haze; g.fillRect(0, b - 160, CW, 160);
    // 近景：先画普通楼，再画地标压在上面
    const lights = [];
    for (const it of items) if (it.tk === 'g') lights.push(...generic(g, it.x, b, it.w, it.h, rnd));
    for (const it of items) if (it.tk !== 'g') lights.push(...L[it.tk].draw(g, it.x, b, rnd));
    return (cache[name] = { canvas: c, w: CW, lights, harbor: spec.harbor });
  }

  // 在主画布上绘制城市；base 为楼宇地基所在的 y
  function draw(ctx, name, cam, alpha, base, groundY, W, now) {
    const city = build(name);
    const off = ((cam * .22) % city.w + city.w) % city.w;
    ctx.save();
    ctx.globalAlpha = alpha;
    for (let dx = -off; dx < W; dx += city.w) {
      ctx.drawImage(city.canvas, dx, base - SKY_H, city.w, SKY_H);
      for (const [lx, ly] of city.lights) { // 航空障碍灯闪烁
        if (Math.floor(now / 600 + lx) % 2) continue;
        const sx = dx + lx;
        if (sx < -5 || sx > W + 5) continue;
        ctx.fillStyle = '#ff2244'; ctx.shadowColor = '#ff2244'; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.arc(sx, base - SKY_H + ly, 2, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
    if (city.harbor && groundY > base) { // 海港倒影
      const hh = groundY - base;
      const wg = ctx.createLinearGradient(0, base, 0, groundY);
      wg.addColorStop(0, '#120a35'); wg.addColorStop(1, '#05021a');
      ctx.fillStyle = wg; ctx.fillRect(0, base, W, hh);
      ctx.save();
      ctx.beginPath(); ctx.rect(0, base, W, hh); ctx.clip();
      ctx.globalAlpha = alpha * .35;
      ctx.translate(0, base); ctx.scale(1, -.4);
      for (let dx = -off; dx < W; dx += city.w) ctx.drawImage(city.canvas, dx, -SKY_H, city.w, SKY_H);
      ctx.restore();
      ctx.globalAlpha = alpha * .5;
      ctx.fillStyle = '#9fd8ff';
      for (let i = 0; i < 26; i++) {
        const y = base + 3 + (i * 7) % hh, x = ((i * 137 - cam * .3 + Math.sin(now / 500 + i) * 20) % W + W) % W;
        ctx.fillRect(x, y, 18 + (i % 4) * 10, 1);
      }
    }
    ctx.restore();
  }

  root.RunnerCity = { draw, build, ORDER, SKY_H };
})(this);
