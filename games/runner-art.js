// 跑酷角色美术：英雄、宠物、坐骑。全部用 Canvas 路径绘制，不依赖图片素材。
// 坐标约定：英雄以脚底为原点，面朝右；角度以「竖直向下」为 0，向前（右）为正。
(function (root) {
  const HEROES = {
    flame: { name: '赤焰战神', desc: '烈焰重甲 · 龙角战盔', armor: '#2a1214', armor2: '#4d1a1d', plate: '#6b2226', trim: '#ffc53d',
             cape: ['#ff3b2f', '#6e0912'], eye: '#ffe066', glow: '#ff5a1f', helm: 'horns', weapon: 'great' },
    ninja: { name: '霓虹忍者', desc: '暗影夜行 · 等离子太刀', armor: '#0f1226', armor2: '#1b2150', plate: '#262d6b', trim: '#33f0ff',
             cape: ['#33f0ff', '#0b2c66'], eye: '#33f0ff', glow: '#33f0ff', helm: 'mask', weapon: 'katana', scarf: true },
    frost: { name: '玄冰帝王', desc: '冰晶王冠 · 寒霜长枪', armor: '#b9d4f2', armor2: '#e9f4ff', plate: '#8fb3df', trim: '#7af7ff',
             cape: ['#4d8dff', '#0b1f66'], eye: '#b2ff59', glow: '#7af7ff', helm: 'crown', weapon: 'spear' },
  };
  const PETS = {
    dragon: { name: '小火龙', desc: '会喷小火苗' },
    fox: { name: '九尾灵狐', desc: '九条发光尾巴' },
    panda: { name: '胖达', desc: '竹蜻蜓飞行' },
  };
  const MOUNTS = {
    wolf: { name: '机甲战狼', desc: '钢铁獠牙', icon: '🐺', color: '#33f0ff' },
    horse: { name: '烈焰战马', desc: '火焰鬃毛', icon: '🐴', color: '#ff7b2a' },
    drake: { name: '雷霆飞龙', desc: '闪电双翼', icon: '🐉', color: '#b26bff' },
  };

  const TAU = Math.PI * 2;
  function rr(g, x, y, w, h, r) { g.beginPath(); if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h); }
  function withGlow(g, color, blur, fn) { g.save(); g.shadowColor = color; g.shadowBlur = blur; fn(); g.restore(); }

  // 两节肢体（大腿+小腿 / 上臂+前臂），返回关节与末端
  function limb(g, x1, y1, a1, l1, a2, l2, w, color) {
    const x2 = x1 + Math.sin(a1) * l1, y2 = y1 + Math.cos(a1) * l1;
    const x3 = x2 + Math.sin(a2) * l2, y3 = y2 + Math.cos(a2) * l2;
    g.strokeStyle = color; g.lineWidth = w; g.lineCap = 'round'; g.lineJoin = 'round';
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.lineTo(x3, y3); g.stroke();
    return [x2, y2, x3, y3];
  }

  // ---------- 英雄 ----------
  // pose: { mode: run|air|slide|ride|dead, ph: 跑步相位, speed: 0~1, t: 秒 }
  function drawHero(g, x, y, hero, pose) {
    const H = typeof hero === 'string' ? HEROES[hero] : hero;
    const { mode = 'run', ph = 0, speed = .5, t = 0 } = pose;
    g.save();
    g.translate(x, y);
    if (mode === 'slide') { g.translate(30, -6); g.rotate(-1.2); }
    else g.rotate(mode === 'ride' ? .05 : .1 + speed * .08);

    // 姿势参数
    let legF, legB, armF, armB;
    if (mode === 'air' || mode === 'dead') {
      legF = [1.3, .2]; legB = [-.5, -1.5]; armF = [1.7, 2.4]; armB = [-1, -.3];
    } else if (mode === 'slide') {
      legF = [.15, .1]; legB = [-.05, -.1]; armF = [2.6, 3]; armB = [2.2, 2.7];
    } else if (mode === 'ride') {
      legF = [1.45, .25]; legB = [1.3, .1]; armF = [1.1, 1.75]; armB = [.9, 1.6];
    } else {
      const s = Math.sin(ph), sb = Math.sin(ph + Math.PI);
      legF = [.85 * s, .85 * s - (.3 + 1.1 * Math.max(0, Math.sin(ph - 1.4)))];
      legB = [.85 * sb, .85 * sb - (.3 + 1.1 * Math.max(0, Math.sin(ph + Math.PI - 1.4)))];
      armF = [-.9 * s, -.9 * s + 1.3]; armB = [-.9 * sb, -.9 * sb + 1.3];
    }
    const hip = [0, -36], sh = [2, -60];

    // 披风 / 围巾
    const wave = Math.sin(t * 14) * 5, wave2 = Math.sin(t * 14 + 1.4) * 6;
    const L = 30 + speed * 32 + (mode === 'air' ? 8 : 0);
    if (!H.scarf) {
      const cg = g.createLinearGradient(0, -64, -L, -20);
      cg.addColorStop(0, H.cape[0]); cg.addColorStop(1, H.cape[1]);
      g.fillStyle = cg;
      g.beginPath();
      g.moveTo(-4, -65);
      g.quadraticCurveTo(-L * .45, -64 + wave * .4, -L, -46 + wave);
      g.lineTo(-L + 4, -34 + wave2);
      g.quadraticCurveTo(-L * .5, -22 + wave2, -10, -30);
      g.lineTo(6, -62);
      g.closePath(); g.fill();
      g.strokeStyle = H.trim; g.lineWidth = 1.2; g.globalAlpha = .7;
      g.beginPath(); g.moveTo(-10, -30); g.quadraticCurveTo(-L * .5, -22 + wave2, -L + 4, -34 + wave2); g.stroke();
      g.globalAlpha = 1;
    } else {
      withGlow(g, H.glow, 10, () => { // 两条飘带，从颈部向后下方垂落
        g.fillStyle = H.cape[0];
        for (let k = 0; k < 2; k++) {
          const len = L * .75 + k * 6, sag = 8 + k * 6, w0 = 4 - k;
          const ex = -len, ey = -60 + sag + (k ? wave2 : wave);
          g.beginPath(); g.moveTo(0, -68);
          g.quadraticCurveTo(-len * .5, -68 + sag * .3 + wave * .5, ex, ey);
          g.lineTo(ex + 3, ey + w0);
          g.quadraticCurveTo(-len * .5, -64 + sag * .3 + wave * .5, 1, -64);
          g.closePath(); g.fill();
        }
      });
    }

    // 背后武器
    g.save();
    if (H.weapon === 'katana') {
      g.strokeStyle = '#2a2f45'; g.lineWidth = 3.5; g.beginPath(); g.moveTo(-14, -34); g.lineTo(-4, -50); g.stroke();
      withGlow(g, H.glow, 14, () => { g.strokeStyle = H.trim; g.lineWidth = 2.5; g.beginPath(); g.moveTo(-4, -50); g.quadraticCurveTo(4, -76, 10, -98); g.stroke(); });
      g.fillStyle = H.trim; g.fillRect(-7, -53, 8, 3);
    } else if (H.weapon === 'spear') {
      g.strokeStyle = '#6e86a8'; g.lineWidth = 3; g.beginPath(); g.moveTo(-22, -22); g.lineTo(12, -100); g.stroke();
      withGlow(g, H.glow, 16, () => {
        g.fillStyle = H.trim; g.beginPath(); g.moveTo(12, -100); g.lineTo(9, -112); g.lineTo(17, -120); g.lineTo(18, -106); g.closePath(); g.fill();
      });
    } else {
      g.fillStyle = '#3a3f52';
      g.beginPath(); g.moveTo(-18, -28); g.lineTo(-10, -32); g.lineTo(14, -92); g.lineTo(10, -100); g.lineTo(4, -90); g.closePath(); g.fill();
      withGlow(g, H.glow, 12, () => { g.strokeStyle = H.glow; g.lineWidth = 1.5; g.beginPath(); g.moveTo(-12, -32); g.lineTo(9, -96); g.stroke(); });
      g.fillStyle = H.trim; g.fillRect(-20, -34, 12, 4);
    }
    g.restore();

    // 后侧肢体
    const back = mode === 'ride' ? H.armor : H.armor;
    limb(g, sh[0] - 2, sh[1] + 2, armB[0], 13, armB[1], 13, 7, back);
    const bl = limb(g, hip[0] - 2, hip[1], legB[0], 18, legB[1], 18, 9, back);
    g.fillStyle = H.armor2; rr(g, bl[2] - 4, bl[3] - 3, 11, 6, 3); g.fill();

    // 躯干
    const tg = g.createLinearGradient(0, -66, 0, -34);
    tg.addColorStop(0, H.armor2); tg.addColorStop(1, H.armor);
    g.fillStyle = tg;
    g.beginPath(); g.moveTo(-11, -66); g.lineTo(13, -66); g.lineTo(11, -38); g.lineTo(-9, -36); g.closePath(); g.fill();
    g.fillStyle = H.plate;
    g.beginPath(); g.moveTo(-6, -62); g.lineTo(12, -62); g.lineTo(9, -46); g.lineTo(2, -42); g.lineTo(-5, -46); g.closePath(); g.fill();
    withGlow(g, H.glow, 14, () => {
      g.fillStyle = H.glow;
      g.beginPath(); g.moveTo(3, -58); g.lineTo(7, -53); g.lineTo(3, -48); g.lineTo(-1, -53); g.closePath(); g.fill();
    });
    g.fillStyle = H.trim; g.fillRect(-9, -40, 21, 3);
    g.fillStyle = H.trim; g.fillRect(0, -41.5, 6, 6);
    g.strokeStyle = 'rgba(255,255,255,.3)'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(-8, -63); g.lineTo(-6, -44); g.stroke();

    // 前侧腿
    const fl = limb(g, hip[0] + 1, hip[1], legF[0], 18, legF[1], 18, 10, H.armor);
    g.fillStyle = H.trim; g.beginPath(); g.arc(fl[0], fl[1], 3.2, 0, TAU); g.fill();
    g.fillStyle = H.armor2; rr(g, fl[2] - 4, fl[3] - 3, 12, 6, 3); g.fill();
    withGlow(g, H.glow, 8, () => { g.fillStyle = H.glow; g.fillRect(fl[2] - 2, fl[3] + 1.5, 9, 1.5); });

    // 头部
    const hx = 5, hy = -73;
    if (H.helm === 'crown') { // 飘逸长发
      g.fillStyle = '#e9f6ff';
      g.beginPath(); g.moveTo(hx - 6, hy - 9);
      g.quadraticCurveTo(hx - 20, hy - 4 + wave * .5, hx - 30 - speed * 10, hy + 6 + wave);
      g.quadraticCurveTo(hx - 16, hy + 8, hx - 4, hy + 6); g.closePath(); g.fill();
    }
    // 头盔：圆顶 + 前探面甲 + 护颊，带金属高光
    const helm = g.createLinearGradient(hx, hy - 12, hx, hy + 10);
    helm.addColorStop(0, H.armor2); helm.addColorStop(1, H.armor);
    g.fillStyle = helm;
    g.beginPath();
    g.moveTo(hx - 9, hy + 8);
    g.quadraticCurveTo(hx - 13, hy - 6, hx - 4, hy - 11);
    g.quadraticCurveTo(hx + 8, hy - 15, hx + 12, hy - 4);
    g.lineTo(hx + 15, hy + 1);
    g.lineTo(hx + 11, hy + 4);
    g.lineTo(hx + 10, hy + 10);
    g.lineTo(hx + 1, hy + 11);
    g.closePath(); g.fill();
    g.strokeStyle = 'rgba(0,0,0,.55)'; g.lineWidth = 1; g.stroke();
    g.strokeStyle = 'rgba(255,255,255,.35)'; g.lineWidth = 1.6;
    g.beginPath(); g.moveTo(hx - 7, hy - 7); g.quadraticCurveTo(hx, hy - 12, hx + 7, hy - 9); g.stroke();
    g.fillStyle = H.helm === 'mask' ? '#05060f' : H.plate;  // 面甲
    g.beginPath(); g.moveTo(hx + 4, hy - 4); g.lineTo(hx + 13, hy - 2); g.lineTo(hx + 10, hy + 9); g.lineTo(hx + 4, hy + 8); g.closePath(); g.fill();
    g.strokeStyle = H.trim; g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(hx + 1, hy + 10); g.lineTo(hx + 3, hy - 4); g.lineTo(hx + 12, hy - 4); g.stroke();
    withGlow(g, H.eye, 14, () => {
      g.fillStyle = H.eye;
      g.beginPath(); g.moveTo(hx + 5, hy - 1); g.lineTo(hx + 13, hy - 2); g.lineTo(hx + 12, hy + .8); g.lineTo(hx + 5, hy + 1.4); g.closePath(); g.fill();
    });
    if (H.helm === 'horns') {
      const hg = g.createLinearGradient(hx, hy - 10, hx - 22, hy - 28);
      hg.addColorStop(0, '#fff2b0'); hg.addColorStop(1, H.trim);
      g.fillStyle = hg;
      g.beginPath(); g.moveTo(hx - 4, hy - 8); g.quadraticCurveTo(hx - 10, hy - 28, hx - 26, hy - 30); g.quadraticCurveTo(hx - 12, hy - 22, hx - 9, hy - 5); g.closePath(); g.fill();
      g.beginPath(); g.moveTo(hx + 3, hy - 10); g.quadraticCurveTo(hx + 2, hy - 28, hx - 10, hy - 34); g.quadraticCurveTo(hx - 2, hy - 24, hx - 2, hy - 9); g.closePath(); g.fill();
      g.fillStyle = H.cape[0]; g.beginPath(); g.moveTo(hx - 2, hy - 12); g.lineTo(hx + 8, hy - 8); g.lineTo(hx - 6, hy - 6); g.closePath(); g.fill();
    } else if (H.helm === 'mask') {
      g.fillStyle = H.trim; g.fillRect(hx - 11, hy - 6, 22, 2.5);
      g.fillStyle = H.armor2; g.beginPath(); g.moveTo(hx - 10, hy - 7); g.lineTo(hx - 2, hy - 18); g.lineTo(hx + 2, hy - 9); g.closePath(); g.fill();
    } else if (H.helm === 'crown') {
      withGlow(g, H.glow, 14, () => {
        g.fillStyle = H.trim;
        const spikes = [[-8, 10], [-3, 15], [2, 19], [7, 14], [11, 9]];
        for (const [dx, h] of spikes) { g.beginPath(); g.moveTo(hx + dx - 3, hy - 8); g.lineTo(hx + dx, hy - 8 - h); g.lineTo(hx + dx + 3, hy - 8); g.closePath(); g.fill(); }
        g.fillRect(hx - 11, hy - 10, 23, 3);
      });
    }

    // 前侧手臂与肩甲
    const fa = limb(g, sh[0] + 2, sh[1] + 2, armF[0], 13, armF[1], 13, 8, H.armor);
    g.fillStyle = H.trim; g.beginPath(); g.arc(fa[2], fa[3], 3.6, 0, TAU); g.fill();
    // 肩甲：两层叠片
    const px0 = sh[0] + 2, py0 = sh[1] + 1;
    const plate = (ox, oy, w, color) => {
      const pg = g.createLinearGradient(ox, oy - w, ox, oy + 4);
      pg.addColorStop(0, H.armor2); pg.addColorStop(1, color);
      g.fillStyle = pg;
      g.beginPath(); g.moveTo(ox - w, oy + 3); g.quadraticCurveTo(ox - w * .2, oy - w * 1.15, ox + w, oy + 2);
      g.lineTo(ox + w - 2, oy + 5); g.quadraticCurveTo(ox - w * .2, oy - w * .45, ox - w + 2, oy + 6); g.closePath(); g.fill();
      g.strokeStyle = H.trim; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(ox - w + 2, oy + 6); g.quadraticCurveTo(ox - w * .2, oy - w * .45, ox + w - 2, oy + 5); g.stroke();
    };
    plate(px0, py0 + 4, 8, H.plate);
    plate(px0 - 1, py0, 10, H.plate);
    g.fillStyle = H.trim;
    g.beginPath(); g.moveTo(px0 - 6, py0 - 4); g.lineTo(px0 - 12, py0 - 12); g.lineTo(px0 - 2, py0 - 6); g.closePath(); g.fill();

    g.restore();
  }

  // 英雄身后的气场（狂热、冲刺时）
  function drawAura(g, x, y, hero, k, t) {
    const H = HEROES[hero];
    g.save(); g.globalCompositeOperation = 'lighter';
    const ag = g.createRadialGradient(x, y - 40, 6, x, y - 40, 70);
    ag.addColorStop(0, H.glow + 'aa'); ag.addColorStop(1, H.glow + '00');
    g.globalAlpha = k * (.6 + .3 * Math.sin(t * 10));
    g.fillStyle = ag; g.beginPath(); g.ellipse(x, y - 40, 50, 70, 0, 0, TAU); g.fill();
    g.restore();
  }

  // ---------- 宠物 ----------
  function drawPet(g, x, y, type, t) {
    g.save(); g.translate(x, y);
    if (type === 'dragon') {
      const flap = Math.sin(t * 16);
      // 尾巴与尾焰
      g.strokeStyle = '#ff7a2a'; g.lineWidth = 5; g.lineCap = 'round';
      g.beginPath(); g.moveTo(-8, 4); g.quadraticCurveTo(-20, 10, -24, 0 + Math.sin(t * 6) * 3); g.stroke();
      withGlow(g, '#ffb000', 12, () => {
        g.fillStyle = Math.random() > .5 ? '#ffd23d' : '#ff6a00';
        g.beginPath(); g.ellipse(-27, Math.sin(t * 6) * 3 - 2, 4, 6 + Math.random() * 2, .6, 0, TAU); g.fill();
      });
      // 翅膀
      g.fillStyle = '#ff9a4d';
      g.beginPath(); g.moveTo(-2, -6); g.lineTo(-16, -18 - flap * 8); g.lineTo(-12, -6 - flap * 3); g.lineTo(-20, -8 - flap * 6); g.lineTo(-4, 0); g.closePath(); g.fill();
      // 身体
      const bg = g.createRadialGradient(-2, -2, 2, 0, 2, 14);
      bg.addColorStop(0, '#ff9a3c'); bg.addColorStop(1, '#e8401c');
      g.fillStyle = bg; g.beginPath(); g.ellipse(0, 3, 11, 9, 0, 0, TAU); g.fill();
      g.fillStyle = '#ffe08a'; g.beginPath(); g.ellipse(3, 6, 6, 5, 0, 0, TAU); g.fill();
      // 头
      g.fillStyle = '#ff7a2a'; g.beginPath(); g.arc(9, -6, 9, 0, TAU); g.fill();
      g.fillStyle = '#fff4c2'; g.beginPath(); g.moveTo(5, -13); g.lineTo(2, -21); g.lineTo(8, -14); g.fill(); g.beginPath(); g.moveTo(10, -14); g.lineTo(11, -22); g.lineTo(14, -13); g.fill();
      g.fillStyle = '#fff'; g.beginPath(); g.arc(12, -7, 3.6, 0, TAU); g.fill();
      g.fillStyle = '#1a1020'; g.beginPath(); g.arc(13, -7, 2.2, 0, TAU); g.fill();
      g.fillStyle = '#fff'; g.beginPath(); g.arc(13.8, -8, .9, 0, TAU); g.fill();
      g.fillStyle = '#ff8fb0'; g.globalAlpha = .7; g.beginPath(); g.arc(11, -2, 2, 0, TAU); g.fill(); g.globalAlpha = 1;
    } else if (type === 'fox') {
      // 九条尾巴
      for (let i = 0; i < 9; i++) {
        const a = -2.2 + i * .19 + Math.sin(t * 4 + i * .7) * .08;
        g.save(); g.rotate(a);
        const tg = g.createLinearGradient(0, 0, 0, -30);
        tg.addColorStop(0, '#ffffff'); tg.addColorStop(.7, '#ffe3f6'); tg.addColorStop(1, '#ff7ad9');
        g.fillStyle = tg;
        g.globalAlpha = .92;
        g.beginPath(); g.moveTo(0, 0); g.quadraticCurveTo(-7, -16, 0, -30); g.quadraticCurveTo(7, -16, 0, 0); g.fill();
        g.restore();
      }
      g.globalAlpha = 1;
      withGlow(g, '#ff7ad9', 12, () => { g.fillStyle = '#ffffff'; g.beginPath(); g.ellipse(0, 4, 11, 8, 0, 0, TAU); g.fill(); });
      g.fillStyle = '#ffffff'; g.beginPath(); g.arc(8, -5, 9, 0, TAU); g.fill();
      g.beginPath(); g.moveTo(14, -6); g.lineTo(20, -3); g.lineTo(14, -1); g.fill();
      g.beginPath(); g.moveTo(2, -10); g.lineTo(2, -22); g.lineTo(9, -13); g.fill();
      g.beginPath(); g.moveTo(9, -13); g.lineTo(14, -23); g.lineTo(15, -10); g.fill();
      g.fillStyle = '#ff9ad5'; g.beginPath(); g.moveTo(4, -12); g.lineTo(4, -19); g.lineTo(8, -13); g.fill();
      g.strokeStyle = '#3a2440'; g.lineWidth = 1.6; g.lineCap = 'round';
      g.beginPath(); g.arc(10, -6, 2.4, Math.PI * 1.1, Math.PI * 1.9); g.stroke();
      g.fillStyle = '#3a2440'; g.beginPath(); g.arc(19.5, -3, 1.2, 0, TAU); g.fill();
      g.fillStyle = '#ff9ad5'; g.globalAlpha = .6; g.beginPath(); g.arc(9, -1, 2, 0, TAU); g.fill(); g.globalAlpha = 1;
    } else { // panda
      g.strokeStyle = '#555'; g.lineWidth = 2; g.beginPath(); g.moveTo(0, -18); g.lineTo(0, -24); g.stroke();
      const blade = Math.sin(t * 40) * 14;
      g.strokeStyle = '#ff5b9b'; g.lineWidth = 3; g.beginPath(); g.moveTo(-blade, -25); g.lineTo(blade, -25); g.stroke();
      g.fillStyle = '#ffd23d'; g.beginPath(); g.arc(0, -25, 2.5, 0, TAU); g.fill();
      g.fillStyle = '#ffffff'; g.beginPath(); g.ellipse(0, 6, 12, 10, 0, 0, TAU); g.fill();
      g.fillStyle = '#1d1d24'; g.beginPath(); g.ellipse(-8, 12, 4, 5, 0, 0, TAU); g.fill(); g.beginPath(); g.ellipse(8, 12, 4, 5, 0, 0, TAU); g.fill();
      g.fillStyle = '#ffffff'; g.beginPath(); g.arc(0, -6, 12, 0, TAU); g.fill();
      g.fillStyle = '#1d1d24';
      g.beginPath(); g.arc(-9, -15, 4.5, 0, TAU); g.fill(); g.beginPath(); g.arc(9, -15, 4.5, 0, TAU); g.fill();
      g.beginPath(); g.ellipse(-4.5, -6, 3.4, 4.2, -.5, 0, TAU); g.fill(); g.beginPath(); g.ellipse(4.5, -6, 3.4, 4.2, .5, 0, TAU); g.fill();
      g.fillStyle = '#fff'; g.beginPath(); g.arc(-4, -6.5, 1.3, 0, TAU); g.fill(); g.beginPath(); g.arc(5, -6.5, 1.3, 0, TAU); g.fill();
      g.fillStyle = '#1d1d24'; g.beginPath(); g.ellipse(0, -1.5, 2, 1.4, 0, 0, TAU); g.fill();
      g.fillStyle = '#ff9ab8'; g.globalAlpha = .7; g.beginPath(); g.arc(-8, -1, 2, 0, TAU); g.fill(); g.beginPath(); g.arc(8, -1, 2, 0, TAU); g.fill(); g.globalAlpha = 1;
    }
    g.restore();
  }

  // ---------- 坐骑 ----------
  // (x, y) 是坐骑脚下地面中心点；返回鞍座位置（英雄臀部应在此处）
  function drawMount(g, x, y, type, t, ph, onGround) {
    g.save(); g.translate(x, y);
    let saddle = [-4, -54];
    const legA = k => (onGround ? Math.sin(ph + k) * .75 : (k % 2 ? -.9 : .9));
    const drawLeg = (hx, hy, a, len, w, color, hoof) => {
      const k = Math.max(0, Math.sin(a * 2)) * .8;
      const p = limb(g, hx, hy, a, len, a - k, len * .9, w, color);
      if (hoof) { g.fillStyle = hoof; g.beginPath(); g.ellipse(p[2] + 2, p[3], 5, 3, 0, 0, TAU); g.fill(); }
      return p;
    };
    if (type === 'wolf') {
      // 能量尾
      withGlow(g, '#33f0ff', 16, () => {
        const tg = g.createLinearGradient(-44, -44, -86, -50);
        tg.addColorStop(0, 'rgba(51,240,255,.95)'); tg.addColorStop(1, 'rgba(51,240,255,0)');
        g.fillStyle = tg;
        g.beginPath(); g.moveTo(-40, -48); g.quadraticCurveTo(-66, -62 + Math.sin(t * 10) * 6, -90, -50 + Math.sin(t * 8) * 8); g.quadraticCurveTo(-64, -40, -40, -38); g.closePath(); g.fill();
      });
      drawLeg(-30, -32, legA(Math.PI) * .9, 16, 7, '#2b3245', '#59657f');
      drawLeg(26, -32, legA(Math.PI * .5) * .9, 16, 7, '#2b3245', '#59657f');
      const bg = g.createLinearGradient(0, -60, 0, -24);
      bg.addColorStop(0, '#7d8aa8'); bg.addColorStop(.5, '#3c4560'); bg.addColorStop(1, '#1e2333');
      g.fillStyle = bg;
      g.beginPath(); g.moveTo(40, -46); g.quadraticCurveTo(10, -60, -32, -54); g.quadraticCurveTo(-46, -50, -42, -36);
      g.quadraticCurveTo(-30, -24, 0, -26); g.lineTo(28, -26); g.quadraticCurveTo(42, -30, 40, -46); g.closePath(); g.fill();
      g.strokeStyle = 'rgba(51,240,255,.7)'; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(-26, -50); g.lineTo(-20, -30); g.moveTo(-6, -54); g.lineTo(-2, -28); g.moveTo(14, -54); g.lineTo(16, -28); g.stroke();
      withGlow(g, '#33f0ff', 14, () => { g.fillStyle = '#33f0ff'; g.beginPath(); g.arc(-8, -40, 4, 0, TAU); g.fill(); });
      // 脖子与头
      g.fillStyle = '#4a5574';
      g.beginPath(); g.moveTo(26, -50); g.lineTo(44, -72); g.lineTo(56, -66); g.lineTo(40, -40); g.closePath(); g.fill();
      g.fillStyle = '#5f6b8c';
      g.beginPath(); g.moveTo(40, -78); g.lineTo(52, -76); g.lineTo(72, -66); g.lineTo(74, -60); g.lineTo(58, -58); g.lineTo(46, -60); g.lineTo(38, -68); g.closePath(); g.fill();
      g.fillStyle = '#e6ecff'; g.beginPath(); g.moveTo(62, -59); g.lineTo(64, -54); g.lineTo(66, -59); g.fill(); g.beginPath(); g.moveTo(68, -60); g.lineTo(69, -55); g.lineTo(71, -61); g.fill();
      g.fillStyle = '#3c4560';
      g.beginPath(); g.moveTo(40, -76); g.lineTo(42, -92); g.lineTo(50, -78); g.closePath(); g.fill();
      g.beginPath(); g.moveTo(46, -77); g.lineTo(52, -90); g.lineTo(55, -75); g.closePath(); g.fill();
      withGlow(g, '#33f0ff', 12, () => { g.fillStyle = '#33f0ff'; g.beginPath(); g.moveTo(52, -71); g.lineTo(60, -69); g.lineTo(53, -67); g.closePath(); g.fill(); });
      drawLeg(-24, -32, legA(0) * .9, 16, 8, '#3c4560', '#7d8aa8');
      drawLeg(32, -32, legA(Math.PI * 1.5) * .9, 16, 8, '#3c4560', '#7d8aa8');
      saddle = [-4, -56];
    } else if (type === 'horse') {
      const flame = (fx, fy, s, n) => {
        for (let i = 0; i < n; i++) {
          const fx2 = fx - i * 6 * s, fy2 = fy + Math.sin(t * 18 + i) * 3;
          const fg = g.createLinearGradient(fx2, fy2, fx2 - 18 * s, fy2 - 10 * s);
          fg.addColorStop(0, '#fff3a0'); fg.addColorStop(.4, '#ffb020'); fg.addColorStop(1, 'rgba(255,60,0,0)');
          g.fillStyle = fg;
          g.beginPath(); g.moveTo(fx2, fy2); g.quadraticCurveTo(fx2 - 10 * s, fy2 - 16 * s - Math.random() * 6, fx2 - 22 * s, fy2 - 8 * s); g.quadraticCurveTo(fx2 - 10 * s, fy2 - 2, fx2, fy2 + 4); g.fill();
        }
      };
      g.save(); g.globalCompositeOperation = 'lighter'; flame(-40, -48, 1.4, 4); g.restore();
      drawLeg(-28, -34, legA(Math.PI), 17, 6, '#4a1210', '#ffb020');
      drawLeg(24, -34, legA(Math.PI * .5), 17, 6, '#4a1210', '#ffb020');
      const bg = g.createLinearGradient(0, -62, 0, -26);
      bg.addColorStop(0, '#a3311d'); bg.addColorStop(1, '#4a1210');
      g.fillStyle = bg;
      g.beginPath(); g.moveTo(36, -50); g.quadraticCurveTo(0, -62, -34, -54); g.quadraticCurveTo(-46, -46, -38, -32);
      g.quadraticCurveTo(-10, -26, 26, -30); g.quadraticCurveTo(40, -36, 36, -50); g.closePath(); g.fill();
      g.fillStyle = '#7a1d14';
      g.beginPath(); g.moveTo(24, -52); g.lineTo(44, -86); g.lineTo(56, -84); g.lineTo(40, -44); g.closePath(); g.fill();
      g.fillStyle = '#a3311d';
      g.beginPath(); g.moveTo(42, -92); g.lineTo(56, -90); g.lineTo(72, -76); g.lineTo(70, -70); g.lineTo(56, -72); g.lineTo(44, -80); g.closePath(); g.fill();
      g.fillStyle = '#d9b46a'; g.fillRect(44, -86, 18, 3);
      withGlow(g, '#ffb020', 10, () => { g.fillStyle = '#ffe066'; g.beginPath(); g.arc(56, -84, 2.2, 0, TAU); g.fill(); });
      g.fillStyle = '#7a1d14'; g.beginPath(); g.moveTo(46, -92); g.lineTo(48, -102); g.lineTo(52, -91); g.fill();
      g.save(); g.globalCompositeOperation = 'lighter'; flame(46, -88, .9, 5); g.restore();
      drawLeg(-22, -34, legA(0), 17, 7, '#6b1a12', '#ffb020');
      drawLeg(30, -34, legA(Math.PI * 1.5), 17, 7, '#6b1a12', '#ffb020');
      g.fillStyle = '#2a0a08'; rr(g, -16, -60, 26, 8, 4); g.fill();
      g.fillStyle = '#d9b46a'; g.fillRect(-16, -54, 26, 2);
      saddle = [-4, -60];
    } else { // drake
      g.translate(0, -16 + Math.sin(t * 4) * 4);
      const flap = Math.sin(t * 9);
      // 远侧翅膀
      g.fillStyle = 'rgba(80,40,140,.85)';
      g.beginPath(); g.moveTo(-6, -54); g.lineTo(-30, -88 - flap * 26); g.lineTo(-48, -70 - flap * 18); g.lineTo(-20, -48); g.closePath(); g.fill();
      // 尾巴
      g.strokeStyle = '#3d2a7a'; g.lineCap = 'round';
      g.lineWidth = 10; g.beginPath(); g.moveTo(-34, -42); g.quadraticCurveTo(-60, -40, -78, -54 + Math.sin(t * 5) * 8); g.stroke();
      g.lineWidth = 5; g.beginPath(); g.moveTo(-76, -52 + Math.sin(t * 5) * 8); g.lineTo(-96, -66 + Math.sin(t * 5 + 1) * 10); g.stroke();
      g.fillStyle = '#b26bff'; g.beginPath(); g.moveTo(-96, -66 + Math.sin(t * 5 + 1) * 10); g.lineTo(-108, -60); g.lineTo(-98, -76); g.closePath(); g.fill();
      // 身体
      const bg = g.createLinearGradient(0, -62, 0, -26);
      bg.addColorStop(0, '#6a4ad6'); bg.addColorStop(1, '#241650');
      g.fillStyle = bg;
      g.beginPath(); g.moveTo(34, -50); g.quadraticCurveTo(0, -64, -36, -50); g.quadraticCurveTo(-44, -36, -26, -30); g.quadraticCurveTo(10, -24, 30, -34); g.closePath(); g.fill();
      g.fillStyle = '#c9b8ff'; g.globalAlpha = .5;
      for (let i = 0; i < 5; i++) { g.beginPath(); g.arc(-22 + i * 11, -32, 4, 0, Math.PI); g.fill(); }
      g.globalAlpha = 1;
      g.fillStyle = '#2b1a5c';
      for (let i = 0; i < 6; i++) { g.beginPath(); g.moveTo(-28 + i * 10, -55 + Math.abs(i - 3)); g.lineTo(-24 + i * 10, -64 + Math.abs(i - 3)); g.lineTo(-20 + i * 10, -55 + Math.abs(i - 3)); g.fill(); }
      // 收起的腿
      limb(g, 20, -32, 1.2, 10, -.2, 10, 6, '#3d2a7a');
      limb(g, -20, -32, 1.4, 10, .2, 10, 6, '#3d2a7a');
      // 脖子与头
      g.strokeStyle = '#4b34a0'; g.lineWidth = 13;
      g.beginPath(); g.moveTo(26, -46); g.quadraticCurveTo(40, -62, 46, -76); g.stroke();
      g.fillStyle = '#5a3ec0';
      g.beginPath(); g.moveTo(38, -84); g.lineTo(54, -86); g.lineTo(76, -76); g.lineTo(74, -70); g.lineTo(54, -68); g.lineTo(40, -72); g.closePath(); g.fill();
      g.fillStyle = '#e8e0ff';
      g.beginPath(); g.moveTo(42, -84); g.quadraticCurveTo(34, -100, 24, -104); g.quadraticCurveTo(36, -94, 38, -82); g.fill();
      g.beginPath(); g.moveTo(48, -86); g.quadraticCurveTo(44, -102, 36, -108); g.quadraticCurveTo(48, -98, 52, -86); g.fill();
      withGlow(g, '#ffe45c', 12, () => { g.fillStyle = '#ffe45c'; g.beginPath(); g.moveTo(54, -80); g.lineTo(62, -78); g.lineTo(55, -76); g.closePath(); g.fill(); });
      // 近侧翅膀
      const wg = g.createLinearGradient(-10, -54, -40, -110);
      wg.addColorStop(0, '#8a5cff'); wg.addColorStop(1, '#d6b8ff');
      g.fillStyle = wg;
      g.beginPath(); g.moveTo(0, -54); g.lineTo(-22, -100 - flap * 30); g.lineTo(-40, -96 - flap * 26); g.lineTo(-30, -80 - flap * 18); g.lineTo(-50, -78 - flap * 16); g.lineTo(-18, -50); g.closePath(); g.fill();
      g.strokeStyle = 'rgba(255,255,255,.5)'; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(-2, -54); g.lineTo(-22, -100 - flap * 30); g.moveTo(-8, -52); g.lineTo(-40, -96 - flap * 26); g.stroke();
      // 闪电
      if (Math.random() < .35) {
        withGlow(g, '#b2a0ff', 14, () => {
          g.strokeStyle = '#e6deff'; g.lineWidth = 1.6;
          let lx = -10 + Math.random() * 40, ly = -40;
          g.beginPath(); g.moveTo(lx, ly);
          for (let i = 0; i < 4; i++) { lx += Math.random() * 14 - 7; ly += 8 + Math.random() * 6; g.lineTo(lx, ly); }
          g.stroke();
        });
      }
      saddle = [-4, -72 + Math.sin(t * 4) * 4];
    }
    g.restore();
    return [x + saddle[0], y + saddle[1]];
  }

  root.RunnerArt = { HEROES, PETS, MOUNTS, drawHero, drawAura, drawPet, drawMount };
})(this);
