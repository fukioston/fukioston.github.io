// 极速跑酷 3D：玩法逻辑沿用 2D 版（像素坐标），渲染换成 Three.js 侧视跟拍。
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { U, HEROES, PETS, MOUNTS, loadHeroAssets, Hero, makePet, animatePet, makeMount, animateMount, makeObstacle, makeItem, setGroundPx, disposeTree } from './runner3d-actors.js';
import { World, loadCityAssets } from './runner3d-world.js';

const $ = id => document.getElementById(id);
const FX = window.FX, CITY = window.RunnerCity;
const W = 960, H = 540, GROUND = 450, PX = 230; // 逻辑坐标（与 2D 版一致）
const GRAV = 2700, JUMP_V = -930, JUMP2_V = -840;
const STAND_H = 72, MOUNT_H = 118, SLIDE_H = 36, PW = 40; // 骑乘时站立更高，滑铲时一起趴低
const ZONE = 40000, LEVEL_SECS = 20, BONUS_SECS = 14;
const BEST_KEY = 'runner-best', LOADOUT_KEY = 'runner-loadout';
setGroundPx(GROUND);

const rand = (a, b) => a + Math.random() * (b - a);
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const toX = px => px * U, toY = py => (GROUND - py) * U;

let best = 0;
try { best = +localStorage.getItem(BEST_KEY) || 0; } catch {}
const loadout = { hero: 'flame', pet: 'dragon', mount: 'wolf' };
try { Object.assign(loadout, JSON.parse(localStorage.getItem(LOADOUT_KEY)) || {}); } catch {}
for (const [k, set] of [['hero', HEROES], ['pet', PETS], ['mount', MOUNTS]]) if (!set[loadout[k]]) loadout[k] = Object.keys(set)[0];

// ---------- 主题 ----------
const THEMES = [
  { name: '霓虹都市', sky: ['#05010f', '#2b0b4f', '#ff3f8e'], sun: ['#ffe45c', '#ff2bd6'], grid: '#ff2bd6', edge: '#33f0ff', accent: '#33f0ff',
    fog: '#2a0a45', hemi: '#6a3cff', aurora: 0, embers: 0, snow: 0, cityA: 1, rockA: 0, iceA: 0 },
  { name: '熔岩峡谷', sky: ['#0a0100', '#4a0b00', '#ff6a00'], sun: ['#fff2a8', '#ff3c00'], grid: '#ff4d00', edge: '#ffb000', accent: '#ffb000',
    fog: '#4a1000', hemi: '#ff6a2a', aurora: 0, embers: 1, snow: 0, cityA: 0, rockA: 1, iceA: 0 },
  { name: '极光冰原', sky: ['#010814', '#06264a', '#1aa6b7'], sun: ['#e8fbff', '#7af7ff'], grid: '#3ef0ff', edge: '#b2ff59', accent: '#b2ff59',
    fog: '#0a3050', hemi: '#7af7ff', aurora: 1, embers: 0, snow: 1, cityA: 0, rockA: 0, iceA: 1 },
];
const KIND = ['city', 'lava', 'ice'];
function hex(c) { const n = parseInt(c.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function mixc(a, b, t) { const x = hex(a), y = hex(b); return '#' + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join(''); }
function cityFor(zone) { const z = zone % 3 === 0 ? zone : zone + 3 - zone % 3; return CITY.ORDER[Math.floor(z / 3) % CITY.ORDER.length]; }
function themeAt(cam) {
  const z = Math.max(0, cam) / ZONE, i = Math.floor(z), f = z - i;
  const a = THEMES[i % 3], b = THEMES[(i + 1) % 3];
  const t = clamp((f - .9) / .1, 0, 1);
  const out = { name: a.name, zone: i, city: cityFor(i) };
  for (const k in a) {
    if (k === 'name') continue;
    out[k] = Array.isArray(a[k]) ? a[k].map((c, j) => mixc(c, b[k][j], t)) : typeof a[k] === 'number' ? a[k] + (b[k] - a[k]) * t : mixc(a[k], b[k], t);
  }
  return out;
}
const zoneKindAt = px => KIND[Math.floor(Math.max(0, px) / ZONE) % 3];

// ---------- 背景音乐 ----------
const Music = {
  ac: null, master: null, nextT: 0, step: 0, timer: null,
  start() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    if (!this.ac) { this.ac = new AC(); this.master = this.ac.createGain(); this.master.connect(this.ac.destination); }
    this.ac.resume(); this.nextT = this.ac.currentTime + .05; this.step = 0;
    clearInterval(this.timer); this.timer = setInterval(() => this.schedule(), 25);
  },
  stop() { clearInterval(this.timer); this.timer = null; if (this.ac) this.ac.suspend(); },
  resume() { if (!this.ac) return this.start(); this.ac.resume(); this.nextT = this.ac.currentTime + .05; clearInterval(this.timer); this.timer = setInterval(() => this.schedule(), 25); },
  schedule() {
    const a = this.ac;
    this.master.gain.value = FX.isMuted() ? 0 : .2;
    const bpm = S.bonus ? 150 : 108 + clamp((S.speed - 400) / 550, 0, 1) * 44, sd = 60 / bpm / 4;
    while (this.nextT < a.currentTime + .12) { this.play(this.step, this.nextT, sd); this.nextT += sd; this.step++; }
  },
  osc(type, f, t, d, vol, f2) {
    const a = this.ac, o = a.createOscillator(), g = a.createGain();
    o.type = type; o.frequency.setValueAtTime(f, t); if (f2) o.frequency.exponentialRampToValueAtTime(f2, t + d);
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(.0001, t + d);
    o.connect(g).connect(this.master); o.start(t); o.stop(t + d + .02);
  },
  noise(t, d, vol, type, freq) {
    const a = this.ac, len = Math.floor(a.sampleRate * d), buf = a.createBuffer(1, len, a.sampleRate), ch = buf.getChannelData(0);
    for (let i = 0; i < len; i++) ch[i] = Math.random() * 2 - 1;
    const s = a.createBufferSource(), f = a.createBiquadFilter(), g = a.createGain();
    s.buffer = buf; f.type = type; f.frequency.value = freq;
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(.0001, t + d);
    s.connect(f).connect(g).connect(this.master); s.start(t);
  },
  play(step, t, sd) {
    const s = step % 16, bar = Math.floor(step / 16) % 4;
    const roots = S.bonus ? [60, 65, 67, 64] : [57, 53, 48, 55];
    const midi = n => 440 * Math.pow(2, (n - 69) / 12);
    if (s % 4 === 0) this.osc('sine', 160, t, .28, .9, 40);
    if (s === 4 || s === 12) this.noise(t, .16, .35, 'bandpass', 1800);
    if (s % 2 === 1) this.noise(t, .04, .12, 'highpass', 7000);
    if (s % 2 === 0) {
      const a = this.ac, o = a.createOscillator(), f = a.createBiquadFilter(), g = a.createGain();
      o.type = 'sawtooth'; o.frequency.value = midi(roots[bar] - 12 + (s % 8 === 6 ? 12 : 0));
      f.type = 'lowpass'; f.frequency.setValueAtTime(1400, t); f.frequency.exponentialRampToValueAtTime(200, t + sd * 2);
      g.gain.setValueAtTime(.32, t); g.gain.exponentialRampToValueAtTime(.0001, t + sd * 1.9);
      o.connect(f).connect(g).connect(this.master); o.start(t); o.stop(t + sd * 2);
    }
    const chord = S.bonus ? [0, 4, 7, 12, 16, 12, 7, 4] : [0, 3, 7, 12, 7, 3, 0, 7];
    const hype = S.feverT > 0 || S.bonus;
    if (S.level >= 2 || hype) this.osc('square', midi(roots[bar] + 12 + chord[s % 8]), t, sd * .9, hype ? .07 : .035);
    if (hype && s % 4 === 2) this.osc('triangle', midi(roots[bar] + 24 + chord[(s + 3) % 8]), t, sd * 2, .06);
  },
};

// ---------- 渲染器与场景 ----------
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = .9;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x3a0f52, 16, 95);
const camera = new THREE.PerspectiveCamera(36, 16 / 9, .1, 1200);
const hemi = new THREE.HemisphereLight(0x6a3cff, 0x0a0410, .55); scene.add(hemi);
const key = new THREE.DirectionalLight(0xd8c8ff, 1.7);
key.castShadow = true; key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -10, right: 14, top: 10, bottom: -4, near: 1, far: 60 });
key.shadow.bias = -.0004; key.shadow.normalBias = .02;
scene.add(key, key.target);
const rim = new THREE.DirectionalLight(0x33f0ff, 3.2); scene.add(rim, rim.target);   // 背光勾边
const fill = new THREE.DirectionalLight(0xff2bd6, .8); scene.add(fill, fill.target);

// 霓虹环境反射：金属盔甲、金币会映出粉青色灯光
{
  const env = new THREE.Scene();
  env.background = new THREE.Color(0x07040e);
  const panel = (c, i, x, y, z, w, h) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(c).multiplyScalar(i), side: THREE.DoubleSide })); m.position.set(x, y, z); m.lookAt(0, 0, 0); env.add(m); };
  panel('#ff2bd6', 3, -6, 2, 4, 6, 3); panel('#33f0ff', 3, 6, 3, -3, 6, 3); panel('#ffffff', 2.2, 0, 8, 0, 8, 8);
  panel('#b26bff', 2, 0, 1, -7, 10, 2); panel('#ffb000', 2, 5, 1, 6, 4, 2);
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(env, .04).texture;
}
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1280, 720), .7, .5, .88);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const r = stage.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  composer.setSize(r.width, r.height);
  camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

// ---------- 粒子（加法混合的点精灵） ----------
const PMAX = 2500;
const pGeo = new THREE.BufferGeometry();
const pPos = new Float32Array(PMAX * 3), pCol = new Float32Array(PMAX * 3), pSize = new Float32Array(PMAX);
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
pGeo.setAttribute('size', new THREE.BufferAttribute(pSize, 1));
const pMat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true,
  uniforms: { scale: { value: 600 } },
  vertexShader: `attribute float size; varying vec3 vC; uniform float scale;
    void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position,1.); gl_PointSize = size * scale / -mv.z; gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `varying vec3 vC; void main(){ float d = length(gl_PointCoord - .5); if (d > .5) discard; float a = smoothstep(.5, 0., d); gl_FragColor = vec4(vC * a, a); }`,
});
const points = new THREE.Points(pGeo, pMat); points.frustumCulled = false; scene.add(points);
const parts = [];
function addPart(x, y, z, vx, vy, vz, life, size, color, grav = 6) {
  if (parts.length >= PMAX) parts.shift();
  const c = new THREE.Color(color);
  parts.push({ x, y, z, vx, vy, vz, life, t: 0, size, r: c.r, g: c.g, b: c.b, grav });
}
// 光环
const rings = [];
function ring(xs, yPx, color) {
  const m = new THREE.Mesh(new THREE.TorusGeometry(.3, .04, 8, 40), new THREE.MeshBasicMaterial({ color, transparent: true, toneMapped: false, blending: THREE.AdditiveBlending, depthWrite: false }));
  m.position.set(toX(S.cam + xs), toY(yPx), .3);
  scene.add(m); rings.push({ m, t: 0 });
}

// ---------- 状态 ----------
let S, state = 'loading', last = 0, slowmo = 1, dyingT = 0, countT = 0, lastCount = 4;
const shake = { t: 0, amp: 0 };
let hero, world, pets = {}, mounts = {}, coinsIM, gemsIM;
const obsMesh = new Map(), itemMesh = new Map();

function reset() {
  S = {
    t: 0, rt: 0, cam: 0, speed: 400, score: 0, coins: 0, perfect: 0, combo: 0, maxCombo: 0, comboT: 0, level: 1, zone: 0,
    fever: 0, feverT: 0, bonusCoins: 0, bonusNeed: 120, bonus: null, mount: { type: loadout.mount },
    p: { y: GROUND, vy: 0, onGround: true, jumps: 0, sliding: false, slideT: 0, slideHeld: false, invuln: 0, runT: 0, dead: false, sq: 1, flip: 0, crouch: 0 },
    pet: new THREE.Vector3(),
    ground: [{ x1: -2000, x2: 1500 }], platforms: [], obstacles: [], coins_: [], items: [], nextX: 1400,
    power: { magnet: 0, boost: 0, double: 0, shield: false },
  };
  for (const [, m] of obsMesh) { scene.remove(m); disposeTree(m); }
  obsMesh.clear();
  for (const [, m] of itemMesh) { scene.remove(m); disposeTree(m); }
  itemMesh.clear();
  parts.length = 0;
}

// ---------- 关卡生成（与 2D 版一致） ----------
function extendGround(x) { const g = S.ground[S.ground.length - 1]; if (g.x2 < x) g.x2 = x; }
function coinLine(x, y, n, gap = 34) { for (let i = 0; i < n; i++) S.coins_.push({ x: x + i * gap, y, value: 10 }); }
function coinArc(x, w, h, y0 = GROUND - 40, n = 7) { for (let i = 0; i < n; i++) { const t = i / (n - 1); S.coins_.push({ x: x + t * w, y: y0 - Math.sin(t * Math.PI) * h, value: 10 }); } }
const PATTERNS = {
  crate(x) { const h = pick([40, 44]); S.obstacles.push({ type: 'crate', x, y: GROUND - h, w: 46, h }); coinArc(x - 70, 186, 120); return 60; },
  crateDouble(x) { S.obstacles.push({ type: 'crate', x, y: GROUND - 44, w: 46, h: 44 }, { type: 'crate', x, y: GROUND - 88, w: 46, h: 44 }); coinArc(x - 80, 206, 170); return 60; },
  spikes(x) { const w = pick([60, 96]); S.obstacles.push({ type: 'spikes', x, y: GROUND - 28, w, h: 28 }); coinArc(x - 60, w + 120, 110); return w; },
  bar(x) { S.obstacles.push({ type: 'bar', x, y: GROUND - 150, w: 80, h: 100 }); coinLine(x - 20, GROUND - 18, 4, 36); return 90; },
  bird(x) {
    const low = Math.random() < .5;
    S.obstacles.push({ type: 'bird', x, y: low ? GROUND - 88 : GROUND - 150, w: 46, h: 30, vx: -rand(60, 140), flap: rand(0, 6), low });
    if (low) coinLine(x, GROUND - 18, 4, 36); else coinLine(x - 40, GROUND - 40, 5);
    return 60;
  },
  pit(x) { const w = rand(120, 150 + S.speed * .12); extendGround(x); S.ground.push({ x1: x + w, x2: x + w + 10 }); coinArc(x - 40, w + 80, 130); return w; },
  platforms(x) {
    let px = x; const n = pick([2, 3]);
    for (let i = 0; i < n; i++) {
      const y = GROUND - rand(110, 170), w = rand(140, 220);
      S.platforms.push({ x: px, y, w }); coinLine(px + 18, y - 26, Math.floor(w / 40), 36);
      if (i === n - 1 && Math.random() < .5) S.obstacles.push({ type: 'crate', x: px + w * .5, y: GROUND - 44, w: 46, h: 44 });
      px += w + rand(70, 130);
    }
    return px - x;
  },
  pitPlatform(x) {
    const w = rand(260, 340); extendGround(x); S.ground.push({ x1: x + w, x2: x + w + 10 });
    const pw = 120; S.platforms.push({ x: x + (w - pw) / 2, y: GROUND - 90, w: pw }); coinLine(x + (w - pw) / 2 + 10, GROUND - 116, 3, 36);
    return w;
  },
  coins(x) { coinLine(x, GROUND - 40, 8); return 8 * 34; },
  combo(x) {
    S.obstacles.push({ type: 'crate', x, y: GROUND - 44, w: 46, h: 44 });
    const gap = S.speed * 1.15;
    S.obstacles.push({ type: 'bar', x: x + gap, y: GROUND - 150, w: 80, h: 100 });
    coinArc(x - 70, 186, 120);
    return gap + 90;
  },
};
function spawn() {
  if (S.bonus) return spawnBonus();
  while (S.nextX < S.cam + W + 500) {
    const x = S.nextX, d = Math.min(1, S.cam / 30000);
    extendGround(x);
    const table = [['crate', 3], ['spikes', 2.5], ['bar', 2.5], ['bird', 1 + d * 2], ['pit', 2 + d], ['platforms', 1.5],
      ['coins', 1.6 - d], ['crateDouble', .5 + d * 2], ['pitPlatform', d * 2], ['combo', d * 2.5]];
    let r = Math.random() * table.reduce((s, t) => s + t[1], 0), name = 'crate';
    for (const [n, w] of table) { if ((r -= w) <= 0) { name = n; break; } }
    const len = PATTERNS[name](x);
    if (Math.random() < .16) S.items.push({ type: pick(['magnet', 'shield', 'boost', 'double']), x: x + len + 140, y: GROUND - rand(70, 150) });
    const end = x + len;
    extendGround(end + 10);
    S.nextX = end + S.speed * rand(.75, 1.25) + 120;
    extendGround(S.nextX);
  }
  cleanup();
}
function cleanup() {
  const cut = S.cam - 300;
  S.obstacles = S.obstacles.filter(o => o.x + o.w > cut);
  S.coins_ = S.coins_.filter(c => c.x > cut && !c.taken);
  S.items = S.items.filter(i => i.x > cut && !i.taken);
  S.platforms = S.platforms.filter(p => p.x + p.w > cut);
  while (S.ground.length > 2 && S.ground[0].x2 < cut) S.ground.shift();
}

// ---------- 超级奖励关 ----------
const GEMS = [
  { rgb: [255, 207, 51], color: '#ffcf33', value: 10, gem: false },
  { rgb: [245, 245, 250], color: '#f2f6ff', value: 30, gem: true },
  { rgb: [40, 32, 50], color: '#3a2a6a', value: 20, gem: true },
  { rgb: [235, 50, 70], color: '#ff3b5c', value: 20, gem: true },
  { rgb: [255, 150, 200], color: '#ff8fd0', value: 20, gem: true },
  { rgb: [60, 150, 255], color: '#3ea0ff', value: 20, gem: true },
  { rgb: [70, 200, 90], color: '#3ee06b', value: 20, gem: true },
  { rgb: [255, 150, 40], color: '#ff9a2e', value: 15, gem: true },
  { rgb: [150, 95, 55], color: '#c27a45', value: 15, gem: true },
];
const BONUS_SHAPES = ['🐼', '❤️', '福', '🐱', '⭐', '🦊', '666', '🍉', '🐷', '发', '🦄', '😎', '🐯', 'GO', '🍩', '🐸'];
const shapeCache = {};
function sampleShape(str) {
  if (shapeCache[str]) return shapeCache[str];
  const text = /^[\x20-\x7e一-鿿]+$/.test(str);
  const size = 72, cw = text ? Math.round(size * Math.max(1, str.length * .62)) : size;
  const c = document.createElement('canvas'); c.width = cw; c.height = size;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.font = text ? `900 ${str.length > 1 ? 58 : 64}px "PingFang SC", "Heiti SC", sans-serif` : '60px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
  g.fillStyle = '#ffcf33'; g.fillText(str, cw / 2, size / 2 + 3);
  const data = g.getImageData(0, 0, cw, size).data;
  const rows = 15, cell = size / rows, cols = Math.round(cw / cell), cells = [];
  const pal = text ? [GEMS[0]] : GEMS;
  for (let r = 0; r < rows; r++) for (let col = 0; col < cols; col++) {
    const votes = new Map(); let opaque = 0, n = 0;
    for (let y = Math.floor(r * cell); y < Math.floor((r + 1) * cell); y++) for (let x = Math.floor(col * cell); x < Math.floor((col + 1) * cell); x++) {
      const i = (y * cw + x) * 4; n++;
      if (data[i + 3] < 128) continue;
      opaque++;
      let bg = pal[0], bd = 1e9;
      for (const gm of pal) { const d = (gm.rgb[0] - data[i]) ** 2 + (gm.rgb[1] - data[i + 1]) ** 2 + (gm.rgb[2] - data[i + 2]) ** 2; if (d < bd) { bd = d; bg = gm; } }
      votes.set(bg, (votes.get(bg) || 0) + 1);
    }
    if (opaque / n < .45) continue;
    let gem = pal[0], most = -1;
    for (const [gm, v] of votes) if (v > most) { most = v; gem = gm; }
    cells.push({ r, c: col, gem });
  }
  return (shapeCache[str] = { cells, cols, rows });
}
function spawnBonus() {
  const B = S.bonus;
  while (B.next < S.cam + W + 400) {
    let shape = sampleShape(B.order[B.idx % B.order.length]);
    if (shape.cells.length < 20) shape = sampleShape('福');
    B.idx++;
    const cell = 21, top = 108 + (15 - shape.rows) * cell / 2;
    for (const c of shape.cells) S.coins_.push({ x: B.next + c.c * cell, y: top + c.r * cell, value: c.gem.value, gem: c.gem.gem ? c.gem.color : null });
    const w = shape.cols * cell;
    coinArc(B.next + w + 40, 200, 90, GROUND - 40, 9);
    B.next += w + 320;
  }
  cleanup();
}
function enterBonus() {
  S.bonus = { t: BONUS_SECS, gained: 0, next: S.cam + W + 60, idx: 0, order: BONUS_SHAPES.slice().sort(() => Math.random() - .5) };
  S.obstacles = []; S.items = []; S.platforms = [];
  S.coins_ = S.coins_.filter(c => c.x < S.cam + PX);
  S.ground = [{ x1: S.cam - 2000, x2: 1e12 }];
  S.p.sliding = false;
  flash('#ffffff', .95); doShake(12, .5);
  banner('超级奖励', '金币雨来啦！连点跳跃可以飞', '#ffe45c');
  FX.sound('mate');
}
function exitBonus() {
  const gained = S.bonus.gained;
  S.bonus = null; S.coins_ = [];
  S.ground = [{ x1: S.cam - 600, x2: S.cam + W + 700 }];
  S.nextX = S.cam + W + 700;
  S.p.invuln = 1.6; S.bonusCoins = 0; S.bonusNeed += 60;
  flash('#33f0ff', .6);
  banner('奖励结束', `本轮收获 ${gained} 枚`, '#33f0ff');
  FX.sound('power');
}

// ---------- 输入 ----------
function jump() {
  if (state === 'over') return start();
  if (state !== 'play') return;
  const p = S.p;
  if (S.power.boost) return;
  if (S.bonus) { p.vy = -660; p.onGround = false; FX.sound('jump'); dust(PX, p.y, 6, '#ffe45c'); return; }
  const maxJumps = 2;
  if (p.onGround || p.jumps < maxJumps) {
    const first = p.onGround;
    p.vy = first ? JUMP_V : JUMP2_V;
    p.onGround = false; p.jumps = first ? 1 : p.jumps + 1; p.sliding = false; p.sq = 1.25;
    if (!first) { if (!S.mount) p.flip = 1; ring(PX, p.y - 50, themeAt(S.cam).accent); }
    p.justJumped = true;
    FX.sound('jump');
    dust(PX, p.y, first ? 10 : 18, first ? '#ffffff' : themeAt(S.cam).accent);
  }
}
function slideDown(on) {
  if (state !== 'play') return;
  const p = S.p;
  p.slideHeld = on;
  if (!on) return;
  if (S.power.boost || S.bonus) { if (!p.onGround) p.vy = Math.max(p.vy, 1400); return; }
  if (p.onGround) { if (!p.sliding) { p.sliding = true; p.slideT = 0; sparks(PX + 10, p.y, 14); } }
  else p.vy = Math.max(p.vy, 1400);
}
function togglePause() {
  if (state === 'play') { state = 'pause'; $('pauseOv').hidden = false; Music.stop(); }
  else if (state === 'pause') { state = 'play'; $('pauseOv').hidden = true; last = performance.now(); Music.resume(); }
}
addEventListener('keydown', e => {
  if (e.repeat && !['ArrowDown', 's', 'S'].includes(e.key)) return;
  if ([' ', 'ArrowUp', 'w', 'W'].includes(e.key)) { e.preventDefault(); if (state === 'menu') start(); else jump(); }
  else if (['ArrowDown', 's', 'S'].includes(e.key)) { e.preventDefault(); slideDown(true); }
  else if (['p', 'P', 'Escape'].includes(e.key)) togglePause();
});
addEventListener('keyup', e => { if (['ArrowDown', 's', 'S'].includes(e.key)) slideDown(false); });
let touchY = null;
stage.addEventListener('touchstart', e => { if (e.target.closest('button, a, .overlay')) return; touchY = e.touches[0].clientY; e.preventDefault(); }, { passive: false });
stage.addEventListener('touchend', e => {
  if (touchY === null) return;
  const dy = e.changedTouches[0].clientY - touchY; touchY = null;
  if (dy > 30) { slideDown(true); setTimeout(() => slideDown(false), 450); } else jump();
});
stage.addEventListener('mousedown', e => { if (!e.target.closest('button, a, .overlay') && state === 'play') jump(); });
$('padJump').addEventListener('touchstart', e => { e.preventDefault(); jump(); });
$('padSlide').addEventListener('touchstart', e => { e.preventDefault(); slideDown(true); });
$('padSlide').addEventListener('touchend', () => slideDown(false));
$('btnStart').onclick = start; $('btnRestart').onclick = start; $('btnResume').onclick = togglePause; $('btnQuit').onclick = start;
$('btnMenu').onclick = () => toMenu();
document.addEventListener('visibilitychange', () => { if (document.hidden && state === 'play') togglePause(); });

function toMenu() {
  state = 'menu'; reset();
  for (const id of ['overOv', 'pauseOv']) $(id).hidden = true;
  $('menuOv').hidden = false; $('hud').hidden = true;
  Music.stop();
  showBest();
}
function start() {
  reset();
  state = 'count'; countT = 3; slowmo = 1; lastCount = 4;
  for (const id of ['menuOv', 'overOv', 'pauseOv']) $(id).hidden = true;
  $('hud').hidden = false;
  last = performance.now();
  Music.stop();
  hero.play('Idle_swordRight', { fade: .2 });
}

// ---------- 特效 ----------
function dust(xs, yPx, n, color) {
  const x = toX(S.cam + xs), y = toY(yPx);
  const c = new THREE.Color(color).multiplyScalar(.45);
  for (let i = 0; i < n; i++) addPart(x, y + .05, rand(-.6, .6), rand(-4, 1), rand(.6, 3.2), rand(-1, 1), rand(.25, .5), rand(.07, .16), c, 5);
}
function sparks(xs, yPx, n, color = '#ffd36b') {
  const x = toX(S.cam + xs), y = toY(yPx);
  for (let i = 0; i < n; i++) addPart(x, y + .05, rand(-.4, .4), rand(-9, -4), rand(.5, 2.5), rand(-1.5, 1.5), rand(.2, .45), rand(.06, .12), color, 8);
}
function burst(xs, yPx, n, colors, power = 320) {
  const x = toX(S.cam + xs), y = toY(yPx), v = power * U;
  for (let i = 0; i < n; i++) {
    const a = rand(0, Math.PI * 2), b = rand(-1, 1), s = rand(.25, 1) * v;
    addPart(x, y, .2, Math.cos(a) * s, Math.sin(a) * s, b * s * .6, rand(.35, .9), rand(.1, .24), pick(colors), 3);
  }
}
function debris(xs, yPx, color) {
  const x = toX(S.cam + xs), y = toY(yPx);
  for (let i = 0; i < 14; i++) addPart(x, y, rand(-.5, .5), rand(-3, 6), rand(3, 9), rand(-3, 3), rand(.6, 1.1), rand(.18, .32), color, 22);
}
function popText(text, xs, yPx, color = '#fff') {
  const el = document.createElement('div');
  el.className = 'pop'; el.textContent = text; el.style.color = color;
  el.dataset.x = toX(S.cam + xs); el.dataset.y = toY(yPx);
  $('pops').appendChild(el);
  placePop(el);
  setTimeout(() => el.remove(), 1100);
}
const vtmp = new THREE.Vector3();
function placePop(el) {
  vtmp.set(+el.dataset.x, +el.dataset.y, 0).project(camera);
  el.style.left = ((vtmp.x + 1) / 2 * 100) + '%'; el.style.top = ((1 - vtmp.y) / 2 * 100) + '%';
}
function banner(text, sub, color) {
  document.querySelectorAll('.rb-banner').forEach(b => b.remove());
  const el = document.createElement('div');
  el.className = 'rb-banner';
  el.innerHTML = `<div class="t" style="color:${color};text-shadow:0 0 30px ${color},0 0 60px ${color}">${text}</div>` + (sub ? `<div class="s">${sub}</div>` : '');
  $('pops').appendChild(el);
  setTimeout(() => el.remove(), 1900);
}
function countBig(text) {
  const el = document.createElement('div'); el.className = 'count'; el.textContent = text;
  $('pops').appendChild(el); setTimeout(() => el.remove(), 1000);
}
function doShake(amp, t = .35) { if (shake.t <= 0 || amp >= shake.amp) { shake.amp = amp; shake.t = t; } }
function flash(color, a = .7) {
  const f = $('flash');
  f.style.transition = 'none'; f.style.background = color; f.style.opacity = a;
  requestAnimationFrame(() => { f.style.transition = 'opacity .5s ease-out'; f.style.opacity = 0; });
}

// ---------- 物理与碰撞 ----------
const onGroundAt = wx => S.ground.some(g => wx >= g.x1 - 6 && wx <= g.x2 + 6);
const standH = () => (S.mount ? MOUNT_H : STAND_H);
function playerBox() { const p = S.p, h = p.sliding ? SLIDE_H : standH(); return { x: S.cam + PX - PW / 2 + 6, y: p.y - h + 6, w: PW - 12, h: h - 8 }; }
const hit = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

function die(reason) {
  const p = S.p;
  p.dead = true; p.vy = -600;
  state = 'dying'; dyingT = 1.3; slowmo = .25;
  S.deathReason = reason;
  FX.sound('hit');
  doShake(1.1, .7);
  flash('#ff2244', .7);
  burst(PX, p.y - 30, 60, ['#ff2244', '#ffd36b', '#ffffff', themeAt(S.cam).accent], 420);
  Music.stop();
  hero.play('Death', { once: true, fade: .1 });
}
function finish() {
  state = 'over';
  const score = Math.floor(S.score), isNew = score > best;
  if (isNew) { best = score; try { localStorage.setItem(BEST_KEY, String(best)); } catch {} }
  const grade = score >= 60000 ? ['S', '#ffe45c'] : score >= 30000 ? ['A', '#ff2bd6'] : score >= 12000 ? ['B', '#33f0ff'] : score >= 4000 ? ['C', '#b2ff59'] : ['D', '#aaaaaa'];
  $('overReason').textContent = S.deathReason;
  const g = $('overGrade'); g.textContent = grade[0]; g.style.color = grade[1]; g.style.textShadow = `0 0 30px ${grade[1]}, 0 0 60px ${grade[1]}`;
  g.style.animation = 'none'; void g.offsetWidth; g.style.animation = '';
  $('overDist').textContent = Math.floor(S.cam / 50) + ' 米';
  $('overCoins').textContent = S.coins; $('overPerfect').textContent = S.perfect; $('overCombo').textContent = S.maxCombo; $('overBest').textContent = best;
  $('overNew').hidden = !isNew; $('overOv').hidden = false; $('hud').hidden = true;
  const t0 = performance.now();
  const roll = now => { const k = Math.min(1, (now - t0) / 900); $('overScore').textContent = Math.floor(score * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(roll); };
  requestAnimationFrame(roll);
  FX.sound(isNew ? 'mate' : 'big');
  showBest();
}
function gainFever(v) {
  if (S.feverT > 0 || S.bonus) return;
  S.fever = Math.min(1, S.fever + v);
  if (S.fever >= 1) { S.fever = 0; S.feverT = 7; banner('狂 热', '金币 ×3 · 自动吸金', '#ff2bd6'); flash('#ff2bd6', .45); doShake(.4, .4); FX.sound('power'); }
}
function collectCoin(c) {
  c.taken = true; S.coins++;
  S.combo++; S.comboT = 1.2; S.maxCombo = Math.max(S.maxCombo, S.combo);
  S.score += c.value * (S.power.double ? 2 : 1) * (S.feverT > 0 ? 3 : 1) * (S.bonus ? 2 : 1);
  S.scoreBump = 1;
  if (S.bonus) S.bonus.gained++; else S.bonusCoins++;
  gainFever(.025);
  burst(c.x - S.cam, c.y, c.gem ? 8 : 5, c.gem ? [c.gem, '#ffffff'] : ['#ffd36b', '#fff4b0'], 160);
  FX.sound('coin');
  if (S.combo % 10 === 0) popText(`连击 ×${S.combo}`, PX + 40, S.p.y - 130, '#ffd36b');
  if (!S.bonus && S.bonusCoins >= S.bonusNeed) enterBonus();
}

function update(dt) {
  S.t += dt;
  const p = S.p, pw = S.power;
  if (!S.bonus) S.rt += dt;
  for (const k of ['magnet', 'boost', 'double']) if (pw[k] > 0) pw[k] = Math.max(0, pw[k] - dt);
  if (p.invuln > 0) p.invuln -= dt;
  if (S.feverT > 0) S.feverT = Math.max(0, S.feverT - dt);
  if (S.bonus && (S.bonus.t -= dt) <= 0) exitBonus();

  const lvl = 1 + Math.floor(S.rt / LEVEL_SECS);
  if (lvl > S.level) { S.level = lvl; banner(`LV.${lvl}`, '速度提升！', '#33f0ff'); flash('#33f0ff', .3); FX.sound('big'); }
  const zone = Math.floor(S.cam / ZONE);
  if (zone > S.zone && !S.bonus) {
    S.zone = zone;
    const th = THEMES[zone % 3];
    banner(zone % 3 === 0 ? cityFor(zone) : th.name, `${th.name} · ${zone * 800} 米`, th.accent);
    flash(th.accent, .45); FX.sound('power');
  }

  S.speed = S.bonus ? 560 : Math.min(950, 400 + S.rt * 9);
  const speed = S.speed * (pw.boost ? 1.75 : 1);
  S.cam += speed * dt;
  const mult = pw.double ? 2 : 1;
  S.score += speed * dt / 12 * mult;
  spawn();

  const wx = S.cam + PX;
  if (pw.boost) {
    p.y += (GROUND - 170 - p.y) * Math.min(1, dt * 6);
    p.vy = 0; p.onGround = false; p.sliding = false; p.jumps = 1;
    if (pw.boost <= dt * 1.5) { p.invuln = 1.2; p.vy = 0; }
    if (Math.random() < .7) addPart(toX(wx) - .5, toY(p.y) + rand(.4, 1.2), rand(-.3, .3), -8, rand(-.3, .3), 0, .35, rand(.15, .3), pick(['#ffd36b', '#ff7b3a', '#ffffff']), 0);
  } else {
    const prevY = p.y;
    const grav = S.bonus ? 1600 : GRAV;
    p.vy += grav * dt; p.y += p.vy * dt;
    if (S.bonus && p.y < 170) { p.y = 170; p.vy = Math.max(0, p.vy); }
    const wasGround = p.onGround;
    p.onGround = false;
    for (const pl of S.platforms) if (p.vy >= 0 && wx > pl.x - 8 && wx < pl.x + pl.w + 8 && prevY <= pl.y + 1 && p.y >= pl.y) { p.y = pl.y; p.vy = 0; p.onGround = true; }
    if (!p.onGround && p.y >= GROUND && prevY <= GROUND + 4 && onGroundAt(wx)) { p.y = GROUND; p.vy = 0; p.onGround = true; }
    if (p.onGround && !wasGround) { p.sq = .75; dust(PX, p.y, S.mount ? 18 : 9, '#ffffff'); if (S.mount) doShake(.15, .15); p.jumps = 0; }
    if (p.y > H + 120) {
      if (pw.shield || p.invuln > 0) {
        if (p.invuln <= 0) pw.shield = false;
        p.y = GROUND - 200; p.vy = JUMP_V * .6; p.invuln = 1.5; pw.boost = 1.2;
        popText('救援成功！', PX, GROUND - 230, '#7fd8ff'); FX.sound('power'); doShake(.3);
      } else return die('掉进了深坑');
    }
  }
  p.sq += (1 - p.sq) * Math.min(1, dt * 10);
  if (p.flip > 0) p.flip = Math.max(0, p.flip - dt * 2.3);
  if (p.sliding) {
    p.slideT += dt;
    if (Math.random() < .6) sparks(PX + 14, p.y, 1);
    if (!p.onGround) p.sliding = false;
    else if (!p.slideHeld && p.slideT > .5) {
      const standBox = { x: wx - PW / 2 + 6, y: p.y - standH() + 6, w: PW - 12, h: standH() - 8 };
      if (!S.obstacles.some(o => !o.dead && (o.type === 'bar' || o.type === 'bird') && hit(standBox, o))) p.sliding = false;
    }
  }
  p.runT += dt * speed / 60;

  const box = playerBox(), th = themeAt(S.cam);
  for (const o of S.obstacles) {
    if (o.type === 'bird') { o.x += o.vx * dt; o.flap += dt * 40; }
    if (o.dead) continue;
    if (box.x < o.x + o.w && box.x + box.w > o.x) {
      const gap = o.y >= box.y + box.h ? o.y - (box.y + box.h) : box.y - (o.y + o.h);
      o.minGap = Math.min(o.minGap ?? 999, gap);
    }
    if (hit(box, o)) {
      const cx = o.x - S.cam + o.w / 2, cy = o.y + o.h / 2;
      if (pw.boost || p.invuln > 0) {
        o.dead = true; burst(cx, cy, 26, ['#ffd36b', '#ffffff', th.accent]); debris(cx, cy, '#ffcc00');
        FX.sound('capture'); S.score += 30; doShake(.25, .2); popText('粉碎 +30', cx, cy - 30, '#ffd36b'); continue;
      }
      if (pw.shield) {
        pw.shield = false; o.dead = true; p.invuln = 1.2;
        burst(cx, cy, 34, ['#7fd8ff', '#ffffff']); debris(cx, cy, '#7fd8ff');
        popText('护盾抵挡！', PX, p.y - 110, '#7fd8ff'); FX.sound('power'); doShake(.4); flash('#7fd8ff', .35);
        continue;
      }
      return die({ crate: '撞上了警示箱', spikes: '踩到了尖刺', bar: '撞上了激光栅栏', bird: '被无人机撞翻' }[o.type]);
    }
    if (!o.passed && o.x + o.w < box.x) {
      o.passed = true;
      if (o.minGap !== undefined && o.minGap >= 0 && o.minGap < 16) {
        S.perfect++; S.score += 50 * mult; gainFever(.15);
        popText('完美闪避 +50', PX + 30, p.y - 100, '#b2ff59'); ring(PX, p.y - 36, '#b2ff59'); FX.sound('big');
      }
    }
  }

  // 金币：英雄或宠物都能吸
  const center = { x: wx, y: p.y - (p.sliding ? SLIDE_H : standH()) / 2 };
  const petW = { x: S.pet.x / U, y: GROUND - S.pet.y / U };
  const magnet = pw.magnet || S.feverT > 0;
  for (const c of S.coins_) {
    if (c.taken) continue;
    const dx = center.x - c.x, dy = center.y - c.y, d = Math.hypot(dx, dy);
    const pdx = petW.x - c.x, pdy = petW.y - c.y, pd = Math.hypot(pdx, pdy);
    if (magnet && d < 280) { c.x += dx / d * 1000 * dt; c.y += dy / d * 1000 * dt; }
    else if (pd < 70) { c.x += pdx / (pd || 1) * 700 * dt; c.y += pdy / (pd || 1) * 700 * dt; }
    if (d < (S.mount && !p.sliding ? 44 : 34) || pd < 18) collectCoin(c);
  }
  if (S.comboT > 0 && (S.comboT -= dt) <= 0) S.combo = 0;
  for (const it of S.items) {
    if (it.taken || Math.hypot(center.x - it.x, center.y - it.y) >= 46) continue;
    it.taken = true;
    FX.sound('power');
    burst(it.x - S.cam, it.y, 36, ['#ffffff', '#7fd8ff', '#ffd36b', th.accent], 380);
    ring(it.x - S.cam, it.y, '#ffffff');
    popText({ magnet: '磁铁吸金！', shield: '能量护盾！', boost: '超级冲刺！', double: '双倍得分！' }[it.type], PX + 20, p.y - 120, '#ffffff');
    if (it.type === 'shield') pw.shield = true; else pw[it.type] = { magnet: 8, boost: 3.5, double: 10 }[it.type];
    if (it.type === 'boost') { doShake(.35, .4); flash('#ff7b3a', .4); }
  }
  if (p.onGround && !p.sliding && Math.random() < speed / (S.mount ? 1500 : 6000)) dust(PX - 10, p.y, 1, '#ffffff');
}

// ---------- 3D 同步 ----------
const coinGeo = new THREE.CylinderGeometry(.21, .21, .06, 28); coinGeo.rotateX(Math.PI / 2);
const coinMat = new THREE.MeshStandardMaterial({ color: 0xffd23f, metalness: .85, roughness: .18, emissive: 0xffa000, emissiveIntensity: .7 });
const gemGeo = new THREE.OctahedronGeometry(.24, 0);
const gemMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: .25, roughness: .08, emissive: 0x222222, emissiveIntensity: 1, flatShading: true });
const dummy = new THREE.Object3D(), tmpColor = new THREE.Color();
const trailN = 22, trailPts = [];
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(trailN * 2 * 3), 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(trailN * 2 * 3), 3));
const idx = []; for (let i = 0; i < trailN - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
trailGeo.setIndex(idx);
const trail = new THREE.Mesh(trailGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false }));
trail.frustumCulled = false;
const shieldMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20), new THREE.MeshBasicMaterial({ color: 0x7fd8ff, transparent: true, opacity: .18, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
const shieldRim = new THREE.Mesh(new THREE.TorusGeometry(1, .03, 8, 60), new THREE.MeshBasicMaterial({ color: 0x7fd8ff, toneMapped: false }));
shieldMesh.add(shieldRim);
const auraLight = new THREE.PointLight(0xff5a1f, 0, 6, 2);

let animState = '';
function setAnim(name, opts) { if (animState === name && !(opts && opts.force)) return; animState = name; hero.play(name, opts); }

function syncScene(dt) {
  const p = S.p, th = themeAt(S.cam), bonus = !!S.bonus;
  const hx = toX(S.cam + PX), hy = toY(p.y);
  const speedK = clamp((S.speed - 400) / 550, 0, 1);
  const menu = state === 'menu';

  // 主题光照与雾
  scene.fog.color.set(bonus ? '#ff9ad5' : th.fog);
  scene.fog.near = bonus ? 40 : 16; scene.fog.far = bonus ? 220 : 95;
  hemi.color.set(bonus ? '#ffd0f0' : th.hemi);
  rim.color.set(bonus ? '#ffffff' : th.accent);
  fill.color.set(bonus ? '#ff9ad5' : th.grid);
  if (th.cityA > .5) world.setCity(th.city, CITY.build(th.city).canvas);

  // 英雄动画（坐骑常驻，英雄全程骑乘）
  const mounted = !!S.mount && !p.dead;
  p.crouch += ((p.sliding ? 1 : 0) - p.crouch) * Math.min(1, dt * 14);
  if (p.dead) { /* 死亡动画已在 die() 播放 */ }
  else if (mounted || menu || state === 'count') setAnim('Idle_swordRight');
  else if (p.sliding) setAnim('Roll', { timeScale: 1.25 });
  else if (!p.onGround && !S.power.boost) { if (p.justJumped) setAnim('Jump', { once: true, force: true, timeScale: 1.5 }); }
  else if (S.power.boost) setAnim('Run_swordRight', { timeScale: 1.6 });
  else setAnim('Run', { timeScale: .85 + speedK * .75 });
  p.justJumped = false;
  hero.update(dt * (state === 'dying' ? slowmo * 1.6 : 1), { speed: menu ? 0 : speedK + (S.power.boost ? .6 : 0), ride: mounted, slide: p.sliding });

  // 坐骑：跟英雄一起跑、一起跳、一起趴下
  for (const m of Object.values(mounts)) m.visible = false;
  let heroY = hy;
  const face = menu ? -.35 : 0;
  if (S.mount) {
    const m = mounts[S.mount.type];
    m.visible = true;
    const dead = p.dead;
    m.position.set(hx, hy, 0);
    const pitch = dead ? -(1.3 - dyingT) * 2.2 : (!p.onGround && !S.bonus ? clamp(-p.vy * .00028, -.3, .32) : 0) * (1 - p.crouch);
    m.rotation.set(0, face, pitch);
    animateMount(m, menu ? performance.now() / 1000 : S.t, menu || state === 'count' ? 0 : p.runT * 1.2, p.onGround || menu, p.crouch);
    if (mounted) {
      const sx = m.userData.saddle.x, sy = m.userData.saddle.y - .82 - p.crouch * .05;
      const c = Math.cos(pitch), s2 = Math.sin(pitch);
      heroY = m.position.y + sx * s2 + sy * c;
      hero.root.position.set(m.position.x + sx * c - sy * s2, heroY, 0);
    } else hero.root.position.set(hx - .4, hy + .4, .3);
    hero.root.rotation.set(0, face, mounted ? pitch : 0);
  } else hero.root.position.set(hx, hy, 0), hero.root.rotation.set(0, face, 0);
  hero.body.position.y = .75; hero.model.position.y = -.75;
  // 滑铲时骑手向后躺平，贴着坐骑一起钻过去
  hero.body.rotation.z = mounted ? p.crouch * 1.15 : p.flip > 0 ? -(1 - p.flip) * Math.PI * 2 : 0;
  hero.body.scale.set(1 + (1 - p.sq) * .4, p.sq, 1);
  const blink = p.invuln > 0 && Math.floor(S.t * 14) % 2 === 0;
  hero.root.visible = !blink;

  // 宠物跟随
  const pet = pets[loadout.pet];
  for (const [k, g] of Object.entries(pets)) g.visible = k === loadout.pet;
  const target = menu ? new THREE.Vector3(hx - .75, 2.05 + Math.sin(S.t * 2) * .12, .4) : new THREE.Vector3(hx - 1.3, heroY + (mounted ? 2.9 : 2.2) + Math.sin(S.t * 3) * .18, .5);
  if (S.pet.lengthSq() === 0) S.pet.copy(target);
  S.pet.lerp(target, Math.min(1, dt * 6));
  pet.position.copy(S.pet);
  pet.rotation.y = menu ? -.5 : 0;
  animatePet(pet, S.t);
  if (loadout.pet === 'dragon' && !menu && Math.random() < .08) addPart(pet.position.x + .45, pet.position.y + .12, pet.position.z, 6, rand(-.4, .4), 0, .3, .14, pick(['#ffd23d', '#ff6a00']), 0);

  // 护盾、气场
  shieldMesh.visible = S.power.shield && !p.dead;
  if (shieldMesh.visible) {
    const h = mounted ? 1.4 : p.sliding ? .5 : .85;
    shieldMesh.position.set(hx, heroY + h, 0); shieldMesh.scale.set(mounted ? 1.4 : .9, h * 1.2, .9);
    shieldRim.rotation.set(Math.PI / 2, 0, S.t * 2);
    shieldMesh.material.opacity = .14 + Math.sin(S.t * 6) * .05;
  }
  const hot = S.power.boost || S.feverT > 0 || bonus;
  auraLight.intensity = hot ? 5 + Math.sin(S.t * 10) * 1.5 : 0;
  auraLight.color.set(HEROES[loadout.hero].glow);
  auraLight.position.set(hx - .3, heroY + 1.6, .6);

  // 光带拖尾（头部附近）
  const tip = new THREE.Vector3(hx - .25, heroY + (mounted ? 1.5 - p.crouch * .9 : p.sliding ? .5 : 1.3), 0);
  if (!menu) { for (const q of trailPts) q.x -= (S.speed * (S.power.boost ? 1.75 : 1)) * U * dt * .35; trailPts.unshift(tip.clone()); }
  if (trailPts.length > trailN) trailPts.length = trailN;
  trail.visible = !menu && !p.dead && trailPts.length > 2;
  if (trail.visible) {
    const pos = trailGeo.attributes.position.array, col = trailGeo.attributes.color.array;
    const accent = new THREE.Color(HEROES[loadout.hero].glow);
    for (let i = 0; i < trailN; i++) {
      const q = trailPts[Math.min(i, trailPts.length - 1)], k = 1 - i / trailN;
      const w = (hot ? .32 : .16) * k + .01;
      pos.set([q.x, q.y + w, q.z, q.x, q.y - w, q.z], i * 6);
      const c = S.feverT > 0 || bonus ? tmpColor.setHSL((S.t * 1.2 + i * .05) % 1, 1, .55) : accent;
      col.set([c.r * k, c.g * k, c.b * k, c.r * k, c.g * k, c.b * k], i * 6);
    }
    trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
  }

  // 障碍
  const seen = new Set();
  for (const o of S.obstacles) {
    if (o.dead) continue;
    if (o.x - S.cam > W + 400) continue;
    let m = obsMesh.get(o);
    if (!m) { m = makeObstacle(o); scene.add(m); obsMesh.set(o, m); }
    seen.add(o);
    const cx = toX(o.x + o.w / 2);
    if (o.type === 'bird') {
      m.position.set(cx, toY(o.y + o.h / 2), 0);
      m.rotation.z = Math.sin(S.t * 3 + o.x) * .08;
      for (const r of m.userData.rotors) r.rotation.y += dt * 40;
    } else m.position.set(cx, 0, 0);
    if (o.type === 'bar') for (const b of m.userData.beams) b.material.emissiveIntensity = 3 + Math.sin(S.t * 30) * 1.2;
  }
  for (const [o, m] of obsMesh) if (!seen.has(o)) { scene.remove(m); disposeTree(m); obsMesh.delete(o); }

  // 道具
  const seenI = new Set();
  for (const it of S.items) {
    if (it.taken) continue;
    let m = itemMesh.get(it);
    if (!m) { m = makeItem(it.type, MOUNTS[loadout.mount].icon, it.type === 'mount' ? MOUNTS[loadout.mount].color : '#ffffff'); scene.add(m); itemMesh.set(it, m); }
    seenI.add(it);
    m.position.set(toX(it.x), toY(it.y) + Math.sin(S.t * 3 + it.x) * .12, 0);
    m.userData.ring.rotation.set(S.t * 1.5, S.t * 2, 0); m.userData.ring2.rotation.z = -S.t * 2;
  }
  for (const [it, m] of itemMesh) if (!seenI.has(it)) { scene.remove(m); disposeTree(m); itemMesh.delete(it); }

  // 金币与宝石（实例化）
  let nc = 0, ng = 0;
  for (const c of S.coins_) {
    if (c.taken || c.x - S.cam > W + 300) continue;
    dummy.position.set(toX(c.x), toY(c.y), 0);
    if (c.gem) {
      if (ng >= 1400) continue;
      dummy.rotation.set(0, S.t * 2 + c.x * .01, 0); dummy.scale.setScalar(1); dummy.updateMatrix();
      gemsIM.setMatrixAt(ng, dummy.matrix); gemsIM.setColorAt(ng, tmpColor.set(c.gem)); ng++;
    } else {
      if (nc >= 900) continue;
      dummy.rotation.set(0, S.t * 3 + c.x * .02, 0); dummy.scale.setScalar(1); dummy.updateMatrix();
      coinsIM.setMatrixAt(nc++, dummy.matrix);
    }
  }
  coinsIM.count = nc; coinsIM.instanceMatrix.needsUpdate = true;
  gemsIM.count = ng; gemsIM.instanceMatrix.needsUpdate = true; if (gemsIM.instanceColor) gemsIM.instanceColor.needsUpdate = true;

  // 平台（复用地面池之外的简单网格）
  syncPlatforms(th);

  // 场景
  const camX = hx + (menu ? 1.2 : 4.6);
  world.update(dt, camX, th, zoneKindAt, S.ground, bonus);

  // 粒子
  const n = parts.length;
  for (let i = n - 1; i >= 0; i--) {
    const q = parts[i];
    q.t += dt * (state === 'dying' ? slowmo : 1);
    if (q.t >= q.life) { parts.splice(i, 1); continue; }
    const d2 = dt * (state === 'dying' ? slowmo : 1);
    q.vy -= q.grav * d2; q.x += q.vx * d2; q.y += q.vy * d2; q.z += q.vz * d2;
  }
  for (let i = 0; i < parts.length; i++) {
    const q = parts[i], k = 1 - q.t / q.life;
    pPos[i * 3] = q.x; pPos[i * 3 + 1] = q.y; pPos[i * 3 + 2] = q.z;
    pCol[i * 3] = q.r * k; pCol[i * 3 + 1] = q.g * k; pCol[i * 3 + 2] = q.b * k;
    pSize[i] = q.size * (.6 + .4 * k);
  }
  pGeo.setDrawRange(0, parts.length);
  pGeo.attributes.position.needsUpdate = pGeo.attributes.color.needsUpdate = pGeo.attributes.size.needsUpdate = true;
  for (let i = rings.length - 1; i >= 0; i--) {
    const r = rings[i]; r.t += dt;
    r.m.scale.setScalar(1 + r.t * 7); r.m.material.opacity = Math.max(0, 1 - r.t / .45);
    r.m.lookAt(camera.position);
    if (r.t > .45) { scene.remove(r.m); disposeTree(r.m); rings.splice(i, 1); }
  }

  // 摄像机
  let cx, cy, cz, lx, ly;
  if (menu) { cx = hx + 1.9; cy = 1.7; cz = 6.4; lx = hx + 1.6; ly = 1.15; }
  else {
    const followY = bonus ? 0 : Math.max(0, hy - 1) * .35;
    cx = hx + 3.9; cy = (bonus ? 4.2 : 2.3) + followY; cz = bonus ? 15 : 11; lx = hx + 3.9; ly = (bonus ? 3.6 : 1.55) + followY;
    if (state === 'dying' || state === 'over') { cz = 9.5; cx = hx + 2; lx = hx + 1.2; ly = Math.max(.8, hy + .6); }
  }
  const k = Math.min(1, dt * (menu ? 3 : 5));
  camera.position.x += (cx - camera.position.x) * (menu ? k : 1);
  camera.position.y += (cy - camera.position.y) * k;
  camera.position.z += (cz - camera.position.z) * k * .6;
  const look = new THREE.Vector3(lx, ly, 0);
  if (shake.t > 0) { shake.t -= dt; const a = shake.amp * Math.min(1, shake.t * 3); camera.position.x += rand(-1, 1) * a * .3; camera.position.y += rand(-1, 1) * a * .3; }
  camera.lookAt(look);
  if (window.__camOverride) window.__camOverride(camera, hx, heroY); // 调试截图用

  key.position.set(hx + 6, 14, 10); key.target.position.set(hx + 3, 0, 0);
  rim.position.set(hx - 4, 6, -10); rim.target.position.set(hx, 1, 0);
  fill.position.set(hx + 10, 3, 6); fill.target.position.set(hx, 1, 0);
  bloom.strength = bonus ? .55 : .7 + (hot ? .25 : 0);

  for (const el of $('pops').querySelectorAll('.pop')) placePop(el);
}

const platPool = [];
const platTop = new THREE.MeshStandardMaterial({ color: 0x120a24, roughness: .3, metalness: .7 });
function syncPlatforms(th) {
  let i = 0;
  for (const pl of S.platforms) {
    let m = platPool[i];
    if (!m) {
      m = new THREE.Group();
      const b = new THREE.Mesh(new THREE.BoxGeometry(1, .34, 1.6), platTop); b.position.y = -.17; b.castShadow = b.receiveShadow = true; m.add(b);
      const e = new THREE.Mesh(new THREE.BoxGeometry(1, .06, 1.62), new THREE.MeshStandardMaterial({ color: 0, emissive: 0xffffff, emissiveIntensity: 2.5, toneMapped: false })); m.add(e);
      const u = new THREE.Mesh(new THREE.BoxGeometry(.9, .03, 1.2), new THREE.MeshStandardMaterial({ color: 0, emissive: 0x33f0ff, emissiveIntensity: 2, toneMapped: false })); u.position.y = -.36; m.add(u);
      m.userData = { b, e, u };
      scene.add(m); platPool[i] = m;
    }
    m.visible = true;
    m.position.set(toX(pl.x + pl.w / 2), toY(pl.y), 0);
    m.userData.b.scale.x = m.userData.e.scale.x = pl.w * U; m.userData.u.scale.x = pl.w * U * .9;
    m.userData.u.material.emissive.set(th.accent);
    i++;
  }
  for (; i < platPool.length; i++) platPool[i].visible = false;
}

// ---------- HUD ----------
const powerEls = {};
function hudPower(id, icon, frac, color) {
  let el = powerEls[id];
  if (!el) {
    el = document.createElement('div'); el.className = 'pw';
    el.innerHTML = `<svg viewBox="0 0 52 52"><circle cx="26" cy="26" r="23" stroke="rgba(255,255,255,.15)"/><circle class="arc" cx="26" cy="26" r="23" stroke-dasharray="144.5" /></svg><span>${icon}</span>`;
    $('hPowers').appendChild(el); powerEls[id] = el;
  }
  el.style.display = frac > 0 ? '' : 'none';
  const arc = el.querySelector('.arc');
  arc.style.stroke = color; arc.style.filter = `drop-shadow(0 0 4px ${color})`;
  arc.style.strokeDashoffset = 144.5 * (1 - frac);
}
let hudCache = {};
function setText(id, v) { if (hudCache[id] !== v) { hudCache[id] = v; $(id).textContent = v; } }
function updateHud() {
  const th = themeAt(S.cam), pw = S.power;
  stage.style.setProperty('--acc', th.accent);
  setText('hScore', String(Math.floor(S.score)));
  setText('hCoins', String(S.coins));
  setText('hMeta', `${Math.floor(S.cam / 50)} 米 · LV.${S.level} · ${S.bonus ? '超级奖励' : S.zone % 3 === 0 ? th.city : th.name}`);
  $('hBonusRow').style.visibility = S.bonus ? 'hidden' : '';
  $('hBonus').style.width = Math.min(100, S.bonusCoins / S.bonusNeed * 100) + '%';
  setText('hBonusTxt', `${S.bonusCoins}/${S.bonusNeed}`);
  const fever = $('hFever');
  if (S.bonus) { $('hFeverBar').style.width = (S.bonus.t / BONUS_SECS * 100) + '%'; fever.classList.add('hot'); setText('hFeverTxt', `🌈 超级奖励 · ${Math.ceil(S.bonus.t)} 秒 · 已收 ${S.bonus.gained}`); }
  else { fever.classList.toggle('hot', S.feverT > 0); $('hFeverBar').style.width = ((S.feverT > 0 ? S.feverT / 7 : S.fever) * 100) + '%'; setText('hFeverTxt', S.feverT > 0 ? '🔥 狂热中' : '狂热'); }
  setText('hCombo', S.combo >= 5 ? `连击 ×${S.combo}` : '');
  hudPower('shield', '🛡️', pw.shield ? 1 : 0, '#7fd8ff');
  hudPower('boost', '🚀', pw.boost / 3.5, '#ff7b3a');
  hudPower('magnet', '🧲', pw.magnet / 8, '#ff5b9b');
  hudPower('double', '✨', pw.double / 10, '#ffd36b');
  // 来袭预警
  let up = false, down = false;
  if (state === 'play' && !S.bonus && !pw.boost) {
    const vis = 640, ahead = S.speed * .7;
    for (const o of S.obstacles) {
      const sx = o.x - S.cam;
      if (o.dead || sx < vis || sx > vis + ahead) continue;
      if (o.type === 'bar' || (o.type === 'bird' && (o.low || S.mount))) down = true; else if (o.type !== 'bird') up = true;
    }
    for (const g of S.ground) { const sx = g.x2 - S.cam; if (sx > vis && sx < vis + ahead) up = true; }
  }
  $('warnUp').hidden = !up; $('warnDown').hidden = !down;
}
function showBest() { $('bestLine').textContent = best ? `最高分 ${best}` : ''; }

// ---------- 选择英雄、宠物、坐骑 ----------
function buildLoadout() {
  const rows = [['loHero', 'hero', HEROES, id => '#' + HEROES[id].glow.toString(16).padStart(6, '0')], ['loPet', 'pet', PETS, id => ({ dragon: '#ff7a2a', fox: '#ff7ad9', panda: '#ffffff' })[id]], ['loMount', 'mount', MOUNTS, id => MOUNTS[id].color]];
  for (const [el, key, set, color] of rows) {
    $(el).innerHTML = Object.entries(set).map(([id, v]) => `<button class="lo-card${loadout[key] === id ? ' on' : ''}" data-k="${key}" data-id="${id}" style="--c:${color(id)}"><div class="nm"><i></i>${v.name}</div><div class="ds">${v.desc}</div></button>`).join('');
    $(el).onclick = e => {
      const card = e.target.closest('.lo-card');
      if (!card) return;
      loadout[key] = card.dataset.id;
      try { localStorage.setItem(LOADOUT_KEY, JSON.stringify(loadout)); } catch {}
      $(el).querySelectorAll('.lo-card').forEach(b => b.classList.toggle('on', b === card));
      if (key === 'mount' && S) S.mount = { type: loadout.mount };
      if (key === 'hero') { hero.setHero(loadout.hero); setAnim('Roll', { force: true }); setTimeout(() => setAnim('Idle_swordRight', { force: true }), 900); }
      FX.sound('coin');
    };
  }
}

// ---------- 主循环 ----------
function frame(now) {
  const dt = Math.min(.033, (now - last) / 1000 || 0);
  last = now;
  if (state === 'count') {
    countT -= dt;
    const n = Math.ceil(countT);
    if (n !== lastCount && n > 0) { lastCount = n; FX.sound('move'); countBig(String(n)); }
    if (countT <= 0) { state = 'play'; FX.sound('power'); Music.start(); banner('GO!', cityFor(0), '#33f0ff'); }
    S.t += dt;
  } else if (state === 'play') { update(dt); updateHud(); }
  else if (state === 'dying') {
    const sdt = dt * slowmo;
    dyingT -= dt;
    S.p.vy += GRAV * sdt; S.p.y = Math.min(S.p.y + S.p.vy * sdt, onGroundAt(S.cam + PX) ? GROUND : H + 300);
    S.t += sdt;
    if (dyingT <= 0) finish();
  } else if (state === 'menu') { S.t += dt; }
  if (S) syncScene(dt);
  composer.render();
  requestAnimationFrame(frame);
}

// ---------- 启动 ----------
async function boot() {
  FX.muteButton($('btnMute'));
  let pa = 0, pb = 0;
  const prog = () => { $('loadBar').style.width = ((pa * .55 + pb * .45) * 100) + '%'; };
  try {
    await Promise.all([loadHeroAssets(v => { pa = v; prog(); }), loadCityAssets(v => { pb = v; prog(); })]);
  } catch (e) {
    $('loadTxt').innerHTML = '模型载入失败：' + e.message + '<br>可以先玩 <a href="runner-2d.html" style="color:#33f0ff">经典 2D 版</a>';
    return;
  }
  hero = new Hero(); hero.fitAccessories(); hero.setHero(loadout.hero);
  scene.add(hero.root);
  for (const k of Object.keys(PETS)) { pets[k] = makePet(k); pets[k].visible = false; scene.add(pets[k]); }
  for (const k of Object.keys(MOUNTS)) { mounts[k] = makeMount(k); mounts[k].visible = false; scene.add(mounts[k]); }
  coinsIM = new THREE.InstancedMesh(coinGeo, coinMat, 900); coinsIM.count = 0; coinsIM.frustumCulled = false; scene.add(coinsIM);
  gemsIM = new THREE.InstancedMesh(gemGeo, gemMat, 1400); gemsIM.count = 0; gemsIM.frustumCulled = false;
  gemsIM.setColorAt(0, new THREE.Color()); scene.add(gemsIM);
  scene.add(trail, shieldMesh, auraLight);
  world = new World(scene);
  buildLoadout();
  reset();
  resize();
  $('loadOv').hidden = true;
  toMenu();
  camera.position.set(1, 1.6, 6);
  last = performance.now();
  requestAnimationFrame(frame);
  // 调试用：摆一段固定场景，便于截图检查
  const demo = async (opts = {}) => {
    start(); state = 'play'; S.p.invuln = 999;
    if (opts.cam) S.cam = opts.cam;
    for (let i = 0; i < 150; i++) { update(1 / 60); syncScene(1 / 60); }
    S.obstacles = []; S.items = []; S.platforms = []; S.ground = [{ x1: S.cam - 2000, x2: S.cam + 620 }, { x1: S.cam + 760, x2: S.cam + 5000 }]; S.nextX = S.cam + 5000;
    S.obstacles.push({ type: 'crate', x: S.cam + 520, y: GROUND - 44, w: 46, h: 44 }, { type: 'bar', x: S.cam + 900, y: 300, w: 80, h: 100 }, { type: 'spikes', x: S.cam + 1080, y: 422, w: 60, h: 28 }, { type: 'bird', x: S.cam + 1200, y: 362, w: 46, h: 30, vx: 0, flap: 0, low: true });
    for (let i = 0; i < 10; i++) S.coins_.push({ x: S.cam + 300 + i * 34, y: 400 - Math.sin(i / 9 * Math.PI) * 80, value: 10 });
    S.items.push({ type: 'shield', x: S.cam + 720, y: 330 });
    if (opts.pre) opts.pre(S);
    for (let i = 0; i < (opts.frames || 6); i++) { update(1 / 60); syncScene(1 / 60); }
    S.p.invuln = 0; state = 'demo';
  };
  window.__runner = { get S() { return S; }, get state() { return state; }, set state(v) { state = v; }, start, jump, slideDown, update, syncScene, enterBonus, exitBonus, hero, camera, scene, renderer, composer, loadout, world, toMenu, demo, enterBonusNow: () => enterBonus() };
}
boot();
