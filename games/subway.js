// 地铁跑酷 3D：三条轨道、火车（静止/迎面驶来）、坡道上车顶、跨栏、翻滚栏、金币、道具、滑板、巡警追逐。
// 坐标：玩家沿 -z 方向前进，s = 已跑距离（世界 z = -s）；x 为轨道横向位置，y 向上。
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { HEROES, loadHeroAssets, Hero, withRetry, keep } from './runner3d-actors.js';

const $ = id => document.getElementById(id);
const stage = $('stage');
const rnd = (a, b) => a + Math.random() * (b - a);
const pick = a => a[Math.floor(Math.random() * a.length)];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---------- 规则常量 ----------
const LANE = 2.2, LX = l => (l - 1) * LANE;
const GRAV = 46, JUMP_V = 14.6, SUPER_V = 20.5;       // 普通跳 ≈2.3 高，超级跳鞋 ≈4.6 高（能跳上车顶）
const STAND_H = 1.55, ROLL_H = .7, ROLL_T = .62;
const TRAIN_TOP = 2.85, CAR_LEN = 9, CAR_GAP = .45, TRAIN_W = 1.9;
const RAMP_LEN = 7;
const JET_Y = 7.2;
const POWER_SECS = { magnet: 12, sneakers: 12, jetpack: 8, double: 14 };
const POWER_INFO = {
  magnet: { icon: '🧲', name: '磁铁', color: '#ff4d6d' },
  sneakers: { icon: '👟', name: '超级跳鞋', color: '#7dff6a' },
  jetpack: { icon: '🚀', name: '喷气背包', color: '#ffb400' },
  double: { icon: '✖2', name: '双倍得分', color: '#4fc3ff' },
};
HEROES.guard = { name: '巡警', armor: 0x1d3f9a, armorMetal: .35, trim: 0xffd23a, skin: 0xc99a76, boots: 0x111522,
                 helm: 'Helmet1', helmColor: 0x1a2b66, weapon: 'ShortSword', cape: [0x2a56c9, 0x0d1f5a], eye: 0xffffff, glow: 0xffd23a };

// ---------- 渲染器、场景、光照 ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
const HAZE = 0xcfe9ff;
scene.fog = new THREE.Fog(HAZE, 55, 170);
const camera = new THREE.PerspectiveCamera(60, 16 / 9, .1, 600);
scene.add(new THREE.HemisphereLight(0xd6eeff, 0x9a7a55, 1.25));
const sun = new THREE.DirectionalLight(0xfff0d0, 2.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -14, right: 14, top: 36, bottom: -14, near: 1, far: 90 });
sun.shadow.bias = -.0005; sun.shadow.normalBias = .03;
scene.add(sun, sun.target);
{
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(new RoomEnvironment(), .04).texture;
}

function canvasTex(w, h, draw, repeat) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  return t;
}

// 天空：竖向渐变 + 几朵云
scene.background = canvasTex(4, 512, (g, w, h) => {
  const gr = g.createLinearGradient(0, 0, 0, h);
  gr.addColorStop(0, '#2c86ff'); gr.addColorStop(.55, '#86c9ff'); gr.addColorStop(1, '#e4f4ff');
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
});
const cloudTex = canvasTex(256, 128, (g, w, h) => {
  for (let i = 0; i < 14; i++) {
    const x = rnd(50, 206), y = rnd(50, 90), r = rnd(22, 40);
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(.7, 'rgba(255,255,255,.9)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
  }
});
const clouds = [];
for (let i = 0; i < 9; i++) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, fog: false, transparent: true, opacity: .95, depthWrite: false }));
  sp.scale.set(rnd(40, 70), rnd(18, 28), 1);
  sp.userData.off = [rnd(-120, 120), rnd(38, 70), rnd(150, 380)];
  clouds.push(sp); scene.add(sp);
}

// ---------- 共享材质与贴图 ----------
const M = {
  steel: keep(new THREE.MeshStandardMaterial({ color: 0xb8c0cc, metalness: .9, roughness: .3 })),
  sleeper: keep(new THREE.MeshStandardMaterial({ color: 0x6b4a32, roughness: .9 })),
  dark: keep(new THREE.MeshStandardMaterial({ color: 0x23262e, roughness: .7, metalness: .4 })),
  pole: keep(new THREE.MeshStandardMaterial({ color: 0x8a96a8, metalness: .7, roughness: .4 })),
  wire: keep(new THREE.MeshBasicMaterial({ color: 0x2a2f3a })),
  head: keep(new THREE.MeshStandardMaterial({ color: 0xfff6c8, emissive: 0xfff2a0, emissiveIntensity: 2.2 })),
  red: keep(new THREE.MeshStandardMaterial({ color: 0xff2a2a, emissive: 0xff1a1a, emissiveIntensity: 1.6 })),
  roof: keep(new THREE.MeshStandardMaterial({ color: 0x9aa3ad, metalness: .6, roughness: .45 })),
  rampSide: keep(new THREE.MeshStandardMaterial({ color: 0x5a6070, metalness: .5, roughness: .5 })),
  post: keep(new THREE.MeshStandardMaterial({ color: 0xf2f2f2, metalness: .3, roughness: .4 })),
};
const gravelTex = canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = '#8f8778'; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 2600; i++) {
    const v = rnd(90, 190) | 0; g.fillStyle = `rgb(${v},${v - 6},${v - 18})`;
    g.fillRect(rnd(0, w), rnd(0, h), rnd(1, 4), rnd(1, 4));
  }
}, [10, 80]);
const bedTex = canvasTex(128, 256, (g, w, h) => {
  g.fillStyle = '#6f6a62'; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 1400; i++) { const v = rnd(70, 150) | 0; g.fillStyle = `rgb(${v},${v - 4},${v - 10})`; g.fillRect(rnd(0, w), rnd(0, h), rnd(1, 3), rnd(1, 3)); }
}, [1, 60]);
const stripeTex = canvasTex(128, 32, (g, w, h) => {
  g.fillStyle = '#ffffff'; g.fillRect(0, 0, w, h);
  g.fillStyle = '#e8262a';
  for (let x = -h; x < w + h; x += 32) { g.beginPath(); g.moveTo(x, h); g.lineTo(x + 16, h); g.lineTo(x + 16 + h, 0); g.lineTo(x + h, 0); g.fill(); }
});
const hazardTex = canvasTex(128, 128, (g, w, h) => {
  g.fillStyle = '#ffc400'; g.fillRect(0, 0, w, h);
  g.fillStyle = '#1b1b1b';
  for (let x = -w; x < w * 2; x += 40) { g.beginPath(); g.moveTo(x, h); g.lineTo(x + 20, h); g.lineTo(x + 20 + h, 0); g.lineTo(x + h, 0); g.fill(); }
});
M.stripe = keep(new THREE.MeshStandardMaterial({ map: stripeTex, roughness: .5 }));
M.hazard = keep(new THREE.MeshStandardMaterial({ map: hazardTex, roughness: .6 }));
M.ramp = keep(new THREE.MeshStandardMaterial({ color: 0xffc400, roughness: .5, metalness: .3 }));

const TAGS = ['DEEPSEEK', 'RUN!', '酷跑', 'SUBWAY', 'YO', 'GO GO', '冲鸭', 'FAST', '666', 'WOW'];
const TAG_COLS = ['#ff4d6d', '#ffd23a', '#4fc3ff', '#7dff6a', '#c77dff', '#ff9a3c', '#ffffff'];
function graffiti(g, x, y, size) {
  const txt = pick(TAGS), c1 = pick(TAG_COLS), c2 = pick(TAG_COLS);
  g.save(); g.translate(x, y); g.rotate(rnd(-.18, .18));
  g.font = `900 ${size}px "Arial Rounded MT Bold", "PingFang SC", sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.lineJoin = 'round';
  g.lineWidth = size * .28; g.strokeStyle = '#1b1b2a'; g.strokeText(txt, 0, 0);
  g.lineWidth = size * .12; g.strokeStyle = c2; g.strokeText(txt, 0, 0);
  g.fillStyle = c1; g.fillText(txt, 0, 0);
  g.restore();
}
const wallTex = canvasTex(1024, 128, (g, w, h) => {
  g.fillStyle = '#b9b2a6'; g.fillRect(0, 0, w, h);
  g.fillStyle = 'rgba(0,0,0,.08)';
  for (let x = 0; x < w; x += 64) g.fillRect(x, 0, 3, h);
  for (let i = 0; i < 7; i++) graffiti(g, 70 + i * 140 + rnd(-20, 20), rnd(52, 78), rnd(34, 50));
  g.fillStyle = 'rgba(0,0,0,.18)'; g.fillRect(0, h - 10, w, 10);
}, [1, 1]);

// 火车涂装：车身两侧、车头
const LIVERIES = [
  { body: '#e8eef5', band: '#e53935', band2: '#ffb400' },
  { body: '#2f6fe0', band: '#ffffff', band2: '#ffd23a' },
  { body: '#ffb400', band: '#1b2340', band2: '#ffffff' },
  { body: '#2bb673', band: '#ffffff', band2: '#1b2340' },
  { body: '#d84bb0', band: '#ffffff', band2: '#4fc3ff' },
  { body: '#c7cfd8', band: '#1e88e5', band2: '#7dff6a' },
];
function sideTex(L) {
  return canvasTex(1024, 256, (g, w, h) => {
    g.fillStyle = L.body; g.fillRect(0, 0, w, h);
    g.fillStyle = L.band; g.fillRect(0, h * .66, w, h * .12);
    g.fillStyle = L.band2; g.fillRect(0, h * .8, w, h * .05);
    // 车窗与车门
    for (let x = 40; x < w - 60; x += 120) {
      const door = (x / 120 | 0) % 3 === 1;
      if (door) {
        g.fillStyle = 'rgba(0,0,0,.25)'; g.fillRect(x - 4, h * .14 - 4, 64, h * .76 + 4);
        g.fillStyle = '#26303d'; g.fillRect(x, h * .14, 56, h * .5);
        g.fillStyle = '#7fb6e6'; g.fillRect(x + 6, h * .18, 18, h * .3); g.fillRect(x + 32, h * .18, 18, h * .3);
      } else {
        g.fillStyle = '#1b2230'; g.fillRect(x, h * .16, 90, h * .38);
        const gr = g.createLinearGradient(x, h * .16, x + 90, h * .54);
        gr.addColorStop(0, '#a8d8ff'); gr.addColorStop(.5, '#4a7fb0'); gr.addColorStop(1, '#2a4a70');
        g.fillStyle = gr; g.fillRect(x + 5, h * .19, 80, h * .32);
        g.fillStyle = 'rgba(255,255,255,.35)'; g.fillRect(x + 10, h * .21, 22, h * .06);
      }
    }
    // 涂鸦
    for (let i = 0; i < 3; i++) if (Math.random() < .7) graffiti(g, rnd(120, w - 120), rnd(h * .62, h * .82), rnd(46, 70));
    g.fillStyle = 'rgba(0,0,0,.25)'; g.fillRect(0, h - 14, w, 14);
  });
}
function frontTex(L) {
  return canvasTex(256, 320, (g, w, h) => {
    g.fillStyle = L.body; g.fillRect(0, 0, w, h);
    g.fillStyle = '#1b2230'; g.beginPath(); g.roundRect(22, 34, w - 44, h * .36, 18); g.fill();
    const gr = g.createLinearGradient(0, 40, 0, h * .44);
    gr.addColorStop(0, '#bfe4ff'); gr.addColorStop(1, '#3d6a98');
    g.fillStyle = gr; g.beginPath(); g.roundRect(30, 42, w - 60, h * .33, 14); g.fill();
    g.fillStyle = L.band; g.fillRect(0, h * .62, w, h * .1);
    g.fillStyle = '#1b2340'; g.fillRect(w / 2 - 34, 8, 68, 22);
    g.fillStyle = '#ffd23a'; g.font = '900 18px sans-serif'; g.textAlign = 'center'; g.fillText(String(rnd(1, 99) | 0).padStart(2, '0'), w / 2, 26);
    g.fillStyle = '#2a2f3a'; g.fillRect(w / 2 - 50, h * .8, 100, h * .14);
  });
}
const liveryMats = LIVERIES.map(L => {
  const side = keep(new THREE.MeshStandardMaterial({ map: sideTex(L), roughness: .4, metalness: .35 }));
  const front = keep(new THREE.MeshStandardMaterial({ map: frontTex(L), roughness: .35, metalness: .3 }));
  const plain = keep(new THREE.MeshStandardMaterial({ color: L.body, roughness: .4, metalness: .35 }));
  return { side, front, plain };
});

// ---------- 轨道、地面、围墙（跟随玩家，按贴图周期对齐，看不出移动） ----------
const TRACK_LEN = 340;
const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, TRACK_LEN), new THREE.MeshStandardMaterial({ map: gravelTex, roughness: 1 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
scene.add(ground);
const trackGroup = new THREE.Group();
scene.add(trackGroup);
{
  const railGeo = new THREE.BoxGeometry(.09, .16, TRACK_LEN);
  const bedGeo = new THREE.PlaneGeometry(2.1, TRACK_LEN);
  const bedMat = new THREE.MeshStandardMaterial({ map: bedTex, roughness: 1 });
  for (let l = 0; l < 3; l++) {
    const bed = new THREE.Mesh(bedGeo, bedMat);
    bed.rotation.x = -Math.PI / 2; bed.position.set(LX(l), .015, 0); bed.receiveShadow = true;
    trackGroup.add(bed);
    for (const dx of [-.55, .55]) {
      const r = new THREE.Mesh(railGeo, M.steel);
      r.position.set(LX(l) + dx, .2, 0); r.receiveShadow = true;
      trackGroup.add(r);
    }
  }
  const n = Math.floor(TRACK_LEN / 1.2);
  const sl = new THREE.InstancedMesh(new THREE.BoxGeometry(1.7, .1, .26), M.sleeper, n * 3);
  const m4 = new THREE.Matrix4();
  let k = 0;
  for (let l = 0; l < 3; l++) for (let i = 0; i < n; i++) { m4.makeTranslation(LX(l), .08, -TRACK_LEN / 2 + i * 1.2); sl.setMatrixAt(k++, m4); }
  sl.receiveShadow = true;
  trackGroup.add(sl);
  const wallGeo = new THREE.BoxGeometry(.5, 2.6, TRACK_LEN);
  for (const sgn of [-1, 1]) {
    const tex = wallTex.clone(); tex.needsUpdate = true; tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(TRACK_LEN / 18, 1);
    const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: .9 });
    const top = new THREE.MeshStandardMaterial({ color: 0x9d968a, roughness: .9 });
    const w = new THREE.Mesh(wallGeo);
    // 只有朝向轨道的那一面画涂鸦（侧面 uv 的横向沿 z）
    w.material = sgn < 0 ? [mat, top, top, top, top, top] : [top, mat, top, top, top, top];
    w.position.set(sgn * (LANE * 1 + 2.75), 1.3, 0);
    w.receiveShadow = true; w.castShadow = true;
    trackGroup.add(w);
  }
}
const SNAP = 18 * 1.2; // 枕木间距、围墙贴图周期的公倍数，按它对齐就看不出位移
function followTrack(s) {
  const z = -Math.floor(s / SNAP) * SNAP - TRACK_LEN / 2 + 30;
  trackGroup.position.z = z;
  ground.position.z = -Math.floor(s / 4) * 4 - TRACK_LEN / 2 + 30;
}

// ---------- 两侧城市：Kenney 楼房循环复用 ----------
const BUILD_NAMES = ['building-a', 'building-b', 'building-c', 'building-d', 'building-e', 'building-f', 'building-g', 'building-h',
                     'building-i', 'building-j', 'building-k', 'building-l', 'building-m', 'building-n',
                     'building-skyscraper-a', 'building-skyscraper-b', 'building-skyscraper-c', 'building-skyscraper-d', 'building-skyscraper-e'];
const buildModels = {};
async function loadBuildings(onProgress) {
  const loader = new GLTFLoader();
  let done = 0;
  await Promise.all(BUILD_NAMES.map(async n => {
    const g = await withRetry(() => loader.loadAsync(`assets/city/${n}.glb`));
    g.scene.traverse(o => { if (o.isMesh) { o.castShadow = false; o.receiveShadow = false; if (o.material.map) o.material.map.anisotropy = 4; } });
    const box = new THREE.Box3().setFromObject(g.scene);
    buildModels[n] = { scene: g.scene, size: box.getSize(new THREE.Vector3()), min: box.min.clone() };
    onProgress && onProgress(++done / BUILD_NAMES.length);
  }));
}
const buildings = [];
const BUILD_STEP = 9;
let buildNext = { '-1': -20, '1': -20 };
function placeBuilding(b, side) {
  const names = Object.keys(buildModels);
  const n = pick(names), m = buildModels[n];
  if (b.model) b.root.remove(b.model);
  const model = m.scene.clone();
  const tall = n.includes('sky');
  const k = (tall ? rnd(22, 34) : rnd(9, 16)) / m.size.y;
  model.scale.setScalar(k);
  model.position.set(-(m.min.x + m.size.x / 2) * k, -m.min.y * k, -(m.min.z + m.size.z / 2) * k);
  b.root.add(model); b.model = model;
  // 转 90° 后：模型 x 沿轨道方向，z 朝向轨道
  const depth = m.size.x * k;
  b.root.position.set(side * (9.5 + m.size.z * k / 2 + rnd(0, 3)), 0, -buildNext[side] - depth / 2);
  b.root.rotation.y = side < 0 ? Math.PI / 2 : -Math.PI / 2;
  b.side = side; b.s1 = buildNext[side] + Math.max(depth, BUILD_STEP);
  buildNext[side] = b.s1 + rnd(.5, 3);
}
function initBuildings() {
  for (const b of buildings) scene.remove(b.root);
  buildings.length = 0;
  buildNext = { '-1': -20, '1': -20 };
  for (let i = 0; i < 36; i++) {
    const b = { root: new THREE.Group(), model: null };
    scene.add(b.root);
    placeBuilding(b, i % 2 ? 1 : -1);
    buildings.push(b);
  }
}
function updateBuildings(s) {
  for (const b of buildings) if (b.s1 < s - 12) placeBuilding(b, b.side);
}

// 接触网立柱（每 30 米一组）
const poles = [];
{
  const postGeo = new THREE.CylinderGeometry(.12, .15, 7.2, 8), beamGeo = new THREE.BoxGeometry(14, .22, .22), wireGeo = new THREE.BoxGeometry(.03, .03, 30);
  for (let i = 0; i < 8; i++) {
    const g = new THREE.Group();
    for (const sx of [-1, 1]) { const p = new THREE.Mesh(postGeo, M.pole); p.position.set(sx * 6.6, 3.6, 0); p.castShadow = true; g.add(p); }
    const beam = new THREE.Mesh(beamGeo, M.pole); beam.position.y = 7; g.add(beam);
    for (let l = 0; l < 3; l++) { const w = new THREE.Mesh(wireGeo, M.wire); w.position.set(LX(l), 6.5, -15); g.add(w); }
    scene.add(g); poles.push(g);
  }
}
function updatePoles(s) {
  const base = Math.floor(s / 30) * 30;
  poles.forEach((g, i) => g.position.z = -(base + (i - 1) * 30));
}

// ---------- 障碍物 ----------
const geo = {
  car: new THREE.BoxGeometry(TRAIN_W, 2.5, CAR_LEN),
  bogie: new THREE.BoxGeometry(1.5, .4, 2.2),
  ac: new THREE.BoxGeometry(1.1, .22, 1.8),
  light: new THREE.BoxGeometry(.32, .16, .05),
  postLow: new THREE.BoxGeometry(.12, 1.05, .12),
  boardLow: new THREE.BoxGeometry(1.95, .34, .1),
  postHigh: new THREE.BoxGeometry(.14, 2.7, .14),
  boardHigh: new THREE.BoxGeometry(1.95, .8, .12),
  block: new THREE.BoxGeometry(1.95, 2.35, .9),
  lamp: new THREE.SphereGeometry(.12, 10, 8),
};
for (const g of Object.values(geo)) keep(g);
const rampGeo = (() => {
  const sh = new THREE.Shape();
  sh.moveTo(0, 0); sh.lineTo(RAMP_LEN, 0); sh.lineTo(RAMP_LEN, TRAIN_TOP); sh.lineTo(0, 0);
  const g = new THREE.ExtrudeGeometry(sh, { depth: TRAIN_W, bevelEnabled: false });
  g.rotateY(Math.PI / 2);               // 形状 x → 世界 -z
  g.translate(-TRAIN_W / 2, 0, 0);
  return keep(g);
})();

function makeTrain(cars, livery, moving) {
  const L = liveryMats[livery];
  const g = new THREE.Group();
  for (let i = 0; i < cars; i++) {
    const lead = i === 0;
    // 盒子面顺序：+x, -x, +y, -y, +z（朝玩家）, -z
    const body = new THREE.Mesh(geo.car, [L.side, L.side, M.roof, M.dark, lead ? L.front : L.plain, L.plain]);
    const z0 = -(i * (CAR_LEN + CAR_GAP) + CAR_LEN / 2);
    body.position.set(0, .35 + 1.25, z0);
    body.castShadow = true; body.receiveShadow = true;
    g.add(body);
    for (const dz of [-CAR_LEN * .32, CAR_LEN * .32]) { const b = new THREE.Mesh(geo.bogie, M.dark); b.position.set(0, .22, z0 + dz); g.add(b); }
    const ac = new THREE.Mesh(geo.ac, M.roof); ac.position.set(0, TRAIN_TOP + .11, z0); ac.castShadow = true; g.add(ac);
    if (lead) for (const dx of [-.6, .6]) { const h = new THREE.Mesh(geo.light, moving ? M.head : M.dark); h.position.set(dx, .75, .01); g.add(h); }
  }
  if (moving) { const lamp = new THREE.PointLight(0xfff2a0, 6, 14, 1.6); lamp.position.set(0, 1.4, 1.5); g.add(lamp); }
  return g;
}
function makeHurdle(type) {
  const g = new THREE.Group();
  if (type === 'low') {
    for (const dx of [-.88, .88]) { const p = new THREE.Mesh(geo.postLow, M.post); p.position.set(dx, .52, 0); p.castShadow = true; g.add(p); }
    const b = new THREE.Mesh(geo.boardLow, M.stripe); b.position.y = .82; b.castShadow = true; g.add(b);
  } else if (type === 'high') {
    for (const dx of [-.88, .88]) { const p = new THREE.Mesh(geo.postHigh, M.post); p.position.set(dx, 1.35, 0); p.castShadow = true; g.add(p); }
    const b = new THREE.Mesh(geo.boardHigh, M.stripe); b.position.y = 1.85; b.castShadow = true; g.add(b);
    for (const dx of [-.5, .5]) { const l = new THREE.Mesh(geo.lamp, M.red); l.position.set(dx, 2.35, 0); g.add(l); }
  } else {
    const b = new THREE.Mesh(geo.block, M.hazard); b.position.y = 1.18; b.castShadow = true; b.receiveShadow = true; g.add(b);
    for (const dx of [-.6, 0, .6]) { const l = new THREE.Mesh(geo.lamp, M.red); l.position.set(dx, 2.42, 0); g.add(l); }
  }
  return g;
}

// ---------- 金币（实例化） ----------
const COIN_CAP = 900;
const coinMesh = new THREE.InstancedMesh(
  new THREE.CylinderGeometry(.34, .34, .09, 24).rotateX(Math.PI / 2),
  new THREE.MeshStandardMaterial({ color: 0xffd23a, metalness: .55, roughness: .28, emissive: 0xc88a00, emissiveIntensity: .55 }),
  COIN_CAP);
coinMesh.castShadow = true;
coinMesh.frustumCulled = false;
scene.add(coinMesh);

// 道具：发光气泡 + emoji
const itemTex = {};
for (const [k, v] of Object.entries(POWER_INFO)) {
  itemTex[k] = canvasTex(128, 128, (g, w, h) => {
    const gr = g.createRadialGradient(64, 64, 10, 64, 64, 62);
    gr.addColorStop(0, 'rgba(255,255,255,.95)'); gr.addColorStop(.7, v.color); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.beginPath(); g.arc(64, 64, 62, 0, 7); g.fill();
    g.font = k === 'double' ? '900 46px sans-serif' : '64px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillStyle = '#1b2340'; g.fillText(v.icon, 64, 68);
  });
}

// 收集时的闪光
const sparkTex = canvasTex(64, 64, (g) => {
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,220,1)'); gr.addColorStop(.3, 'rgba(255,214,80,.9)'); gr.addColorStop(1, 'rgba(255,180,0,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
});
const sparks = [];
for (let i = 0; i < 40; i++) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: sparkTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  sp.visible = false; scene.add(sp); sparks.push({ sp, t: 1 });
}
let sparkIdx = 0;
function spark(x, y, z, size = 1.1) {
  const s = sparks[sparkIdx++ % sparks.length];
  s.sp.position.set(x, y, z); s.t = 0; s.size = size; s.sp.visible = true;
}

// ---------- 角色 ----------
let hero = null, guard = null, board = null, jetpack = null;
function makeBoard() {
  const g = new THREE.Group();
  const deck = new THREE.Mesh(new THREE.BoxGeometry(.62, .08, 1.6), new THREE.MeshStandardMaterial({ color: 0xff3d7f, roughness: .3, metalness: .4 }));
  g.add(deck);
  const glow = new THREE.Mesh(new THREE.BoxGeometry(.66, .03, 1.66), new THREE.MeshBasicMaterial({ color: 0x4fe3ff }));
  glow.position.y = -.05; g.add(glow);
  for (const dz of [-.55, .55]) for (const dx of [-.22, .22]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(.08, .08, .08, 10).rotateZ(Math.PI / 2), M.dark);
    w.position.set(dx, -.1, dz); g.add(w);
  }
  g.visible = false;
  return g;
}
function makeJetpack() {
  const g = new THREE.Group();
  const tankMat = new THREE.MeshStandardMaterial({ color: 0xd8dde6, metalness: .8, roughness: .25 });
  const flameMat = new THREE.MeshBasicMaterial({ color: 0xffa020, transparent: true, opacity: .9 });
  g.userData.flames = [];
  for (const dx of [-.18, .18]) {
    const t = new THREE.Mesh(new THREE.CylinderGeometry(.13, .13, .62, 12), tankMat); t.position.set(dx, 0, 0); g.add(t);
    const f = new THREE.Mesh(new THREE.ConeGeometry(.12, .6, 10).rotateX(Math.PI), flameMat); f.position.set(dx, -.6, 0); g.add(f);
    g.userData.flames.push(f);
  }
  g.visible = false;
  return g;
}

// ---------- 游戏状态 ----------
let state = 'load';           // load | menu | run | dead | over | pause
let heroId = 'ninja';
try { heroId = localStorage.getItem('subway-hero') || 'ninja'; } catch {}
if (!HEROES[heroId] || heroId === 'guard') heroId = 'ninja';
let S = null;
const obstacles = [], coins = [], items = [];

function reset() {
  for (const o of obstacles) scene.remove(o.mesh);
  for (const it of items) scene.remove(it.mesh);
  obstacles.length = 0; coins.length = 0; items.length = 0;
  S = {
    s: 0, speed: 15, score: 0, coins: 0, boards: 3, boardT: 0, invT: 0, gen: 40, movers: [],
    power: {}, powersUsed: 0, chaseT: 0, guardGap: 2.2, deadT: 0, shake: 0, reason: '',
    p: { lane: 1, from: 1, x: 0, y: 0, vy: 0, onGround: true, rollT: 0, prevS: 0 },
  };
  initBuildings();
  followTrack(0); updatePoles(0);
}

// ---------- 关卡生成 ----------
function addObstacle(o) {
  o.mesh.position.set(LX(o.lane), 0, -o.s0);
  scene.add(o.mesh);
  obstacles.push(o);
  return o;
}
function addTrain(lane, s0, cars, { ramp = false, v = 0 } = {}) {
  if (ramp) {
    const m = new THREE.Mesh(rampGeo, [M.ramp, M.hazard]);
    m.castShadow = true; m.receiveShadow = true;
    addObstacle({ type: 'ramp', lane, s0, s1: s0 + RAMP_LEN, mesh: m, walk: true });
    s0 += RAMP_LEN;
  }
  const len = cars * (CAR_LEN + CAR_GAP) - CAR_GAP;
  return addObstacle({ type: 'train', lane, s0, s1: s0 + len, y0: 0, y1: TRAIN_TOP, walk: true, v, mesh: makeTrain(cars, Math.random() * LIVERIES.length | 0, v > 0) });
}
function addHurdle(lane, s, type) {
  const d = type === 'block' ? .45 : .12;
  const [y0, y1] = type === 'low' ? [0, 1.0] : type === 'high' ? [1.2, 2.7] : [0, 2.35];
  addObstacle({ type, lane, s0: s - d, s1: s + d, y0, y1, walk: false, mesh: makeHurdle(type) });
}
function coinLine(lane, s0, n, y = .9, step = 2.2) { for (let i = 0; i < n; i++) coins.push({ x: LX(lane), y, s: s0 + i * step }); }
function coinArc(lane, sMid, base = 0) {
  for (let i = -3; i <= 3; i++) { const k = 1 - (i / 3.5) ** 2; coins.push({ x: LX(lane), y: base + .9 + k * 1.7, s: sMid + i * 1.6 }); }
}
function addItem(lane, s, kind) {
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: itemTex[kind], transparent: true }));
  sp.scale.set(1.25, 1.25, 1);
  sp.position.set(LX(lane), 1.3, -s);
  scene.add(sp);
  items.push({ kind, lane, s, mesh: sp, y: 1.3 });
}
function maybeItem(lane, s) {
  if (Math.random() > .16) return;
  addItem(lane, s, pick(['magnet', 'magnet', 'sneakers', 'jetpack', 'double']));
}

function genRow() {
  const sp = S.speed;
  const s0 = S.gen + rnd(12, 20) + sp * .45;
  const lanes = [0, 1, 2].sort(() => Math.random() - .5);
  if (S.gen > 300 && Math.random() < .2) { genMoverSection(s0); return; }
  const r = Math.random();
  if (r < .46) {
    // 火车组：一到两条轨道停着火车，至少留一条空轨道
    const n = Math.random() < .6 ? 2 : 1;
    let end = s0;
    for (let i = 0; i < n; i++) {
      const lane = lanes[i], cars = 1 + (Math.random() * 3 | 0), ramp = Math.random() < .42;
      const t = addTrain(lane, s0 + rnd(0, 10), cars, { ramp });
      if (ramp) coinLine(lane, t.s0 - RAMP_LEN + 1, Math.floor((t.s1 - t.s0 + RAMP_LEN) / 2.2) - 1, 0, 2.2);
      end = Math.max(end, t.s1);
    }
    const free = lanes[n];
    // 车顶金币：沿坡道爬升后铺满车顶
    for (const c of coins) if (c.y === 0) {
      const o = obstacles.find(o => o.lane === Math.round(c.x / LANE) + 1 && c.s >= o.s0 && c.s <= o.s1);
      c.y = o ? (o.type === 'ramp' ? TRAIN_TOP * (c.s - o.s0) / RAMP_LEN : TRAIN_TOP) + .9 : .9;
    }
    if (n === 2 && Math.random() < .55) {
      const hs = rnd(s0 + 4, Math.max(s0 + 5, end - 6));
      const type = Math.random() < .5 ? 'low' : 'high';
      addHurdle(free, hs, type);
      if (type === 'low') coinArc(free, hs); else coinLine(free, hs - 4.4, 5, .5, 2.2);
    } else {
      coinLine(free, s0 + 2, Math.min(10, Math.floor((end - s0) / 2.2)));
      maybeItem(free, end - 2);
    }
    S.gen = end;
  } else if (r < .84) {
    // 一排栏杆：跨栏（跳）、高栏（翻滚）、路障（只能换道）
    const types = lanes.map(() => pick(['low', 'low', 'high', 'high', 'block', 'none']));
    if (types.every(t => t === 'block' || t === 'none')) types[0] = pick(['low', 'high']);
    if (!types.some(t => t !== 'block')) types[1] = 'none';
    lanes.forEach((lane, i) => {
      const t = types[i];
      if (t !== 'none') addHurdle(lane, s0, t);
      if (i === 0) { if (t === 'low') coinArc(lane, s0); else if (t === 'high') coinLine(lane, s0 - 4.4, 5, .5); else if (t === 'none') coinLine(lane, s0 - 6, 6); }
    });
    if (Math.random() < .5) { const lane = lanes[types.indexOf('none')] ?? lanes[2]; maybeItem(lane, s0 + 6); }
    S.gen = s0 + 1;
  } else {
    // 金币蛇形路线 + 道具
    let lane = lanes[0];
    for (let i = 0; i < 4; i++) {
      coinLine(lane, s0 + i * 9, 4);
      lane = clamp(lane + pick([-1, 1]), 0, 2);
    }
    maybeItem(lane, s0 + 38);
    S.gen = s0 + 36;
  }
}

// 迎面驶来的火车：生成一段 140 米的路，留一条空轨道给它，另外两条轨道错开放火车和栏杆，保证始终有路可走
function genMoverSection(s0) {
  const lane = pick([0, 2]), far = 2 - lane;  // 迎面火车走一侧边道；中间道只放栏杆，随时能躲进去
  const end = s0 + 140;
  let s = s0 + 4;
  while (s < end - 22) {
    const t = addTrain(far, s, 1 + (Math.random() * 2 | 0), { ramp: Math.random() < .35 });
    if (Math.random() < .55) {
      const hs = rnd(t.s0 + 2, t.s1 - 2), type = pick(['low', 'high']);
      addHurdle(1, hs, type);
      if (type === 'low') coinArc(1, hs); else coinLine(1, hs - 4.4, 5, .5);
    } else coinLine(1, t.s0, 6);
    s = t.s1 + rnd(4, 8);
  }
  S.movers.push({ lane, start: s0, end, cars: 1 + (Math.random() * 2 | 0) });
  S.gen = end;
}
function spawnMovers() {
  for (let i = S.movers.length - 1; i >= 0; i--) {
    const m = S.movers[i];
    if (S.s < m.start - 22) continue;
    S.movers.splice(i, 1);
    const len = m.cars * (CAR_LEN + CAR_GAP);
    for (const c of coins) if (Math.abs(c.x - LX(m.lane)) < .5 && c.s > S.s && c.y < 4) c.gone = true;
    addTrain(m.lane, m.end - len, m.cars, { v: rnd(8, 12) });
    FX.sound('check');
    pop('🚆 火车来了！', '#ffd23a');
  }
}

// ---------- 道具 ----------
function givePower(kind) {
  S.power[kind] = POWER_SECS[kind];
  S.powersUsed++;
  FX.sound('power');
  bigPop(POWER_INFO[kind].icon + ' ' + POWER_INFO[kind].name, POWER_INFO[kind].color);
  if (kind === 'jetpack') {
    const p = S.p;
    p.vy = 0; p.rollT = 0;
    // 空中金币航线
    let lane = p.lane, s = S.s + 14;
    const dist = POWER_SECS.jetpack * S.speed - 10;
    for (let d = 0; d < dist; d += 2.4) {
      if (Math.random() < .06) lane = clamp(lane + pick([-1, 1]), 0, 2);
      coins.push({ x: LX(lane), y: JET_Y + .3, s: s + d });
    }
  }
}

// ---------- HUD ----------
function pop(text, color = '#fff') {
  const el = document.createElement('div');
  el.className = 'pop'; el.textContent = text; el.style.color = color;
  el.style.left = '50%'; el.style.top = '44%';
  $('pops').appendChild(el); setTimeout(() => el.remove(), 1100);
}
function bigPop(text, color = '#fff', sub = '') {
  const el = document.createElement('div');
  el.className = 'big-pop'; el.style.color = color;
  el.innerHTML = text + (sub ? `<small>${sub}</small>` : '');
  $('pops').appendChild(el); setTimeout(() => el.remove(), 1600);
}
function flash(color = '#fff', a = .6) {
  const f = $('flash'); f.style.background = color;
  f.animate([{ opacity: a }, { opacity: 0 }], { duration: 450, easing: 'ease-out' });
}
const mult = () => Math.min(6, 1 + Math.floor(S.s / 500)) * (S.power.double ? 2 : 1);
let hudCache = '';
function updateHud() {
  $('hScore').textContent = Math.floor(S.score).toLocaleString();
  $('hCoins').textContent = S.coins;
  const m = mult();
  $('hMult').textContent = '×' + m; $('hMult').classList.toggle('hot', !!S.power.double);
  $('hBoards').innerHTML = (S.boardT > 0 ? `🛹 滑板生效 ${Math.ceil(S.boardT)}s` : `🛹 ×${S.boards}`) + '<kbd>E</kbd>';
  $('hChase').hidden = !(S.chaseT > 0 && state === 'run');
  const key = Object.entries(S.power).map(([k, t]) => k + Math.ceil(t * 4)).join();
  if (key !== hudCache) {
    hudCache = key;
    $('hPowers').innerHTML = Object.entries(S.power).map(([k, t]) => {
      const f = t / POWER_SECS[k], C = 2 * Math.PI * 22;
      return `<div class="pw">${POWER_INFO[k].icon}<svg viewBox="0 0 50 50"><circle cx="25" cy="25" r="22" stroke="rgba(255,255,255,.2)"/><circle cx="25" cy="25" r="22" stroke="${POWER_INFO[k].color}" stroke-dasharray="${C * f} ${C}"/></svg></div>`;
    }).join('');
  }
}

// ---------- 操作 ----------
function move(dir) {
  if (state !== 'run') return;
  const p = S.p, to = clamp(p.lane + dir, 0, 2);
  if (to === p.lane) { stumble(true); return; }
  p.from = p.lane; p.lane = to;
  FX.sound('move');
}
function jump() {
  if (state !== 'run' || S.power.jetpack) return;
  const p = S.p;
  if (!p.onGround) return;
  p.vy = S.power.sneakers ? SUPER_V : JUMP_V;
  p.onGround = false; p.rollT = 0;
  hero.play('Jump', { once: true, fade: .08, timeScale: S.power.sneakers ? .8 : 1.15 });
  FX.sound('jump');
}
function roll() {
  if (state !== 'run' || S.power.jetpack) return;
  const p = S.p;
  if (!p.onGround) p.vy = -34;            // 空中急降
  p.rollT = ROLL_T;
  hero.play('Roll', { once: true, fade: .06, timeScale: 1.55 });
  FX.sound('capture');
}
function useBoard() {
  if (state !== 'run' || S.boardT > 0 || S.boards <= 0) return;
  S.boards--; S.boardT = 30;
  FX.sound('big');
  bigPop('🛹 滑板', '#ff3d7f', '30 秒内撞一次不会输');
}

addEventListener('keydown', e => {
  const k = e.key;
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' '].includes(k)) e.preventDefault();
  if (state === 'menu' && (k === 'Enter' || k === ' ')) { start(); return; }
  if (state === 'over' && (k === 'Enter' || k === ' ')) { start(); return; }
  if (k === 'p' || k === 'P' || k === 'Escape') { togglePause(); return; }
  if (e.repeat) return;
  if (k === 'ArrowLeft' || k === 'a' || k === 'A') move(-1);
  else if (k === 'ArrowRight' || k === 'd' || k === 'D') move(1);
  else if (k === 'ArrowUp' || k === 'w' || k === 'W' || k === ' ') jump();
  else if (k === 'ArrowDown' || k === 's' || k === 'S') roll();
  else if (k === 'e' || k === 'E') useBoard();
});
{
  let sx = 0, sy = 0, st = 0, lastTap = 0, done = false;
  stage.addEventListener('pointerdown', e => { if (e.target.closest('button, a, .hero-card')) return; sx = e.clientX; sy = e.clientY; st = performance.now(); done = false; });
  stage.addEventListener('pointermove', e => {
    if (done || !st || state !== 'run') return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 28) return;
    done = true;
    if (Math.abs(dx) > Math.abs(dy)) move(dx > 0 ? 1 : -1); else if (dy < 0) jump(); else roll();
  });
  stage.addEventListener('pointerup', e => {
    if (!st) return;
    const now = performance.now();
    if (!done && now - st < 250 && state === 'run') { if (now - lastTap < 320) useBoard(); lastTap = now; }
    st = 0;
  });
  $('padL').onclick = () => move(-1); $('padR').onclick = () => move(1);
  $('padJ').onclick = () => jump(); $('padD').onclick = () => roll();
}

// ---------- 碰撞 ----------
function surfaceTop(o, s) {
  return o.type === 'ramp' ? TRAIN_TOP * clamp((s - o.s0) / RAMP_LEN, 0, 1) : o.y1;
}
function crash(o, side) {
  if (S.invT > 0) return;
  if (S.boardT > 0) {
    S.boardT = 0; S.invT = 1.6;
    flash('#4fe3ff', .7); S.shake = .4;
    bigPop('滑板碎了！', '#4fe3ff', '这次没事，继续跑');
    FX.sound('hit');
    if (side) S.p.lane = S.p.from;
    return;
  }
  die(o.type === 'train' ? (o.v ? '被迎面的火车撞飞了' : '一头撞上了火车') : o.type === 'low' ? '被跨栏绊倒了' : o.type === 'high' ? '撞上了高栏（要翻滚）' : '撞上了路障');
}
function stumble(wall) {
  if (S.invT > 0) return;
  const p = S.p;
  if (!wall) p.lane = p.from;
  S.shake = .3;
  FX.sound('hit');
  if (S.chaseT > 0) { die('被巡警抓住了'); return; }
  S.chaseT = 7; S.guardGap = 2.4;
  pop(wall ? '撞墙！' : '擦到车厢了！', '#ff6b6b');
  if (!wall) S.invT = .5;
}
function collide(dt) {
  const p = S.p;
  if (S.power.jetpack) return;
  const h = p.rollT > 0 ? ROLL_H : STAND_H;
  const a = Math.min(p.prevS, S.s) - .3, b = S.s + .3;
  for (const o of obstacles) {
    if (o.s1 < a - 1 || o.s0 > b + 1) continue;
    const dx = Math.abs(p.x - LX(o.lane));
    if (dx > TRAIN_W / 2 + .32) continue;
    if (o.s1 < a || o.s0 > b) continue;
    if (o.walk) {
      const top = surfaceTop(o, clamp(S.s, o.s0, o.s1));
      if (o.type === 'ramp' || p.y >= top - .55) continue;    // 站在坡道或车顶上
    } else if (p.y >= o.y1 - .05 || p.y + h <= o.y0) continue; // 从上方跳过或从下方翻过
    // 侧面擦碰 vs 正面相撞：上一帧已经和障碍并排就算侧面
    const side = dx > .5 && p.prevS + .3 > (o.ps0 ?? o.s0);
    if (side) stumble(false); else crash(o, false);
    if (state !== 'run') return;
  }
}
function groundAt() {
  const p = S.p;
  let g = 0;
  for (const o of obstacles) {
    if (!o.walk || S.s < o.s0 || S.s > o.s1) continue;
    if (Math.abs(p.x - LX(o.lane)) > TRAIN_W / 2 + .1) continue;
    const top = surfaceTop(o, S.s);
    if (top <= p.y + .55 && top > g) g = top;
  }
  return g;
}

// ---------- 主循环 ----------
let last = performance.now(), menuT = 0;
const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpE = new THREE.Euler(), tmpV = new THREE.Vector3(), one = new THREE.Vector3(1, 1, 1);
const camPos = new THREE.Vector3(0, 4, 8), camLook = new THREE.Vector3();

function update(dt) {
  const p = S.p;
  if (state === 'run') {
    S.speed = Math.min(31, 15 + S.s / 140);
    p.prevS = S.s;
    S.s += S.speed * dt;
    S.score += S.speed * dt * mult();
    for (const k of Object.keys(S.power)) if ((S.power[k] -= dt) <= 0) {
      delete S.power[k];
      if (k === 'jetpack') { S.invT = 1.2; p.vy = 0; p.onGround = false; }
    }
    if (S.boardT > 0) S.boardT -= dt;
    if (S.invT > 0) S.invT -= dt;
    if (S.chaseT > 0) S.chaseT -= dt;
    // 生成关卡与迎面火车
    while (S.gen < S.s + 190) genRow();
    spawnMovers();
  }
  // 迎面火车移动
  for (const o of obstacles) {
    o.ps0 = o.s0;
    if (o.v && state !== 'menu') { o.s0 -= o.v * dt; o.s1 -= o.v * dt; o.mesh.position.z = -o.s0; }
  }
  if (state === 'run') {
    // 横向移动
    const tx = LX(p.lane);
    p.x += (tx - p.x) * Math.min(1, dt * 15);
    if (Math.abs(tx - p.x) < .02) { p.x = tx; p.from = p.lane; }
    // 纵向
    if (S.power.jetpack) {
      p.y += (JET_Y - p.y) * Math.min(1, dt * 3); p.vy = 0; p.onGround = false;
    } else {
      p.vy -= GRAV * dt;
      p.y += p.vy * dt;
      const g = groundAt();
      if (p.y <= g) {
        if (!p.onGround && p.vy < -8) { FX.sound('move'); }
        p.y = g; p.vy = 0;
        if (!p.onGround && p.rollT <= 0) hero.play('Run', { fade: .12 });
        p.onGround = true;
      } else if (p.y > g + .08) p.onGround = false;
    }
    if (p.rollT > 0 && (p.rollT -= dt) <= 0) hero.play(p.onGround ? 'Run' : 'Jump', { fade: .1 });
    collide(dt);
    // 金币
    const magnet = S.power.magnet;
    for (const c of coins) {
      if (c.gone) continue;
      const ds = c.s - S.s;
      if (ds < -2) { c.gone = true; continue; }
      if (magnet && ds < 22 && ds > -1) {
        c.x += (p.x - c.x) * Math.min(1, dt * 9); c.y += (p.y + .9 - c.y) * Math.min(1, dt * 9); c.s += (S.s - c.s) * Math.min(1, dt * 6);
      }
      if (Math.abs(c.s - S.s) < .9 && Math.abs(c.x - p.x) < .85 && c.y > p.y - .3 && c.y < p.y + (p.rollT > 0 ? 1.4 : 2.1)) {
        c.gone = true; S.coins++; S.score += 5 * mult();
        spark(c.x, c.y, -c.s, 1);
        FX.sound('coin');
      }
    }
    for (let i = items.length - 1; i >= 0; i--) {
      const it = items[i];
      if (Math.abs(it.s - S.s) < 1 && Math.abs(LX(it.lane) - p.x) < 1 && Math.abs(it.y - (p.y + .9)) < 1.6) {
        givePower(it.kind); spark(LX(it.lane), it.y, -it.s, 3);
        scene.remove(it.mesh); items.splice(i, 1);
      }
    }
  }
  // 清理身后的东西
  for (let i = obstacles.length - 1; i >= 0; i--) if (obstacles[i].s1 < S.s - 20) { scene.remove(obstacles[i].mesh); obstacles.splice(i, 1); }
  for (let i = items.length - 1; i >= 0; i--) if (items[i].s < S.s - 10) { scene.remove(items[i].mesh); items.splice(i, 1); }
  if (coins.length > 600 || (coins.length && coins[0].gone)) for (let i = coins.length - 1; i >= 0; i--) if (coins[i].gone || coins[i].s < S.s - 5) coins.splice(i, 1);

  if (state === 'dead') {
    S.deadT += dt;
    if (S.deadT > 1.6 && state === 'dead') showOver();
  }
}

function syncScene(dt, t) {
  const p = S.p;
  const menu = state === 'menu';
  // 英雄
  hero.root.position.set(p.x, p.y + (S.boardT > 0 ? .16 : 0), -S.s);
  const lean = (LX(p.lane) - p.x) * -.12;
  hero.root.rotation.set(0, Math.PI / 2 + lean, 0);
  hero.update(dt, { speed: menu ? 0 : S.speed / 30, slide: p.rollT > 0 });
  hero.body.visible = !(S.invT > 0 && Math.floor(t * 16) % 2 && state === 'run');
  board.visible = S.boardT > 0 && !S.power.jetpack;
  board.position.set(p.x, p.y + .09, -S.s);
  board.rotation.set(0, lean * 2, 0);
  jetpack.visible = !!S.power.jetpack;
  if (jetpack.visible) {
    const torso = hero.bones.Torso;
    torso.updateWorldMatrix(true, false);
    tmpV.set(0, .35, -.55).applyMatrix4(torso.matrixWorld);
    jetpack.position.copy(tmpV);
    for (const f of jetpack.userData.flames) f.scale.y = .7 + Math.random() * .7;
  }
  // 巡警：撞一次后追到身后，几秒后被甩开
  const chasing = S.chaseT > 0 || state === 'dead' || (state === 'run' && S.s < 30);
  const gapTarget = state === 'dead' ? .9 : chasing ? (state === 'run' && S.s < 30 ? 2.5 + S.s * .3 : S.guardGap) : 16;
  S.guardGapNow = (S.guardGapNow ?? 16) + (gapTarget - (S.guardGapNow ?? 16)) * Math.min(1, dt * (chasing ? 3 : .8));
  guard.root.visible = !menu && S.guardGapNow < 14;
  guard.root.position.set(guard.root.position.x + (p.x - guard.root.position.x) * Math.min(1, dt * 4), 0, -S.s + S.guardGapNow);
  guard.root.rotation.set(0, Math.PI / 2, 0);
  guard.update(dt, { speed: .7 });
  // 障碍物上的金币和道具
  let n = 0;
  const spin = t * 4;
  for (const c of coins) {
    if (c.gone || c.s > S.s + 150 || n >= COIN_CAP) continue;
    tmpE.set(0, spin + c.s * .3, 0); tmpQ.setFromEuler(tmpE);
    tmpM.compose(tmpV.set(c.x, c.y + Math.sin(t * 3 + c.s) * .06, -c.s), tmpQ, one);
    coinMesh.setMatrixAt(n++, tmpM);
  }
  coinMesh.count = n;
  coinMesh.instanceMatrix.needsUpdate = true;
  for (const it of items) it.mesh.position.y = it.y + Math.sin(t * 3 + it.s) * .18;
  for (const s of sparks) if (s.sp.visible) {
    s.t += dt * 2.6;
    if (s.t >= 1) { s.sp.visible = false; continue; }
    s.sp.scale.setScalar(s.size * (.4 + s.t * 1.4)); s.sp.material.opacity = 1 - s.t;
    s.sp.position.y += dt * 1.5;
  }
  // 场景跟随
  followTrack(S.s); updatePoles(S.s); updateBuildings(S.s);
  for (const c of clouds) { const [ox, oy, oz] = c.userData.off; c.position.set(ox, oy, -S.s * .15 - oz); }
  // 镜头
  if (menu) {
    menuT += dt;
    const a = Math.sin(menuT * .35) * .5;
    camPos.set(Math.sin(a) * 4.2 + 1.3, 1.7, -S.s - Math.cos(a) * 4.2);
    camLook.set(1.1, 1.05, -S.s);
  } else {
    const fly = S.power.jetpack ? 1 : 0;
    const ty = Math.max(0, p.y * .75) + 3.3 + fly * 1.2;
    camPos.x += (p.x * .55 - camPos.x) * Math.min(1, dt * 6);
    camPos.y += (ty - camPos.y) * Math.min(1, dt * 4);
    camPos.z = -S.s + 6.8 + fly * 1.5;
    camLook.set(p.x * .7, p.y * .6 + 1.2 + fly * 1.2, -S.s - 8);
  }
  camera.position.copy(camPos);
  if (S.shake > 0) { S.shake -= dt; camera.position.x += (Math.random() - .5) * S.shake; camera.position.y += (Math.random() - .5) * S.shake; }
  camera.lookAt(camLook);
  sun.position.set(p.x + 9, 22, -S.s + (menu ? -14 : 8)); // 菜单里镜头在角色正面，太阳也挪到前面
  sun.target.position.set(p.x, 0, -S.s - 12);
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(.05, (now - last) / 1000);
  last = now;
  if (state === 'load') return;
  if (state !== 'pause') {
    update(dt);
    syncScene(dt, now / 1000);
    if (state === 'run' || state === 'dead') updateHud();
  }
  renderer.render(scene, camera);
}

// ---------- 流程 ----------
function start() {
  reset();
  state = 'run';
  hero.play('Run', { fade: .2 });
  guard.play('Run');
  $('menuOv').hidden = true; $('overOv').hidden = true; $('pauseOv').hidden = true; $('hud').hidden = false;
  S.chaseT = 0;
  bigPop('快跑！', '#ffd23a', '巡警追来了');
  FX.sound('big');
}
function die(reason) {
  if (state !== 'run') return;
  state = 'dead';
  S.reason = reason; S.deadT = 0; S.shake = .6;
  S.p.rollT = 0;
  flash('#ff3b3b', .55);
  hero.play('Death', { once: true, fade: .08 });
  FX.sound('hit');
}
function showOver() {
  state = 'over';
  $('hud').hidden = true;
  const score = Math.floor(S.score);
  let best = 0;
  try { best = +localStorage.getItem('subway-best') || 0; } catch {}
  const isNew = score > best;
  if (isNew) { best = score; try { localStorage.setItem('subway-best', best); } catch {} }
  $('overReason').textContent = S.reason;
  $('overScore').textContent = score.toLocaleString();
  $('overNew').hidden = !isNew;
  $('overDist').textContent = Math.floor(S.s) + ' 米';
  $('overCoins').textContent = S.coins;
  $('overPowers').textContent = S.powersUsed;
  $('overBest').textContent = best.toLocaleString();
  $('overOv').hidden = false;
  FX.sound(isNew ? 'mate' : 'big');
}
function toMenu() {
  reset();
  state = 'menu';
  hero.play('Idle', { fade: .2 });
  $('hud').hidden = true; $('overOv').hidden = true; $('pauseOv').hidden = true; $('menuOv').hidden = false;
  let best = 0;
  try { best = +localStorage.getItem('subway-best') || 0; } catch {}
  $('bestLine').textContent = best ? `最高分 ${best.toLocaleString()}` : '';
}
function togglePause() {
  if (state === 'run') { state = 'pause'; $('pauseOv').hidden = false; }
  else if (state === 'pause') { state = 'run'; $('pauseOv').hidden = true; last = performance.now(); }
}
function renderHeroCards() {
  $('heroCards').innerHTML = Object.entries(HEROES).filter(([k]) => k !== 'guard').map(([k, h]) =>
    `<button class="hero-card${k === heroId ? ' on' : ''}" data-hero="${k}"><b><i style="background:#${h.glow.toString(16).padStart(6, '0')}"></i>${h.name}</b><small>${h.desc}</small></button>`).join('');
}
$('heroCards').addEventListener('click', e => {
  const c = e.target.closest('[data-hero]');
  if (!c) return;
  heroId = c.dataset.hero;
  try { localStorage.setItem('subway-hero', heroId); } catch {}
  hero.setHero(heroId);
  renderHeroCards();
  FX.sound('move');
});
$('btnStart').onclick = start;
$('btnRestart').onclick = start;
$('btnMenu').onclick = toMenu;
$('btnResume').onclick = togglePause;
$('btnQuit').onclick = toMenu;
FX.muteButton($('btnMute'));
addEventListener('blur', () => { if (state === 'run') togglePause(); });

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
resize();

(async function boot() {
  try {
    let pa = 0, pb = 0;
    const prog = () => { $('loadBar').style.width = ((pa + pb) / 2 * 100) + '%'; };
    await Promise.all([loadHeroAssets(v => { pa = v; prog(); }), loadBuildings(v => { pb = v; prog(); })]);
  } catch (e) {
    $('loadTxt').textContent = '模型载入失败：' + e.message + '（刷新重试）';
    return;
  }
  hero = new Hero(); hero.fitAccessories(); hero.setHero(heroId);
  // 白天强光下披风容易过曝：压暗一些、缩小一点，别挡住角色
  hero.capeMat.color.setScalar(.5); hero.cape.scale.set(.8, .85, .8);
  scene.add(hero.root);
  guard = new Hero(); guard.fitAccessories(); guard.setHero('guard');
  guard.capeMat.color.setScalar(.5); guard.cape.scale.set(.8, .85, .8);
  for (const w of Object.values(guard.weapons)) w.visible = false;
  scene.add(guard.root);
  board = makeBoard(); scene.add(board);
  jetpack = makeJetpack(); scene.add(jetpack);
  renderHeroCards();
  $('loadOv').hidden = true;
  toMenu();
  requestAnimationFrame(t => { last = t; frame(t); });
})();

// 调试入口
window.__subway = {
  get S() { return S; }, get state() { return state; }, obstacles, coins, items, start, move, jump, roll, useBoard, givePower, toMenu,
  renderer, scene, camera, update, syncScene,
};
