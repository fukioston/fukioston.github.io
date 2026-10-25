// 跑酷 3D 角色与物件：骑士英雄（Quaternius CC0 模型）、宠物、坐骑、障碍、金币、道具。
import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

export const U = 1 / 50; // 逻辑像素 → 3D 单位

export const HEROES = {
  flame: { name: '赤焰战神', desc: '暗红重甲 · 巨剑 · 火焰披风', armor: 0x5a1414, armorMetal: .75, trim: 0xffb02e, skin: 0x8a5a3a, boots: 0x1a0a06,
           helm: 'Helmet3', helmColor: 0x2a1010, weapon: 'Sword', cape: [0xff3b2f, 0x5e0710], eye: 0xff8a1f, glow: 0xff5a1f },
  ninja: { name: '霓虹忍者', desc: '暗影装甲 · 等离子太刀', armor: 0x14172e, armorMetal: .6, trim: 0x33f0ff, skin: 0x6b4a36, boots: 0x05060c,
           helm: 'Helmet2', helmColor: 0x101428, weapon: 'Katana', cape: [0x1fd6ff, 0x08204a], eye: 0x33f0ff, glow: 0x33f0ff },
  frost: { name: '玄冰帝王', desc: '银白冰甲 · 寒霜长剑 · 冰晶王冠', armor: 0xc9dcf2, armorMetal: .9, trim: 0x7af7ff, skin: 0xd9b89a, boots: 0x3a4d6b,
           helm: 'Helmet1', helmColor: 0xe6f2ff, weapon: 'ShortSword', cape: [0x4d8dff, 0x0b1f66], eye: 0xb2ff59, glow: 0x7af7ff, crown: true },
};
export const PETS = { dragon: { name: '小火龙', desc: '会喷小火苗' }, fox: { name: '九尾灵狐', desc: '九条发光尾巴' }, panda: { name: '胖达', desc: '竹蜻蜓飞行' } };
export const MOUNTS = {
  wolf: { name: '机甲战狼', desc: '全程陪跑 · 能量尾焰', icon: '🐺', color: '#33f0ff' },
  horse: { name: '烈焰战马', desc: '全程陪跑 · 火焰鬃毛', icon: '🐴', color: '#ff7b2a' },
  drake: { name: '雷霆飞龙', desc: '全程陪跑 · 闪电双翼', icon: '🐉', color: '#b26bff' },
};

const std = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: .45, metalness: .3, ...o });
const glowMat = (color, intensity = 2.5) => new THREE.MeshStandardMaterial({ color: 0x000000, emissive: color, emissiveIntensity: intensity, toneMapped: false });

// 偶发的网络中断时重试几次
export async function withRetry(fn, times = 4) {
  for (let i = 0; ; i++) {
    try { return await fn(); }
    catch (e) { if (i >= times - 1) throw e; await new Promise(r => setTimeout(r, 300 * (i + 1))); }
  }
}

// 释放网格占用的显存；标了 keep 的共享几何体、材质不动
export function keep(...things) { for (const t of things) if (t) t.userData.keep = true; return things[0]; }
export function disposeTree(root) {
  root.traverse(o => {
    if (o.geometry && !o.geometry.userData.keep) o.geometry.dispose();
    if (o.material) for (const m of [].concat(o.material)) if (!m.userData.keep) m.dispose();
  });
}

// ---------- 英雄 ----------
let heroAssets = null;
export async function loadHeroAssets(onProgress) {
  const loader = new FBXLoader();
  const names = ['KnightCharacter', 'Helmet1', 'Helmet2', 'Helmet3', 'ShoulderPads', 'Sword', 'Katana', 'ShortSword'];
  const out = {};
  let done = 0;
  await Promise.all(names.map(async n => {
    out[n] = await withRetry(() => loader.loadAsync(`assets/knight/${n}.fbx`));
    onProgress && onProgress(++done / names.length);
  }));
  heroAssets = out;
  return out;
}

export class Hero {
  constructor() {
    const A = heroAssets;
    this.root = new THREE.Group();
    this.body = new THREE.Group();           // 用于空翻、死亡旋转
    this.root.add(this.body);
    this.model = SkeletonUtils.clone(A.KnightCharacter);
    this.model.scale.setScalar(.003);
    this.model.rotation.y = Math.PI / 2;     // 面朝 +x
    this.body.add(this.model);
    this.bones = {};
    this.mats = {};
    this.model.traverse(o => {
      if (o.isBone) this.bones[o.name] = o;
      if (o.isMesh) {
        o.castShadow = true;
        const mats = [].concat(o.material).map(m => {
          const nm = std(m.color, { name: m.name });
          this.mats[m.name] = nm;
          return nm;
        });
        o.material = mats.length > 1 ? mats : mats[0];
      }
    });
    this.mixer = new THREE.AnimationMixer(this.model);
    this.actions = {};
    for (const clip of A.KnightCharacter.animations) {
      const short = clip.name.split('|').pop();
      this.actions[short] = this.mixer.clipAction(clip);
    }
    this.current = null;

    // 头盔、肩甲、武器挂到骨骼上
    this.helmets = {};
    for (const h of ['Helmet1', 'Helmet2', 'Helmet3']) {
      const m = A[h].clone();
      m.traverse(o => { if (o.isMesh) { o.castShadow = true; o.material = [].concat(o.material).map(x => std(x.color, { metalness: .85, roughness: .3 })); if (o.material.length === 1) o.material = o.material[0]; } });
      m.visible = false;
      this.bones.Head.add(m);
      this.helmets[h] = m;
    }
    this.pads = A.ShoulderPads.clone();
    this.pads.traverse(o => { if (o.isMesh) { o.castShadow = true; o.material = std(0x777777, { metalness: .9, roughness: .25 }); } });
    this.bones.Torso.add(this.pads);
    this.weapons = {};
    for (const w of ['Sword', 'Katana', 'ShortSword']) {
      const m = A[w].clone();
      m.traverse(o => { if (o.isMesh) { o.castShadow = true; o.material = [].concat(o.material).map(x => std(x.color, { metalness: .9, roughness: .2 })); if (o.material.length === 1) o.material = o.material[0]; } });
      m.visible = false;
      this.bones.PalmR.add(m);
      this.weapons[w] = m;
    }
    this.blade = glowMat(0xffffff, 3);
    // 发光眼睛
    this.eyes = new THREE.Mesh(new THREE.BoxGeometry(.32, .07, .06), glowMat(0xff8a1f, 6));
    this.bones.Head.add(this.eyes);
    // 胸口能量核心
    this.core = new THREE.Mesh(new THREE.OctahedronGeometry(.13), glowMat(0xff5a1f, 4));
    this.bones.Torso.add(this.core);
    // 冰晶王冠
    this.crown = new THREE.Group();
    for (let i = 0; i < 7; i++) {
      const c = new THREE.Mesh(new THREE.ConeGeometry(.07, .32 + (i % 2) * .2, 5), glowMat(0x7af7ff, 2.2));
      const a = (i / 7) * Math.PI * 2;
      c.position.set(Math.cos(a) * .42, .1 + (i % 2) * .1, Math.sin(a) * .42);
      this.crown.add(c);
    }
    this.bones.Head.add(this.crown);
    // 披风：挂在躯干骨骼，每帧更新顶点做飘动
    const capeGeo = new THREE.PlaneGeometry(.62, 1.05, 4, 10);
    capeGeo.translate(0, -.525, 0);
    this.capeBase = capeGeo.attributes.position.array.slice();
    this.capeMat = new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: .7, metalness: .1, vertexColors: true });
    const colors = new Float32Array(capeGeo.attributes.position.count * 3);
    capeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.cape = new THREE.Mesh(capeGeo, this.capeMat);
    this.cape.castShadow = true;
    this.capeAnchor = new THREE.Object3D();
    this.root.add(this.capeAnchor);
    this.capeAnchor.add(this.cape);
    this.t = 0;
    this.ridePose = [1.35, -1.5, 'x']; // 骑乘：大腿前抬、小腿后收
    this.setHero('flame');
  }

  setHero(id) {
    const H = HEROES[id];
    this.id = id;
    this.H = H;
    if (this.mats.Armor) { this.mats.Armor.color.setHex(H.armor); this.mats.Armor.metalness = H.armorMetal; this.mats.Armor.roughness = .32; }
    if (this.mats.Skin) this.mats.Skin.color.setHex(H.skin);
    if (this.mats.Boots) this.mats.Boots.color.setHex(H.boots);
    for (const [k, m] of Object.entries(this.helmets)) {
      m.visible = k === H.helm;
      m.traverse(o => { if (o.isMesh) [].concat(o.material).forEach((x, i) => { x.color.setHex(i === 0 ? H.helmColor : H.trim); x.emissive.setHex(i === 0 ? 0 : H.trim); x.emissiveIntensity = i === 0 ? 0 : .5; }); });
    }
    this.pads.traverse(o => { if (o.isMesh) { o.material.color.setHex(H.trim); o.material.emissive.setHex(H.trim); o.material.emissiveIntensity = .25; } });
    for (const [k, m] of Object.entries(this.weapons)) {
      m.visible = k === H.weapon;
      m.traverse(o => { if (o.isMesh) [].concat(o.material).forEach((x, i) => { if (i === 0) { x.emissive.setHex(H.glow); x.emissiveIntensity = .9; } }); });
    }
    this.eyes.material.emissive.setHex(H.eye);
    this.core.material.emissive.setHex(H.glow);
    this.crown.visible = !!H.crown;
    const c0 = new THREE.Color(H.cape[0]), c1 = new THREE.Color(H.cape[1]);
    const pos = this.cape.geometry.attributes.position, col = this.cape.geometry.attributes.color;
    for (let i = 0; i < pos.count; i++) {
      const k = Math.min(1, -this.capeBase[i * 3 + 1] / 1.05);
      const c = c0.clone().lerp(c1, k);
      col.setXYZ(i, c.r, c.g, c.b);
    }
    col.needsUpdate = true;
  }

  // 位置调校：骨骼内部有 100 倍缩放，1 个骨骼单位约等于 0.26 个世界单位；+Y 沿骨骼向上，+Z 朝前
  fitAccessories() {
    for (const m of Object.values(this.helmets)) { m.position.set(0, .48, .04); m.rotation.set(0, 0, 0); m.scale.setScalar(.0085); }
    this.eyes.position.set(0, .5, .56);
    this.crown.position.set(0, 1.05, 0);
    this.pads.position.set(0, .5, 0); this.pads.scale.setScalar(.0078);
    this.core.position.set(0, .45, .52);
    for (const m of Object.values(this.weapons)) { m.position.set(0, .1, 0); m.rotation.set(0, 0, 0); m.scale.setScalar(.0085); }
  }

  play(name, { once = false, fade = .15, timeScale = 1 } = {}) {
    const a = this.actions[name];
    if (!a) return;
    a.timeScale = timeScale;
    if (this.current === a && !once) return;
    a.reset();
    a.setLoop(once ? THREE.LoopOnce : THREE.LoopRepeat, Infinity);
    a.clampWhenFinished = once;
    a.enabled = true;
    if (this.current && this.current !== a) a.crossFadeFrom(this.current, fade, false);
    a.play();
    this.current = a;
  }

  update(dt, { speed = .5, ride = false, slide = false }) {
    this.t += dt;
    this.mixer.update(dt);
    if (ride) { // 骑乘：大腿前抬，小腿下垂
      const b = this.bones;
      const [ux, lx, axis] = this.ridePose;
      for (const s of ['L', 'R']) {
        if (b['UpperLeg' + s]) b['UpperLeg' + s].rotation[axis] = ux;
        if (b['LowerLeg' + s]) b['LowerLeg' + s].rotation[axis] = lx;
      }
    }
    // 披风跟随躯干并飘动
    const torso = this.bones.Torso;
    if (torso) {
      torso.updateWorldMatrix(true, false);
      const p = new THREE.Vector3(0, .62, -.42).applyMatrix4(torso.matrixWorld);
      this.root.worldToLocal(p);
      this.capeAnchor.position.copy(p);
    }
    this.capeAnchor.rotation.set(0, 0, 0);
    const pos = this.cape.geometry.attributes.position, base = this.capeBase;
    const flow = .25 + speed * .9 + (slide ? .5 : 0);
    for (let i = 0; i < pos.count; i++) {
      const x0 = base[i * 3], y0 = base[i * 3 + 1];
      const k = -y0 / 1.05; // 0 顶部 → 1 底部
      const wave = Math.sin(this.t * 13 - k * 5 + x0 * 3) * .07 * k;
      const back = -k * k * flow * .75 - .12 * k;
      pos.setXYZ(i, back + wave * .5, y0 * (1 - flow * .28 * k) + Math.abs(back) * .2, x0 + wave);
    }
    pos.needsUpdate = true;
    this.cape.geometry.computeVertexNormals();
  }
}

// ---------- 宠物（低多边形） ----------
export function makePet(type) {
  const g = new THREE.Group();
  const parts = { wings: [], tails: [], prop: null, flame: null };
  const flat = { flatShading: true };
  if (type === 'dragon') {
    const body = new THREE.Mesh(new THREE.SphereGeometry(.17, 10, 8), std(0xff7a2a, flat));
    body.scale.set(1.2, 1, 1); g.add(body);
    const belly = new THREE.Mesh(new THREE.SphereGeometry(.12, 8, 6), std(0xffe08a, flat)); belly.position.set(.06, -.04, 0); g.add(belly);
    const head = new THREE.Mesh(new THREE.SphereGeometry(.15, 10, 8), std(0xff8a3a, flat)); head.position.set(.2, .14, 0); g.add(head);
    for (const z of [-.06, .06]) {
      const horn = new THREE.Mesh(new THREE.ConeGeometry(.03, .12, 5), std(0xfff4c2)); horn.position.set(.16, .3, z); horn.rotation.z = .4; g.add(horn);
      const eye = new THREE.Mesh(new THREE.SphereGeometry(.035, 8, 6), std(0x111111, { roughness: .1 })); eye.position.set(.33, .17, z * 1.2); g.add(eye);
      const hl = new THREE.Mesh(new THREE.SphereGeometry(.012, 6, 4), glowMat(0xffffff, 2)); hl.position.set(.36, .19, z * 1.2); g.add(hl);
      const wing = new THREE.Mesh(new THREE.ConeGeometry(.12, .3, 3), std(0xffa060, { ...flat, side: THREE.DoubleSide }));
      wing.geometry.translate(0, .15, 0); wing.scale.set(1, 1, .15); wing.position.set(-.02, .1, z * 1.6); wing.rotation.x = z > 0 ? .8 : -.8;
      g.add(wing); parts.wings.push(wing);
    }
    const tail = new THREE.Mesh(new THREE.ConeGeometry(.06, .3, 6), std(0xff7a2a, flat)); tail.position.set(-.26, -.02, 0); tail.rotation.z = Math.PI / 2 + .4; g.add(tail);
    parts.flame = new THREE.Mesh(new THREE.SphereGeometry(.06, 8, 6), glowMat(0xffb000, 4)); parts.flame.position.set(-.42, .04, 0); g.add(parts.flame);
  } else if (type === 'fox') {
    const body = new THREE.Mesh(new THREE.SphereGeometry(.16, 10, 8), std(0xffffff, flat)); body.scale.set(1.3, .95, 1); g.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(.14, 10, 8), std(0xffffff, flat)); head.position.set(.2, .14, 0); g.add(head);
    const snout = new THREE.Mesh(new THREE.ConeGeometry(.06, .14, 6), std(0xffffff, flat)); snout.position.set(.34, .11, 0); snout.rotation.z = -Math.PI / 2; g.add(snout);
    const nose = new THREE.Mesh(new THREE.SphereGeometry(.02, 6, 4), std(0x222222)); nose.position.set(.41, .11, 0); g.add(nose);
    for (const z of [-.07, .07]) {
      const ear = new THREE.Mesh(new THREE.ConeGeometry(.05, .14, 4), std(0xffffff, flat)); ear.position.set(.16, .3, z); g.add(ear);
      const inner = new THREE.Mesh(new THREE.ConeGeometry(.025, .08, 4), glowMat(0xff7ad9, 1.5)); inner.position.set(.175, .29, z); g.add(inner);
      const eye = new THREE.Mesh(new THREE.BoxGeometry(.03, .01, .04), std(0x3a2440)); eye.position.set(.3, .18, z); g.add(eye);
    }
    for (let i = 0; i < 9; i++) {
      const tail = new THREE.Group();
      const t = new THREE.Mesh(new THREE.ConeGeometry(.055, .42, 6), std(0xffffff, flat)); t.position.y = .21; tail.add(t);
      const tip = new THREE.Mesh(new THREE.ConeGeometry(.035, .12, 6), glowMat(0xff7ad9, 2.5)); tip.position.y = .44; tail.add(tip);
      tail.position.set(-.18, .02, 0);
      tail.userData.base = -2.1 + i * .17;
      tail.rotation.z = tail.userData.base + Math.PI / 2;
      tail.rotation.x = (i - 4) * .12;
      g.add(tail); parts.tails.push(tail);
    }
  } else {
    const white = std(0xffffff, flat), black = std(0x1d1d24, flat);
    const body = new THREE.Mesh(new THREE.SphereGeometry(.17, 10, 8), white); body.position.y = -.08; g.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(.17, 12, 10), white); head.position.y = .16; g.add(head);
    for (const z of [-.11, .11]) {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(.06, 8, 6), black); ear.position.set(-.02, .31, z); g.add(ear);
      const patch = new THREE.Mesh(new THREE.SphereGeometry(.05, 8, 6), black); patch.scale.set(.6, 1, 1); patch.position.set(.14, .18, z * .55); g.add(patch);
      const eye = new THREE.Mesh(new THREE.SphereGeometry(.015, 6, 4), glowMat(0xffffff, 1.5)); eye.position.set(.17, .19, z * .55); g.add(eye);
      const arm = new THREE.Mesh(new THREE.SphereGeometry(.06, 8, 6), black); arm.position.set(.08, -.05, z * 1.4); g.add(arm);
    }
    const nose = new THREE.Mesh(new THREE.SphereGeometry(.025, 6, 4), black); nose.position.set(.17, .12, 0); g.add(nose);
    const stick = new THREE.Mesh(new THREE.CylinderGeometry(.01, .01, .12), std(0x555555)); stick.position.y = .38; g.add(stick);
    const prop = new THREE.Group();
    for (const r of [0, Math.PI / 2]) { const b = new THREE.Mesh(new THREE.BoxGeometry(.4, .01, .05), glowMat(0xff5b9b, 1.6)); b.rotation.y = r; prop.add(b); }
    prop.position.y = .44; g.add(prop); parts.prop = prop;
  }
  g.traverse(o => { if (o.isMesh) o.castShadow = true; });
  g.userData.parts = parts;
  g.userData.type = type;
  return g;
}
export function animatePet(g, t) {
  const p = g.userData.parts;
  for (const w of p.wings) w.rotation.x = (w.position.z > 0 ? 1 : -1) * (.5 + Math.sin(t * 18) * .6);
  p.tails.forEach((tl, i) => { tl.rotation.z = tl.userData.base + Math.PI / 2 + Math.sin(t * 4 + i * .6) * .12; });
  if (p.prop) p.prop.rotation.y = t * 30;
  if (p.flame) { const s = .8 + Math.random() * .6; p.flame.scale.setScalar(s); }
}

// ---------- 坐骑（低多边形 + 程序动画） ----------
export function makeMount(type) {
  const g = new THREE.Group();
  const legs = [], extra = { flames: [], wings: [], tail: null, bolt: null };
  const flat = { flatShading: true };
  const box = (w, h, d, mat, x, y, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); g.add(m); return m; };
  const leg = (x, z, mat, hoof, len = .42) => {
    const hip = new THREE.Group(); hip.position.set(x, .78, z); g.add(hip);
    const upper = new THREE.Mesh(new THREE.BoxGeometry(.13, len, .13), mat); upper.position.y = -len / 2; hip.add(upper);
    const knee = new THREE.Group(); knee.position.y = -len; hip.add(knee);
    const lower = new THREE.Mesh(new THREE.BoxGeometry(.1, len, .1), mat); lower.position.y = -len / 2; knee.add(lower);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(.16, .07, .14), hoof); foot.position.set(.03, -len, 0); knee.add(foot);
    legs.push({ hip, knee, phase: legs.length });
  };
  if (type === 'wolf') {
    const steel = std(0xa8b6d6, { metalness: .85, roughness: .28, ...flat }), dark = std(0x4a5470, { metalness: .8, roughness: .35, ...flat });
    const cyan = glowMat(0x33f0ff, 3);
    box(1.05, .38, .46, steel, 0, 1.0, 0);
    box(.5, .32, .5, dark, .42, 1.05, 0);
    box(.9, .06, .48, cyan, 0, 1.2, 0).scale.set(1, .5, 1);
    const neck = box(.3, .42, .34, steel, .62, 1.3, 0); neck.rotation.z = -.6;
    const head = box(.42, .26, .3, steel, .86, 1.5, 0);
    box(.26, .14, .22, dark, 1.08, 1.43, 0);
    for (const z of [-.1, .1]) { const ear = new THREE.Mesh(new THREE.ConeGeometry(.07, .2, 4), dark); ear.position.set(.8, 1.7, z); g.add(ear); box(.06, .04, .02, glowMat(0x33f0ff, 6), 1.0, 1.55, z * 1.5); }
    box(.12, .12, .5, cyan, -.1, 1.02, 0).scale.set(1, 1, 1.02);
    const tail = new THREE.Mesh(new THREE.ConeGeometry(.12, .7, 6), glowMat(0x33f0ff, 2.5)); tail.position.set(-.82, 1.12, 0); tail.rotation.z = Math.PI / 2 + .3; g.add(tail); extra.tail = tail;
    for (const [x, z] of [[.4, -.17], [.4, .17], [-.38, -.17], [-.38, .17]]) leg(x, z, dark, steel);
    g.userData.saddle = new THREE.Vector3(-.05, 1.22, 0);
  } else if (type === 'horse') {
    const coat = std(0xc23a22, { roughness: .45, ...flat }), dark = std(0x5a160c, flat), gold = std(0xffd27a, { metalness: .9, roughness: .25 });
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(.24, .75, 4, 8), coat); body.rotation.z = Math.PI / 2; body.position.y = 1.02; g.add(body);
    const neck = new THREE.Mesh(new THREE.CapsuleGeometry(.14, .42, 4, 8), coat); neck.position.set(.5, 1.36, 0); neck.rotation.z = -.7; g.add(neck);
    const head = new THREE.Mesh(new THREE.CapsuleGeometry(.11, .3, 4, 8), coat); head.position.set(.74, 1.58, 0); head.rotation.z = -1.9; g.add(head);
    box(.3, .05, .02, gold, .72, 1.6, .1); box(.3, .05, .02, gold, .72, 1.6, -.1);
    for (const z of [-.08, .08]) { const ear = new THREE.Mesh(new THREE.ConeGeometry(.04, .14, 4), dark); ear.position.set(.64, 1.78, z); g.add(ear); box(.04, .03, .02, glowMat(0xffe066, 5), .8, 1.66, z * 1.4); }
    box(.42, .08, .5, dark, -.02, 1.27, 0); box(.44, .03, .52, gold, -.02, 1.22, 0);
    for (let i = 0; i < 7; i++) { const f = new THREE.Mesh(new THREE.ConeGeometry(.07, .3, 5), glowMat(i % 2 ? 0xffb020 : 0xff5a10, 3)); f.position.set(.38 + i * .055, 1.42 + i * .07, 0); f.rotation.z = .9; g.add(f); extra.flames.push(f); }
    for (let i = 0; i < 4; i++) { const f = new THREE.Mesh(new THREE.ConeGeometry(.09, .45, 5), glowMat(i % 2 ? 0xffb020 : 0xff5a10, 3)); f.position.set(-.55 - i * .08, 1.1 - i * .04, 0); f.rotation.z = Math.PI / 2 + .5; g.add(f); extra.flames.push(f); }
    for (const [x, z] of [[.36, -.13], [.36, .13], [-.36, -.13], [-.36, .13]]) leg(x, z, coat, glowMat(0xffb020, 2.5));
    g.userData.saddle = new THREE.Vector3(-.02, 1.3, 0);
  } else {
    const scale = std(0x7a5cf0, { roughness: .3, metalness: .45, ...flat }), dark = std(0x3a2680, flat), belly = std(0xdcd0ff, flat);
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(.26, .7, 4, 8), scale); body.rotation.z = Math.PI / 2; body.position.y = 1.1; g.add(body);
    const bel = new THREE.Mesh(new THREE.CapsuleGeometry(.18, .55, 4, 8), belly); bel.rotation.z = Math.PI / 2; bel.position.set(0, .98, 0); g.add(bel);
    const neck = new THREE.Mesh(new THREE.CapsuleGeometry(.13, .45, 4, 8), scale); neck.position.set(.52, 1.42, 0); neck.rotation.z = -.8; g.add(neck);
    const head = box(.42, .2, .26, scale, .82, 1.68, 0);
    for (const z of [-.08, .08]) {
      const horn = new THREE.Mesh(new THREE.ConeGeometry(.04, .3, 5), std(0xe8e0ff)); horn.position.set(.66, 1.86, z); horn.rotation.z = .9; g.add(horn);
      box(.05, .03, .02, glowMat(0xffe45c, 6), .96, 1.72, z * 1.5);
    }
    for (let i = 0; i < 6; i++) { const sp = new THREE.Mesh(new THREE.ConeGeometry(.05, .14, 4), dark); sp.position.set(-.35 + i * .13, 1.38, 0); g.add(sp); }
    const tail = new THREE.Mesh(new THREE.ConeGeometry(.14, 1.1, 6), scale); tail.position.set(-.9, 1.05, 0); tail.rotation.z = Math.PI / 2 + .25; g.add(tail); extra.tail = tail;
    for (const side of [-1, 1]) {
      const wing = new THREE.Group(); wing.position.set(0, 1.32, side * .2); g.add(wing);
      const mem = new THREE.Mesh(new THREE.BufferGeometry(), std(0x8a5cff, { side: THREE.DoubleSide, emissive: 0x3a1a80, emissiveIntensity: .6, ...flat }));
      mem.geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, -.3, .05, side * 1.1, -.75, -.1, side * .9, -.5, -.05, side * .4, 0, 0, 0, -.75, -.1, side * .9], 3));
      mem.geometry.computeVertexNormals();
      wing.add(mem);
      wing.userData.side = side;
      extra.wings.push(wing);
    }
    for (const [x, z] of [[.3, -.14], [.3, .14], [-.3, -.14], [-.3, .14]]) leg(x, z, dark, dark, .28);
    legs.forEach(l => { l.hip.position.y = .9; });
    const bolt = new THREE.Mesh(new THREE.BoxGeometry(.03, .7, .03), glowMat(0xd6c8ff, 6)); bolt.visible = false; g.add(bolt); extra.bolt = bolt;
    g.userData.saddle = new THREE.Vector3(-.05, 1.4, 0);
  }
  g.traverse(o => { if (o.isMesh) o.castShadow = true; });
  g.userData.legs = legs;
  g.userData.extra = extra;
  g.userData.type = type;
  return g;
}
export function animateMount(g, t, ph, onGround, crouch = 0) {
  const { legs, extra, type } = g.userData;
  const lerp = (a, b, k) => a + (b - a) * k;
  for (const l of legs) {
    const off = [0, Math.PI, Math.PI * .5, Math.PI * 1.5][l.phase];
    const front = l.phase < 2;
    let a = onGround && type !== 'drake' ? Math.sin(ph + off) * .7 : type === 'drake' ? .9 : (front ? -.8 : .8);
    let k = -Math.max(0, Math.sin(ph + off + 1.2)) * (onGround ? 1.1 : .2) - (type === 'drake' ? 1.2 : 0);
    // 趴下：前腿向前伸、后腿向后伸，贴地
    a = lerp(a, front ? 1.45 : -1.45, crouch);
    k = lerp(k, 0, crouch);
    l.hip.rotation.z = a;
    l.knee.rotation.z = k;
  }
  g.position.y -= crouch * (type === 'drake' ? .5 : .62);
  if (extra.tail) extra.tail.rotation.y = Math.sin(t * 6) * .25;
  for (const f of extra.flames) f.scale.set(1, .7 + Math.random() * .7, 1);
  for (const w of extra.wings) w.rotation.x = w.userData.side * Math.sin(t * 8) * .55;
  if (extra.bolt) {
    extra.bolt.visible = Math.random() < .12;
    if (extra.bolt.visible) { extra.bolt.position.set(Math.random() * .8 - .4, .6, (Math.random() - .5) * .6); extra.bolt.rotation.z = Math.random() - .5; }
  }
  if (type === 'drake') g.position.y += Math.sin(t * 4) * .06 * (1 - crouch);
}

// ---------- 贴图工具 ----------
export function canvasTex(w, h, draw, { repeat = null } = {}) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  return t;
}
export const hazardTex = () => canvasTex(128, 128, (g, w, h) => {
  g.fillStyle = '#ffcc00'; g.fillRect(0, 0, w, h);
  g.fillStyle = '#121212';
  for (let k = -4; k < 8; k += 2) { g.beginPath(); g.moveTo(k * 22, h); g.lineTo((k + 1) * 22, h); g.lineTo((k + 1) * 22 + h, 0); g.lineTo(k * 22 + h, 0); g.fill(); }
  g.strokeStyle = '#000'; g.lineWidth = 10; g.strokeRect(0, 0, w, h);
});

// ---------- 障碍 ----------
const OBS = {};
function obsMaterials() {
  if (OBS.ready) return OBS;
  OBS.hazard = std(0xffffff, { map: hazardTex(), roughness: .55, metalness: .2 });
  OBS.dark = std(0x16161c, { metalness: .8, roughness: .35 });
  OBS.metal = std(0xc8ccd6, { metalness: 1, roughness: .18 });
  OBS.red = glowMat(0xff2030, 4);
  OBS.white = glowMat(0xffffff, 3);
  OBS.warnFace = new THREE.MeshBasicMaterial({ map: canvasTex(128, 128, (g) => {
    g.fillStyle = '#111'; g.fillRect(16, 16, 96, 96);
    g.fillStyle = '#ff2a2a'; g.font = '900 92px -apple-system, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('!', 64, 70);
  }), transparent: true, toneMapped: false });
  for (const k of ['hazard', 'dark', 'metal', 'red', 'white', 'warnFace']) keep(OBS[k]);
  OBS.ready = true;
  return OBS;
}
export function makeObstacle(o) {
  const M = obsMaterials(), g = new THREE.Group();
  const w = o.w * U, h = o.h * U;
  if (o.type === 'crate') {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, w), M.hazard); b.position.y = h / 2; g.add(b);
    const face = new THREE.Mesh(new THREE.PlaneGeometry(w * .62, h * .62), M.warnFace); face.position.set(0, h / 2, w / 2 + .002); g.add(face);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(b.geometry), new THREE.LineBasicMaterial({ color: 0xffffff }));
    edges.position.copy(b.position); edges.scale.setScalar(1.02); g.add(edges);
    g.userData.pulse = face;
  } else if (o.type === 'spikes') {
    const base = new THREE.Mesh(new THREE.BoxGeometry(w + .1, .1, 1.1), M.hazard); base.position.y = .05; g.add(base);
    const n = Math.round(o.w / 20);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
      const c = new THREE.Mesh(new THREE.ConeGeometry(.12, h + .1, 6), M.metal); c.position.set(-w / 2 + (i + .5) * w / n, (h + .1) / 2 + .08, (k - 1) * .32); g.add(c);
      const tip = new THREE.Mesh(new THREE.ConeGeometry(.045, .14, 6), M.red); tip.position.set(c.position.x, h + .14, c.position.z); g.add(tip);
    }
  } else if (o.type === 'bar') { // 激光栅栏：后方立柱 + 横向激光
    const top = (GROUND_PX - o.y) * U, bottom = (GROUND_PX - o.y - o.h) * U;
    for (const dx of [-w / 2, w / 2]) {
      const post = new THREE.Mesh(new THREE.BoxGeometry(.18, top + .5, .18), M.hazard); post.position.set(dx, (top + .5) / 2, -.75); g.add(post);
      const arm = new THREE.Mesh(new THREE.BoxGeometry(.12, top - bottom, .9), M.dark); arm.position.set(dx, (top + bottom) / 2, -.3); g.add(arm);
    }
    const cap = new THREE.Mesh(new THREE.BoxGeometry(w + .3, .14, .3), M.hazard); cap.position.set(0, top + .5, -.75); g.add(cap);
    g.userData.beams = [];
    for (let k = 0; k < 5; k++) {
      const y = bottom + .1 + k * (top - bottom - .2) / 4;
      for (const z of [-.25, .2]) {
        const beam = new THREE.Mesh(new THREE.CylinderGeometry(.035, .035, w, 8), M.red); beam.rotation.z = Math.PI / 2; beam.position.set(0, y, z); g.add(beam);
        const core = new THREE.Mesh(new THREE.CylinderGeometry(.012, .012, w, 6), M.white); core.rotation.z = Math.PI / 2; core.position.set(0, y, z); g.add(core);
        g.userData.beams.push(beam);
      }
    }
  } else if (o.type === 'bird') { // 无人机
    const body = new THREE.Mesh(new THREE.BoxGeometry(w * .9, h * .7, .5), M.hazard); g.add(body);
    const eye = new THREE.Mesh(new THREE.SphereGeometry(.09, 10, 8), M.red); eye.position.set(-w * .45, 0, .1); g.add(eye);
    g.userData.rotors = [];
    for (const [x, z] of [[-.4, -.35], [.4, -.35], [-.4, .35], [.4, .35]]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(.05, .05, .05), M.dark); arm.position.set(x, .12, z); g.add(arm);
      const rotor = new THREE.Mesh(new THREE.BoxGeometry(.5, .015, .05), M.white); rotor.position.set(x, .2, z); g.add(rotor);
      g.userData.rotors.push(rotor);
    }
    g.position.y = 0;
  }
  g.traverse(m => { if (m.isMesh) { m.castShadow = true; } });
  return g;
}
let GROUND_PX = 450;
export function setGroundPx(v) { GROUND_PX = v; }

// ---------- 道具 ----------
const ICON = { magnet: '🧲', shield: '🛡️', boost: '🚀', double: '✨' };
const iconCache = {};
export function makeItem(type, mountIcon, color) {
  const g = new THREE.Group();
  const col = new THREE.Color(color);
  const orb = new THREE.Mesh(new THREE.IcosahedronGeometry(.38, 2), new THREE.MeshStandardMaterial({ color: 0x223355, transparent: true, opacity: .55, roughness: .05, metalness: .6, emissive: col, emissiveIntensity: .25 }));
  g.add(orb);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(.5, .022, 8, 48), glowMat(col, 1.6)); g.add(ring);
  const ring2 = new THREE.Mesh(new THREE.TorusGeometry(.44, .014, 8, 48), glowMat(0x9fe8ff, 1.2)); ring2.rotation.x = Math.PI / 2; g.add(ring2);
  const icon = type === 'mount' ? mountIcon : ICON[type];
  if (!iconCache[icon]) iconCache[icon] = canvasTex(128, 128, (c) => { c.font = '96px "Apple Color Emoji", "Segoe UI Emoji", sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(icon, 64, 72); });
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: iconCache[icon], toneMapped: false }));
  sp.scale.setScalar(.62); sp.renderOrder = 2; sp.material.depthTest = false; g.add(sp);
  g.scale.setScalar(1.25);
  g.userData.ring = ring; g.userData.ring2 = ring2;
  return g;
}
