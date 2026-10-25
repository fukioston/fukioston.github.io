// 跑酷 3D 场景：天空、复古太阳、远景天际线、中景建筑（Kenney CC0）、熔岩与冰原景物、地面与深渊、超级奖励关布景。
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { U, canvasTex, hazardTex, withRetry, keep, disposeTree } from './runner3d-actors.js';

const BUILDINGS = ['building-a', 'building-b', 'building-c', 'building-d', 'building-e', 'building-f', 'building-g', 'building-h', 'building-i',
  'building-j', 'building-k', 'building-l', 'building-m', 'building-n'];
const TOWERS = ['building-skyscraper-a', 'building-skyscraper-b', 'building-skyscraper-c', 'building-skyscraper-d', 'building-skyscraper-e'];
const models = {};

export async function loadCityAssets(onProgress) {
  const loader = new GLTFLoader();
  const all = [...BUILDINGS, ...TOWERS];
  let done = 0;
  await Promise.all(all.map(async n => {
    const g = await withRetry(() => loader.loadAsync(`assets/city/${n}.glb`));
    g.scene.traverse(o => {
      if (!o.isMesh) return;
      o.castShadow = false; o.receiveShadow = false;
      const m = o.material;
      // 夜景：墙面压暗，窗户用贴图自发光
      if (m.userData.night) return;
      m.userData.night = true;
      m.color.multiplyScalar(.16);
      m.emissive = new THREE.Color(0x9fd8ff);
      m.emissiveMap = m.map;
      m.emissiveIntensity = .55;
      m.roughness = .5; m.metalness = .5;
      keep(m, o.geometry);
      // 只让偏蓝的窗户发光，墙面保持暗色
      m.onBeforeCompile = sh => {
        sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `
          #ifdef USE_EMISSIVEMAP
            vec4 ec = texture2D( emissiveMap, vEmissiveMapUv );
            float win = smoothstep(.04, .16, ec.b - ec.r) * smoothstep(.45, .7, ec.b);
            totalEmissiveRadiance *= win;
          #endif`);
      };
      m.needsUpdate = true;
    });
    models[n] = g.scene;
    onProgress && onProgress(++done / all.length);
  }));
}

const rnd = (a, b) => a + Math.random() * (b - a);
const pick = a => a[Math.floor(Math.random() * a.length)];
const glow = (c, i = 2.5) => new THREE.MeshStandardMaterial({ color: 0x000000, emissive: c, emissiveIntensity: i, toneMapped: false });

// ---------- 城市地标（程序建模，比例按真实建筑缩放） ----------
const GM = (c, i = 2) => new THREE.MeshStandardMaterial({ color: 0x000000, emissive: c, emissiveIntensity: i, toneMapped: false });
const steel = () => new THREE.MeshStandardMaterial({ color: 0x1a1c2c, roughness: .3, metalness: .85 });
function edgesOf(mesh, color, i = 1.6) {
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry, 20), new THREE.LineBasicMaterial({ color, toneMapped: false }));
  e.position.copy(mesh.position); e.rotation.copy(mesh.rotation); e.scale.copy(mesh.scale);
  return e;
}
const LANDMARKS = {
  香港: [
    w => { // 国际金融中心二期
      const g = new THREE.Group(), m = w.windowMats()[1];
      const b = w.boxBuilding(4, 30, 4, m); g.add(b);
      const b2 = w.boxBuilding(3.4, 4, 3.4, m); b2.position.y = 30; g.add(b2);
      const b3 = w.boxBuilding(2.8, 2, 2.8, m); b3.position.y = 34; g.add(b3);
      for (let i = 0; i < 5; i++) { const f = new THREE.Mesh(new THREE.BoxGeometry(.25, 3 + (i % 2) * 1.5, .3), GM(0xe8f4ff, 1.6)); f.position.set(-1.1 + i * .55, 37 + (i % 2) * .7, 1.2); g.add(f); }
      return { obj: g, w: 4 };
    },
    w => { // 环球贸易广场
      const g = new THREE.Group(), m = w.windowMats()[1];
      const b = w.boxBuilding(4.4, 38, 4.4, m); g.add(b);
      for (let i = 0; i < 9; i++) { const s = new THREE.Mesh(new THREE.BoxGeometry(.12, 36, .05), GM(new THREE.Color().setHSL(.55 + i * .04, 1, .6), 1.4)); s.position.set(-2 + i * .5, 19, 2.23); g.add(s); }
      const top = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 2.6, 2, 4), steel()); top.rotation.y = Math.PI / 4; top.position.y = 39; g.add(top);
      return { obj: g, w: 4.4 };
    },
    w => { // 中银大厦：三角棱柱 + 白色 X 形钢架
      const g = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color: 0x1b2440, roughness: .15, metalness: .9 });
      const parts = [[0, 32], [1.4, 26], [-1.4, 20]];
      for (const [x, h] of parts) {
        const p = new THREE.Mesh(new THREE.CylinderGeometry(.01, 2.2, h, 3), mat); p.position.set(x, h / 2, 0); p.rotation.y = Math.PI / 6; g.add(p);
        g.add(edgesOf(p, 0xeaf4ff));
        const box = new THREE.Mesh(new THREE.BoxGeometry(2.6, h * .55, 2.6), mat); box.position.set(x, h * .275, 0); g.add(box);
        const x1 = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x - 1.3, 0, 1.31), new THREE.Vector3(x + 1.3, h * .55, 1.31), new THREE.Vector3(x + 1.3, 0, 1.31), new THREE.Vector3(x - 1.3, h * .55, 1.31)]), new THREE.LineBasicMaterial({ color: 0xffffff, toneMapped: false }));
        g.add(x1);
      }
      for (const x of [-.4, .4]) { const mast = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, 5), steel()); mast.position.set(x, 34, 0); g.add(mast); }
      const blink = new THREE.Mesh(new THREE.SphereGeometry(.15, 8, 6), GM(0xff2244, 5)); blink.position.y = 36.6; g.add(blink);
      return { obj: g, w: 5, blink };
    },
  ],
  新加坡: [
    w => { // 滨海湾金沙
      const g = new THREE.Group(), m = w.windowMats()[0];
      for (let i = 0; i < 3; i++) { const t = w.boxBuilding(2, 18, 3, m); t.position.x = i * 4; g.add(t); }
      const park = new THREE.Mesh(new THREE.BoxGeometry(14, .7, 3.4), steel()); park.position.set(5, 18.4, 0); g.add(park);
      const bow = new THREE.Mesh(new THREE.ConeGeometry(.5, 3, 4), steel()); bow.rotation.z = -Math.PI / 2; bow.position.set(13.4, 18.4, 0); g.add(bow);
      const rim = new THREE.Mesh(new THREE.BoxGeometry(16, .08, .08), GM(0x33f0ff, 2)); rim.position.set(5.6, 18.8, 1.72); g.add(rim);
      for (let i = 0; i < 16; i++) { const t = new THREE.Mesh(new THREE.SphereGeometry(.22, 6, 4), new THREE.MeshStandardMaterial({ color: 0x2a8a3a, emissive: 0x1a6a2a, emissiveIntensity: .6 })); t.position.set(-1 + i * .8, 19, 0); g.add(t); }
      return { obj: g, w: 14 };
    },
    () => { // 摩天观景轮
      const g = new THREE.Group();
      const ring = new THREE.Mesh(new THREE.TorusGeometry(8, .12, 8, 80), GM(0x33f0ff, 2.2)); ring.position.y = 10; g.add(ring);
      const ring2 = new THREE.Mesh(new THREE.TorusGeometry(7.4, .05, 6, 80), GM(0xff9ad5, 1.6)); ring2.position.y = 10; g.add(ring2);
      const spokes = []; for (let i = 0; i < 28; i++) { const a = i / 28 * Math.PI * 2; spokes.push(new THREE.Vector3(0, 10, 0), new THREE.Vector3(Math.cos(a) * 8, 10 + Math.sin(a) * 8, 0)); }
      g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(spokes), new THREE.LineBasicMaterial({ color: 0x8899bb })));
      for (let i = 0; i < 28; i++) { const a = i / 28 * Math.PI * 2; const c = new THREE.Mesh(new THREE.SphereGeometry(.28, 8, 6), GM(0xffffff, 1.5)); c.position.set(Math.cos(a) * 8.3, 10 + Math.sin(a) * 8.3, 0); g.add(c); }
      for (const x of [-2.5, 2.5]) { const leg = new THREE.Mesh(new THREE.BoxGeometry(.3, 11, .3), steel()); leg.position.set(x / 2, 5, 0); leg.rotation.z = x > 0 ? .22 : -.22; g.add(leg); }
      return { obj: g, w: 16 };
    },
  ],
  台北: [
    () => { // 台北 101：八节斗形塔身
      const g = new THREE.Group();
      const glass = new THREE.MeshStandardMaterial({ color: 0x0f3a34, roughness: .12, metalness: .9, emissive: 0x0a3a2a, emissiveIntensity: .4 });
      const pod = new THREE.Mesh(new THREE.BoxGeometry(6, 5, 6), glass); pod.position.y = 2.5; g.add(pod);
      for (let i = 0; i < 8; i++) {
        const seg = new THREE.Mesh(new THREE.CylinderGeometry(2.9, 2.2, 4, 4, 1), glass); seg.rotation.y = Math.PI / 4; seg.position.y = 7 + i * 4; g.add(seg);
        const trim = new THREE.Mesh(new THREE.CylinderGeometry(2.95, 2.95, .14, 4, 1, true), GM(new THREE.Color().setHSL(.33 + i * .02, 1, .55), 2)); trim.rotation.y = Math.PI / 4; trim.position.y = 9 + i * 4; g.add(trim);
      }
      const t1 = new THREE.Mesh(new THREE.BoxGeometry(2.4, 2.5, 2.4), glass); t1.position.y = 38.2; g.add(t1);
      const t2 = new THREE.Mesh(new THREE.BoxGeometry(1.4, 2, 1.4), glass); t2.position.y = 40.4; g.add(t2);
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(.08, .2, 8), new THREE.MeshStandardMaterial({ color: 0xcfd6e6, metalness: 1, roughness: .2 })); sp.position.y = 45.4; g.add(sp);
      const blink = new THREE.Mesh(new THREE.SphereGeometry(.2, 8, 6), GM(0xff2244, 5)); blink.position.y = 49.5; g.add(blink);
      return { obj: g, w: 6, blink };
    },
  ],
  上海: [
    () => { // 东方明珠
      const g = new THREE.Group();
      const pink = new THREE.MeshStandardMaterial({ color: 0xff4fb8, emissive: 0xff2a9a, emissiveIntensity: 1.1, roughness: .25, metalness: .5 });
      for (const a of [0, 2.1, 4.2]) { const leg = new THREE.Mesh(new THREE.CylinderGeometry(.35, .5, 12), steel()); leg.position.set(Math.cos(a) * 2.2, 6, Math.sin(a) * 2.2); leg.rotation.set(Math.sin(a) * .2, 0, -Math.cos(a) * .2); g.add(leg); }
      for (const a of [0, 2.1, 4.2]) { const col = new THREE.Mesh(new THREE.CylinderGeometry(.45, .45, 26), steel()); col.position.set(Math.cos(a) * .7, 18, Math.sin(a) * .7); g.add(col); }
      const b1 = new THREE.Mesh(new THREE.SphereGeometry(3.2, 24, 16), pink); b1.position.y = 12; g.add(b1);
      for (let i = 0; i < 5; i++) { const s = new THREE.Mesh(new THREE.SphereGeometry(.55, 12, 8), pink); s.position.y = 18 + i * 2.2; g.add(s); }
      const b2 = new THREE.Mesh(new THREE.SphereGeometry(2, 20, 14), pink); b2.position.y = 30; g.add(b2);
      const b3 = new THREE.Mesh(new THREE.SphereGeometry(.8, 14, 10), pink); b3.position.y = 35; g.add(b3);
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(.08, .25, 8), steel()); sp.position.y = 40; g.add(sp);
      const blink = new THREE.Mesh(new THREE.SphereGeometry(.2, 8, 6), GM(0xff2244, 5)); blink.position.y = 44.2; g.add(blink);
      return { obj: g, w: 7, blink };
    },
    w => { // 上海中心：分段扭转
      const g = new THREE.Group(), m = w.windowMats()[1];
      for (let i = 0; i < 12; i++) {
        const r = 2.6 - i * .14;
        const seg = new THREE.Mesh(new THREE.CylinderGeometry(r - .07, r, 3.6, 5), new THREE.MeshStandardMaterial({ color: 0x1a3550, roughness: .1, metalness: .9, emissive: 0x2a6aa0, emissiveIntensity: .35 }));
        seg.position.y = 1.8 + i * 3.6; seg.rotation.y = i * .12; g.add(seg); g.add(edgesOf(seg, 0x9fdcff));
      }
      return { obj: g, w: 5.2 };
    },
    w => { // 环球金融中心
      const g = new THREE.Group(), m = w.windowMats()[3];
      const shape = new THREE.Shape(); shape.moveTo(-2, 0); shape.lineTo(2, 0); shape.lineTo(1.3, 36); shape.lineTo(-1.3, 36); shape.closePath();
      const hole = new THREE.Path(); hole.moveTo(-.8, 31); hole.lineTo(.8, 31); hole.lineTo(.7, 34); hole.lineTo(-.7, 34); hole.closePath(); shape.holes.push(hole);
      const geo = new THREE.ExtrudeGeometry(shape, { depth: 3, bevelEnabled: false }); geo.translate(0, 0, -1.5);
      const b = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x18243a, roughness: .12, metalness: .9, emissive: 0x16304a, emissiveIntensity: .5 }));
      g.add(b); g.add(edgesOf(b, 0x9fdcff));
      return { obj: g, w: 4 };
    },
  ],
  东京: [
    () => { // 东京塔
      const g = new THREE.Group();
      const red = new THREE.MeshStandardMaterial({ color: 0xff4a1a, emissive: 0xff3a0a, emissiveIntensity: .9, roughness: .4 });
      const white = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffe0c0, emissiveIntensity: .7 });
      const segs = [[4, 2.6, 8, red], [2.6, 1.6, 6, white], [1.6, .9, 6, red], [.9, .5, 5, white], [.5, .2, 5, red]];
      let y = 0;
      for (const [r1, r2, h, mat] of segs) {
        const s = new THREE.Mesh(new THREE.CylinderGeometry(r2, r1, h, 4, 1, true), mat); s.material.side = THREE.DoubleSide; s.material.wireframe = true; s.rotation.y = Math.PI / 4; s.position.y = y + h / 2; g.add(s);
        const solid = new THREE.Mesh(new THREE.CylinderGeometry(r2 * .5, r1 * .5, h, 4), mat); solid.position.y = y + h / 2; solid.rotation.y = Math.PI / 4; g.add(solid);
        y += h;
      }
      const deck = new THREE.Mesh(new THREE.BoxGeometry(3, .8, 3), white); deck.position.y = 8.4; g.add(deck);
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(.04, .12, 5), white); sp.position.y = y + 2.5; g.add(sp);
      return { obj: g, w: 8 };
    },
    () => { // 晴空塔
      const g = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color: 0xc8d8ff, emissive: 0x6a8cff, emissiveIntensity: .6, roughness: .3, metalness: .6, wireframe: true });
      const body = new THREE.Mesh(new THREE.CylinderGeometry(.9, 3, 36, 6, 12), mat); body.position.y = 18; g.add(body);
      const core = new THREE.Mesh(new THREE.CylinderGeometry(.7, 1.6, 36, 8), new THREE.MeshStandardMaterial({ color: 0x1a2240, roughness: .3, metalness: .8 })); core.position.y = 18; g.add(core);
      for (const [y, r] of [[25, 2], [31, 1.4]]) { const ring = new THREE.Mesh(new THREE.CylinderGeometry(r, r, .9, 24), GM(0xb48cff, 1.8)); ring.position.y = y; g.add(ring); }
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(.08, .3, 8), new THREE.MeshStandardMaterial({ color: 0xdde6ff })); sp.position.y = 40; g.add(sp);
      const blink = new THREE.Mesh(new THREE.SphereGeometry(.2, 8, 6), GM(0xff2244, 5)); blink.position.y = 44.2; g.add(blink);
      return { obj: g, w: 6, blink };
    },
  ],
  迪拜: [
    () => { // 哈利法塔：逐级收分
      const g = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color: 0x8aa0c8, roughness: .15, metalness: .95, emissive: 0x2a4a80, emissiveIntensity: .45 });
      const tiers = [[3.4, 8], [2.9, 7], [2.4, 7], [1.9, 6], [1.5, 6], [1.1, 5], [.8, 5], [.5, 4], [.3, 4]];
      let y = 0;
      for (const [r, h] of tiers) { const t = new THREE.Mesh(new THREE.CylinderGeometry(r * .92, r, h, 3), mat); t.position.y = y + h / 2; g.add(t); g.add(edgesOf(t, 0xc8e0ff)); y += h; }
      const sp = new THREE.Mesh(new THREE.CylinderGeometry(.04, .2, 10), mat); sp.position.y = y + 5; g.add(sp);
      const blink = new THREE.Mesh(new THREE.SphereGeometry(.2, 8, 6), GM(0xff2244, 5)); blink.position.y = y + 10.2; g.add(blink);
      return { obj: g, w: 7, blink };
    },
    () => { // 帆船酒店
      const g = new THREE.Group();
      const shape = new THREE.Shape(); shape.moveTo(0, 0); shape.lineTo(0, 22); shape.quadraticCurveTo(7, 15, 6.5, 0); shape.closePath();
      const geo = new THREE.ExtrudeGeometry(shape, { depth: 2.4, bevelEnabled: false }); geo.translate(-3, 0, -1.2);
      const sail = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x8a6cff, emissiveIntensity: .7, roughness: .3 }));
      g.add(sail); g.add(edgesOf(sail, 0xffffff));
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(.12, .12, 26), steel()); mast.position.set(-3, 13, 0); g.add(mast);
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 1.2, .15, 20), GM(0xffd36b, 1.4)); pad.position.set(1, 16, 1.4); g.add(pad);
      return { obj: g, w: 7 };
    },
  ],
};

export class World {
  constructor(scene) {
    this.scene = scene;
    this.t = 0;
    // ---- 天空 ----
    this.skyMat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { top: { value: new THREE.Color() }, mid: { value: new THREE.Color() }, bot: { value: new THREE.Color() } },
      vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }',
      fragmentShader: `uniform vec3 top; uniform vec3 mid; uniform vec3 bot; varying vec3 vP;
        void main(){ float h = vP.y; vec3 c = h > .12 ? mix(mid, top, smoothstep(.12, .7, h)) : mix(bot, mid, smoothstep(-.05, .12, h)); gl_FragColor = vec4(c, 1.); }`,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(500, 32, 16), this.skyMat);
    scene.add(this.sky);
    // 星星
    const sp = [];
    for (let i = 0; i < 900; i++) { const a = Math.random() * Math.PI * 2, e = .12 + Math.random() * 1.3; sp.push(Math.cos(a) * Math.cos(e) * 480, Math.sin(e) * 480, Math.sin(a) * Math.cos(e) * 480); }
    const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
    this.stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xffffff, size: 1.6, sizeAttenuation: false, transparent: true, fog: false, depthWrite: false }));
    scene.add(this.stars);
    // 复古条纹太阳
    this.sunTex = canvasTex(512, 512, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, '#ffffff'); gr.addColorStop(1, '#ffffff');
      g.fillStyle = gr; g.beginPath(); g.arc(w / 2, h / 2, w / 2 - 4, 0, Math.PI * 2); g.fill();
      g.globalCompositeOperation = 'destination-out';
      for (let i = 0; i < 9; i++) { const y = h * .55 + i * 26, hh = 4 + i * 2.6; g.fillRect(0, y, w, hh); }
    });
    this.sunMat = new THREE.MeshBasicMaterial({ map: this.sunTex, transparent: true, fog: false, toneMapped: false, depthWrite: false });
    this.sunGrad = canvasTex(4, 256, (g, w, h) => { const gr = g.createLinearGradient(0, 0, 0, h); gr.addColorStop(0, '#ffffff'); gr.addColorStop(1, '#888888'); g.fillStyle = gr; g.fillRect(0, 0, w, h); });
    this.sun = new THREE.Mesh(new THREE.PlaneGeometry(90, 90), this.sunMat);
    scene.add(this.sun);
    this.sunHalo = new THREE.Mesh(new THREE.PlaneGeometry(220, 220), new THREE.MeshBasicMaterial({
      map: canvasTex(256, 256, (g, w) => { const r = g.createRadialGradient(w / 2, w / 2, 10, w / 2, w / 2, w / 2); r.addColorStop(0, 'rgba(255,255,255,.55)'); r.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = r; g.fillRect(0, 0, w, w); }),
      transparent: true, fog: false, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    scene.add(this.sunHalo);
    // 极光
    this.auroraMat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, fog: false,
      uniforms: { t: { value: 0 }, k: { value: 0 } },
      vertexShader: 'varying vec2 vU; uniform float t; void main(){ vU = uv; vec3 p = position; p.y += sin(p.x*.02 + t*.6)*14.; p.z += sin(p.x*.015 + t*.4)*20.; gl_Position = projectionMatrix * modelViewMatrix * vec4(p,1.); }',
      fragmentShader: `varying vec2 vU; uniform float t; uniform float k;
        void main(){ float band = smoothstep(0.,.35,vU.y)*smoothstep(1.,.45,vU.y); float r = .5+.5*sin(vU.x*40.+t*1.5);
          vec3 c = mix(vec3(.2,1.,.6), vec3(.5,.4,1.), vU.y) ; gl_FragColor = vec4(c*(.6+.4*r), band*k*.55); }`,
    });
    this.aurora = new THREE.Mesh(new THREE.PlaneGeometry(900, 90, 120, 4), this.auroraMat);
    scene.add(this.aurora);

    // ---- 远景天际线（三层：都市 / 熔岩 / 冰原） ----
    this.farGroup = new THREE.Group(); scene.add(this.farGroup);
    this.farCity = this.makeFar(null);
    this.farLava = this.makeFar(this.volcanoCanvas());
    this.farIce = this.makeFar(this.iceCanvas());
    this.cityName = null;

    // ---- 中景：建筑 / 岩柱 / 冰晶，按所在区域决定样式，循环复用 ----
    this.slots = [];
    for (let i = 0; i < 46; i++) { const s = new THREE.Group(); scene.add(s); this.slots.push({ g: s, x: -1e9, band: i % 2 }); }
    this.mats = {
      rock: new THREE.MeshStandardMaterial({ color: 0x1c0a06, roughness: .95, flatShading: true }),
      lava: glow(0xff5a10, 3.2),
      ice: new THREE.MeshPhysicalMaterial({ color: 0x9fe8ff, roughness: .08, metalness: .1, transmission: .0, transparent: true, opacity: .82, flatShading: true, emissive: 0x1a6a8a, emissiveIntensity: .45 }),
      iceEdge: glow(0xc8ffff, 1.6),
    };
    Object.values(this.mats).forEach(m => keep(m));
    this.neonCols = [0xff2bd6, 0x33f0ff, 0xffe45c, 0xb26bff, 0x3ef0a0];

    // ---- 地面 ----
    this.gridTex = canvasTex(256, 256, (g, w, h) => {
      g.fillStyle = '#07030f'; g.fillRect(0, 0, w, h);
      g.strokeStyle = 'rgba(255,255,255,.95)'; g.lineWidth = 3; g.strokeRect(0, 0, w, h);
      g.strokeStyle = 'rgba(255,255,255,.25)'; g.lineWidth = 1; g.beginPath(); g.moveTo(w / 2, 0); g.lineTo(w / 2, h); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
    });
    this.gridTex.wrapS = this.gridTex.wrapT = THREE.RepeatWrapping;
    this.rainbowTex = canvasTex(256, 256, (g, w, h) => {
      const cols = ['#ff3b5c', '#ff9a2e', '#ffe45c', '#3ee06b', '#3ea0ff', '#6b4cff', '#c46bff'];
      cols.forEach((c, i) => { g.fillStyle = c; g.fillRect(0, i * h / 7, w, h / 7 + 1); });
      g.fillStyle = 'rgba(255,255,255,.35)'; g.fillRect(0, 0, w / 3, h);
    });
    this.rainbowTex.wrapS = this.rainbowTex.wrapT = THREE.RepeatWrapping;
    this.groundPool = [];
    this.hazTex = hazardTex();
    // 高架赛道下方的地面：都市街道 / 熔岩湖 / 冰面
    this.streetTex = canvasTex(512, 512, (g, w, h) => {
      g.fillStyle = '#0a0612'; g.fillRect(0, 0, w, h);
      g.fillStyle = '#14101e'; for (let i = 0; i < 40; i++) g.fillRect(Math.random() * w, Math.random() * h, 40 + Math.random() * 90, 30 + Math.random() * 70);
      g.strokeStyle = 'rgba(255,220,120,.55)'; g.lineWidth = 3; g.setLineDash([26, 22]);
      g.beginPath(); g.moveTo(0, h * .5); g.lineTo(w, h * .5); g.stroke();
      g.setLineDash([]); g.strokeStyle = 'rgba(255,255,255,.18)'; g.lineWidth = 2;
      g.beginPath(); g.moveTo(0, h * .2); g.lineTo(w, h * .2); g.moveTo(0, h * .8); g.lineTo(w, h * .8); g.stroke();
    }, { repeat: [40, 6] });
    this.lavaTex = canvasTex(512, 512, (g, w, h) => {
      g.fillStyle = '#ff5a00'; g.fillRect(0, 0, w, h);
      for (let i = 0; i < 70; i++) { const r = 20 + Math.random() * 60; const gr = g.createRadialGradient(0, 0, 2, 0, 0, r); gr.addColorStop(0, 'rgba(30,6,0,.95)'); gr.addColorStop(1, 'rgba(30,6,0,0)'); g.save(); g.translate(Math.random() * w, Math.random() * h); g.fillStyle = gr; g.beginPath(); g.arc(0, 0, r, 0, Math.PI * 2); g.fill(); g.restore(); }
      g.globalCompositeOperation = 'lighter'; g.strokeStyle = 'rgba(255,230,120,.7)'; g.lineWidth = 2;
      for (let i = 0; i < 30; i++) { g.beginPath(); let x = Math.random() * w, y = Math.random() * h; g.moveTo(x, y); for (let k = 0; k < 6; k++) { x += Math.random() * 40 - 20; y += Math.random() * 40 - 20; g.lineTo(x, y); } g.stroke(); }
    }, { repeat: [24, 4] });
    this.iceTex = canvasTex(512, 512, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, w, h); gr.addColorStop(0, '#a8e6ff'); gr.addColorStop(1, '#4f9cc8'); g.fillStyle = gr; g.fillRect(0, 0, w, h);
      g.strokeStyle = 'rgba(255,255,255,.7)'; g.lineWidth = 1.5;
      for (let i = 0; i < 40; i++) { g.beginPath(); let x = Math.random() * w, y = Math.random() * h; g.moveTo(x, y); for (let k = 0; k < 4; k++) { x += Math.random() * 60 - 30; y += Math.random() * 60 - 30; g.lineTo(x, y); } g.stroke(); }
    }, { repeat: [30, 5] });
    this.floorY = -2.6;
    this.floors = [this.streetTex, this.lavaTex, this.iceTex].map((tex, i) => {
      const mat = i === 1
        ? new THREE.MeshStandardMaterial({ color: 0x220800, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 1.1, roughness: .9 })
        : new THREE.MeshStandardMaterial({ map: tex, roughness: i === 2 ? .15 : .55, metalness: i === 2 ? .3 : .5, emissive: i === 0 ? 0xffffff : 0x0a3050, emissiveMap: i === 0 ? tex : null, emissiveIntensity: i === 0 ? .25 : .4, transparent: true });
      const m = new THREE.Mesh(new THREE.PlaneGeometry(260, 90), mat);
      m.rotation.x = -Math.PI / 2; m.position.y = this.floorY + i * .01; m.receiveShadow = true;
      scene.add(m);
      return m;
    });
    this.abyss = this.floors[0];
    // 高架桥墩
    this.piers = [];
    const pierMat = new THREE.MeshStandardMaterial({ color: 0x0c0818, metalness: .7, roughness: .4 });
    for (let i = 0; i < 12; i++) {
      const p = new THREE.Mesh(new THREE.BoxGeometry(.5, 1, 1.2), pierMat);
      p.geometry.translate(0, -.5, 0);
      scene.add(p); this.piers.push(p);
    }

    // ---- 粒子氛围：火星 / 雪花 ----
    const N = 500, ap = new Float32Array(N * 3);
    this.ambN = N; this.ambV = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) this.respawnAmb(ap, i, 0, true);
    const ag = new THREE.BufferGeometry(); ag.setAttribute('position', new THREE.BufferAttribute(ap, 3));
    this.amb = new THREE.Points(ag, new THREE.PointsMaterial({ size: .09, color: 0xffffff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
    scene.add(this.amb);

    // ---- 超级奖励关布景 ----
    this.bonus = new THREE.Group(); this.bonus.visible = false; scene.add(this.bonus);
    const cols = [0xff3b5c, 0xff9a2e, 0xffe45c, 0x3ee06b, 0x3ea0ff, 0x6b4cff, 0xc46bff];
    cols.forEach((c, i) => {
      const arc = new THREE.Mesh(new THREE.TorusGeometry(46 - i * 1.6, .8, 8, 80, Math.PI), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: .75, toneMapped: false, fog: false }));
      arc.position.set(20, -6, -90); this.bonus.add(arc);
    });
    this.clouds = [];
    const cm = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, emissive: 0xffd0f0, emissiveIntensity: .35, flatShading: true });
    for (let i = 0; i < 14; i++) {
      const c = new THREE.Group();
      for (let k = 0; k < 5; k++) { const s = new THREE.Mesh(new THREE.IcosahedronGeometry(rnd(1.4, 2.6), 1), cm); s.position.set(k * 1.6 - 3, Math.sin(k) * .8, rnd(-1, 1)); c.add(s); }
      c.position.set(rnd(-40, 80), rnd(9, 20), rnd(-60, -32));
      this.bonus.add(c); this.clouds.push(c);
    }
  }

  makeFar(canvas) {
    const mat = new THREE.MeshBasicMaterial({ transparent: true, fog: false, depthWrite: false, opacity: 0 });
    if (canvas) this.setFarCanvas(mat, canvas);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(900, 100), mat);
    m.position.z = -230;
    this.farGroup.add(m);
    return m;
  }
  setFarCanvas(mat, canvas) {
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = THREE.SRGBColorSpace; t.wrapS = THREE.RepeatWrapping; t.anisotropy = 8;
    mat.map = t; mat.needsUpdate = true;
    mat.userData.tileW = 100 * canvas.width / canvas.height; // 每张贴图在世界里的宽度
    t.repeat.x = 900 / mat.userData.tileW;
  }
  volcanoCanvas() {
    const c = document.createElement('canvas'); c.width = 2048; c.height = 340;
    const g = c.getContext('2d');
    for (let layer = 0; layer < 2; layer++) {
      g.fillStyle = layer ? '#1a0603' : '#2e0b04';
      g.beginPath(); g.moveTo(0, 340);
      for (let x = 0; x <= 2048; x += 16) {
        const v = Math.sin(x * .004 + layer * 2) * 50 + Math.sin(x * .013 + layer) * 22 + (layer ? 170 : 120);
        const peak = Math.max(0, 1 - Math.abs(((x + layer * 500) % 900) - 450) / 140) * (layer ? 90 : 150);
        g.lineTo(x, 340 - v - peak);
      }
      g.lineTo(2048, 340); g.fill();
    }
    g.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 6; i++) { // 熔岩河
      const x0 = (i * 377) % 2048;
      const lg = g.createLinearGradient(0, 60, 0, 340); lg.addColorStop(0, 'rgba(255,200,60,.9)'); lg.addColorStop(1, 'rgba(255,60,0,.2)');
      g.strokeStyle = lg; g.lineWidth = 3 + (i % 3);
      g.beginPath(); let x = x0, y = 120; g.moveTo(x, y);
      while (y < 340) { x += Math.sin(y * .05 + i) * 6; y += 8; g.lineTo(x, y); }
      g.stroke();
    }
    return c;
  }
  iceCanvas() {
    const c = document.createElement('canvas'); c.width = 2048; c.height = 340;
    const g = c.getContext('2d');
    for (let layer = 0; layer < 2; layer++) {
      const gr = g.createLinearGradient(0, 40, 0, 340);
      gr.addColorStop(0, layer ? '#d8f6ff' : '#9fd4ee'); gr.addColorStop(.4, layer ? '#4e86a8' : '#3a6a8c'); gr.addColorStop(1, '#0b2236');
      g.fillStyle = gr;
      g.beginPath(); g.moveTo(0, 340);
      let x = 0;
      while (x <= 2048) { const w = 80 + Math.random() * 160, h = (layer ? 140 : 200) + Math.random() * 110; g.lineTo(x + w / 2, 340 - h); g.lineTo(x + w, 340 - h * .35); x += w * .8; }
      g.lineTo(2048, 340); g.fill();
    }
    return c;
  }

  setCity(name, canvas) {
    if (this.cityName === name) return;
    this.cityName = name;
    this.setFarCanvas(this.farCity.material, canvas);
  }

  respawnAmb(arr, i, camX, init) {
    arr[i * 3] = camX + rnd(-10, 26); arr[i * 3 + 1] = init ? rnd(-1, 12) : rnd(-1, 12); arr[i * 3 + 2] = rnd(-8, 4);
  }

  windowMats() {
    if (this._winMats) return this._winMats;
    const palettes = [['#ffd59a', '#ffe9c4', '#ffb35c'], ['#9fd8ff', '#d8f1ff', '#7ab8ff'], ['#ffd59a', '#9fd8ff', '#ff9ad5'], ['#ffffff', '#b8fff0', '#ffe9c4']];
    this._winMats = palettes.map((pal, pi) => {
      const tex = canvasTex(256, 256, (g, w, h) => {
        g.fillStyle = '#05040a'; g.fillRect(0, 0, w, h);
        for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
          const on = Math.random() < (pi === 3 ? .35 : .55);
          g.fillStyle = on ? pick(pal) : '#14121e';
          g.globalAlpha = on ? .55 + Math.random() * .45 : 1;
          g.fillRect(c * 32 + 5, r * 32 + 6, 22, 20);
        }
        g.globalAlpha = 1;
      });
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      return keep(new THREE.MeshStandardMaterial({ color: 0x9090a0, map: tex, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 1.15, roughness: .35, metalness: .7 }));
    });
    this._roof = keep(new THREE.MeshStandardMaterial({ color: 0x0b0912, roughness: .8, metalness: .3 }));
    return this._winMats;
  }
  // 带窗户的楼体：侧面按尺寸平铺窗户，顶面不铺
  boxBuilding(w, h, d, mat) {
    const geo = new THREE.BoxGeometry(w, h, d);
    geo.translate(0, h / 2, 0);
    const uv = geo.attributes.uv;
    for (let f = 0; f < 6; f++) {
      const sx = (f < 2 ? d : w) / 1.6, sy = f === 2 || f === 3 ? 1 : h / 1.6;
      for (let k = 0; k < 4; k++) { const i = f * 4 + k; uv.setXY(i, uv.getX(i) * sx / 8, uv.getY(i) * sy / 8); }
    }
    const roof = this._roof;
    return new THREE.Mesh(geo, [mat, mat, roof, roof, mat, mat]);
  }
  makeTower(h, w, d) {
    const mats = this.windowMats(), mat = pick(mats), g = new THREE.Group();
    const body = this.boxBuilding(w, h, d, mat); g.add(body);
    let top = h;
    if (Math.random() < .55) { // 收分楼顶
      const h2 = rnd(2, 5); const t2 = this.boxBuilding(w * .7, h2, d * .7, mat); t2.position.y = h; g.add(t2); top = h + h2;
    }
    const col = pick(this.neonCols);
    for (const sx of [-1, 1]) { // 楼角霓虹竖线
      if (Math.random() < .5) continue;
      const e = new THREE.Mesh(new THREE.BoxGeometry(.07, h, .07), glow(col, 1.4)); e.position.set(sx * w / 2, h / 2, d / 2); g.add(e);
    }
    const crown = new THREE.Mesh(new THREE.BoxGeometry(w * 1.02, .1, d * 1.02), glow(col, 1.8)); crown.position.y = h; g.add(crown);
    let blink = null;
    if (Math.random() < .65) {
      const ant = new THREE.Mesh(new THREE.CylinderGeometry(.04, .07, 3), this._roof); ant.position.y = top + 1.5; g.add(ant);
      blink = new THREE.Mesh(new THREE.SphereGeometry(.13, 8, 6), glow(0xff2244, 5)); blink.position.y = top + 3.05; g.add(blink);
    }
    if (Math.random() < .3) { // 外墙大屏
      const scr = new THREE.Mesh(new THREE.PlaneGeometry(w * .8, w * .5), glow(pick(this.neonCols), 1.2)); scr.position.set(0, h * rnd(.5, .8), d / 2 + .02); g.add(scr);
    }
    return { obj: g, w, blink };
  }

  // 霓虹招牌：竖排中文或横排英文，随机配色
  makeSign() {
    this.signCache = this.signCache || [];
    if (this.signCache.length < 18) {
      const words = ['酒店', '茶餐厅', '大押', '药房', '夜市', '拉面', '冰室', '电玩', '麻雀', '金行', '书局', '影院', '湾仔', '旺角', '居酒屋', '网吧'];
      const eng = ['HOTEL', 'BAR', 'NOODLE', 'CLUB', '24H', 'ARCADE', 'LIVE', 'CAFE'];
      const col = pick(['#ff2bd6', '#33f0ff', '#ffe45c', '#ff4d6d', '#7dffb0', '#b48cff']);
      const vertical = Math.random() < .65;
      const text = vertical ? pick(words) : pick(eng);
      const n = text.length, cw = vertical ? 96 : 64 * n + 40, ch = vertical ? 80 * n + 40 : 110;
      const tex = canvasTex(cw, ch, (g, w, h) => {
        g.fillStyle = 'rgba(8,4,16,.92)'; g.fillRect(0, 0, w, h);
        g.strokeStyle = col; g.lineWidth = 5; g.shadowColor = col; g.shadowBlur = 16; g.strokeRect(8, 8, w - 16, h - 16);
        g.fillStyle = '#ffffff'; g.font = `900 ${vertical ? 64 : 70}px "PingFang SC", "Heiti SC", sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
        g.shadowBlur = 22;
        if (vertical) [...text].forEach((c, i) => g.fillText(c, w / 2, 20 + 40 + i * 80));
        else g.fillText(text, w / 2, h / 2 + 4);
        g.globalCompositeOperation = 'source-atop'; g.fillStyle = col; g.globalAlpha = .55; g.fillRect(0, 0, w, h);
      });
      const mat = keep(new THREE.MeshBasicMaterial({ map: tex, toneMapped: false, transparent: true }));
      const geo = keep(new THREE.PlaneGeometry(cw / 140, ch / 140));
      this.signCache.push({ mat, geo });
    }
    const { mat, geo } = pick(this.signCache);
    const m = new THREE.Mesh(geo, mat);
    return m;
  }

  // 中景单元：根据区域生成对应景物
  fillSlot(s, kind) {
    for (const c of [...s.g.children]) { s.g.remove(c); disposeTree(c); }
    s.kind = kind;
    const far = s.band === 1;
    if (kind === 'city') {
      if (far) {
        const list = LANDMARKS[this.cityName] || [];
        this.lmGap = (this.lmGap || 0) + 1;
        if (list.length && this.lmGap > 5 && Math.random() < .35) { // 每隔几栋出现一个城市地标
          this.lmGap = 0;
          this.lmIdx = ((this.lmIdx || 0) + 1) % list.length;
          const lm = list[this.lmIdx](this);
          s.g.add(lm.obj); s.w = lm.w; s.blink = lm.blink || null;
          return;
        }
        const t = this.makeTower(rnd(10, 26), rnd(2.4, 4.2), rnd(2.4, 4));
        s.g.add(t.obj); s.w = t.w; s.blink = t.blink;
        return;
      }
      const m = models[pick(BUILDINGS)].clone();
      m.scale.setScalar(rnd(2.8, 3.8));
      m.rotation.y = pick([0, Math.PI / 2, Math.PI, -Math.PI / 2]);
      s.g.add(m);
      const box = new THREE.Box3().setFromObject(m), size = box.getSize(new THREE.Vector3());
      const strip = new THREE.Mesh(new THREE.BoxGeometry(size.x * 1.01, .08, size.z * 1.01), glow(pick(this.neonCols), 1.6));
      strip.position.y = size.y + .02; s.g.add(strip);
      const sign = this.makeSign();
      const side = Math.random() < .6;
      sign.scale.setScalar(1.5);
      sign.position.set(side ? size.x / 2 + .1 : rnd(-size.x * .25, size.x * .25), rnd(1.6, Math.max(1.8, size.y * .65)), size.z / 2 + .08);
      s.g.add(sign);
      s.blink = null;
      s.w = size.x;
    } else if (kind === 'lava') {
      const n = far ? 3 : 2;
      for (let i = 0; i < n; i++) {
        const h = far ? rnd(7, 18) : rnd(3, 8), r = far ? rnd(1.6, 3) : rnd(.8, 1.6);
        const pillar = new THREE.Mesh(new THREE.CylinderGeometry(r * .55, r, h, 6, 3), this.mats.rock);
        const p = pillar.geometry.attributes.position;
        for (let k = 0; k < p.count; k++) p.setXYZ(k, p.getX(k) * rnd(.85, 1.15), p.getY(k), p.getZ(k) * rnd(.85, 1.15));
        pillar.geometry.computeVertexNormals();
        pillar.position.set(i * r * 1.6, h / 2, rnd(-1, 1));
        s.g.add(pillar);
        for (let k = 0; k < 3; k++) {
          const crack = new THREE.Mesh(new THREE.BoxGeometry(.08, h * rnd(.3, .7), .08), this.mats.lava);
          crack.position.set(pillar.position.x + rnd(-r * .4, r * .4), h * rnd(.3, .6), pillar.position.z + r * .7); crack.rotation.z = rnd(-.3, .3);
          s.g.add(crack);
        }
      }
      s.blink = null; s.w = 4;
    } else {
      const n = far ? 4 : 3;
      for (let i = 0; i < n; i++) {
        const h = far ? rnd(6, 16) : rnd(2, 6), r = far ? rnd(.8, 1.6) : rnd(.4, .9);
        const cr = new THREE.Mesh(new THREE.ConeGeometry(r, h, 5), this.mats.ice);
        cr.position.set(i * r * 1.3 + rnd(-.3, .3), h / 2, rnd(-1, 1)); cr.rotation.z = rnd(-.15, .15); cr.rotation.y = rnd(0, 3);
        s.g.add(cr);
        const edge = new THREE.Mesh(new THREE.CylinderGeometry(.02, .02, h * .95), this.mats.iceEdge);
        edge.position.copy(cr.position); edge.position.x += r * .3; edge.rotation.z = cr.rotation.z + .1;
        s.g.add(edge);
      }
      s.blink = null; s.w = 3;
    }
  }

  // 每帧：把地面、中景、远景同步到摄像机位置
  update(dt, camX, th, zoneKindAt, groundSegs, bonus) {
    this.t += dt;
    const sky = bonus ? ['#ffb8e8', '#ff8fd0', '#7a5cff'] : th.sky;
    this.skyMat.uniforms.top.value.set(sky[0]); this.skyMat.uniforms.mid.value.set(sky[1]); this.skyMat.uniforms.bot.value.set(sky[2]);
    this.sky.position.set(camX, 0, 0);
    this.stars.position.set(camX, 0, 0);
    this.stars.material.opacity = bonus ? .3 : (1 - th.embers * .7);
    this.sun.position.set(camX + 40, 34, -240);
    this.sunHalo.position.set(camX + 40, 34, -242);
    this.sunMat.color.set(bonus ? '#fff3c0' : th.sun[0]).lerp(new THREE.Color(bonus ? '#ff9ad5' : th.sun[1]), .45);
    this.sunHalo.material.color.set(bonus ? '#ffd0f0' : th.sun[1]);
    this.aurora.position.set(camX, 70, -200);
    this.auroraMat.uniforms.t.value = this.t;
    this.auroraMat.uniforms.k.value = bonus ? 0 : th.aurora;

    // 远景：跟随摄像机，贴图偏移制造视差
    this.farGroup.position.set(camX, 14, 0);
    const far = [[this.farCity, th.cityA], [this.farLava, th.rockA], [this.farIce, th.iceA]];
    for (const [m, a] of far) {
      m.material.opacity = bonus ? 0 : a;
      m.visible = m.material.opacity > .01;
      if (m.material.map) m.material.map.offset.x = (camX * .25) / m.material.userData.tileW;
    }

    // 中景循环
    const span = 120;
    for (const s of this.slots) {
      if (s.x < camX - 40) {
        const prev = this.slots.reduce((mx, o) => (o.band === s.band ? Math.max(mx, o.x + (o.w || 3)) : mx), camX - 40);
        s.x = Math.max(prev, camX - 40) + (s.band ? rnd(1.5, 5) : rnd(1, 4));
        if (s.x > camX + span) s.x = camX + rnd(-30, span);
        this.fillSlot(s, zoneKindAt(s.x / U));
        s.g.position.set(s.x, this.floorY, s.band ? rnd(-46, -30) : rnd(-19, -14));
      }
      s.g.visible = !bonus;
      if (s.blink) s.blink.visible = Math.floor(this.t * 1.6 + s.x) % 2 === 0;
    }

    // 地面
    let used = 0;
    for (const seg of groundSegs) {
      const x1 = Math.max(seg.x1 * U, camX - 30), x2 = Math.min(seg.x2 * U, camX + 60);
      if (x2 <= x1) continue;
      const m = this.groundMesh(used++), u = m.userData;
      const len = x2 - x1;
      m.position.set((x1 + x2) / 2, 0, 0);
      u.box.scale.set(len, 3, 3.2);
      u.edge.scale.x = len; u.edge2.scale.x = len;
      u.top.map = bonus ? u.rainbow : u.grid;
      u.top.map.repeat.set(len / 1.6, 2);
      u.top.map.offset.x = (x1 / 1.6) % 1;
      u.top.emissiveMap = u.top.map;
      u.top.emissive.set(bonus ? '#ffffff' : th.grid);
      u.top.emissiveIntensity = bonus ? .45 : .55;
      u.edgeMat.emissive.set(bonus ? '#ffffff' : th.edge);
      u.haz[0].position.x = -len / 2; u.haz[1].position.x = len / 2;
      u.haz[0].visible = !bonus && seg.x1 * U > camX - 30;
      u.haz[1].visible = !bonus && seg.x2 * U < camX + 60;
      m.visible = true;
    }
    for (let i = used; i < this.groundPool.length; i++) this.groundPool[i].visible = false;
    const fw = [th.cityA, th.rockA, th.iceA];
    this.floors.forEach((f, i) => {
      f.position.x = camX;
      f.visible = !bonus && fw[i] > .02;
      f.material.opacity = fw[i];
      f.material.transparent = fw[i] < .99;
      if (f.material.emissiveMap) f.material.emissiveMap.offset.x = camX / 260 * (f.material.emissiveMap.repeat.x);
      if (f.material.map) f.material.map.offset.x = camX / 260 * f.material.map.repeat.x;
    });
    if (fw[1] > .02) this.lavaTex.offset.y = this.t * .02;
    // 桥墩：每 6 个单位一根，排在有地面的地方
    let k = 0;
    const startX = Math.floor((camX - 20) / 6) * 6;
    for (let x = startX; x < camX + 50 && k < this.piers.length; x += 6) {
      const onRoad = groundSegs.some(g => x / U >= g.x1 && x / U <= g.x2);
      const p = this.piers[k++];
      p.visible = onRoad && !bonus;
      p.position.set(x, -3, 0); p.scale.y = -this.floorY - 3 + .01;
    }
    for (; k < this.piers.length; k++) this.piers[k].visible = false;

    // 氛围粒子
    const ap = this.amb.geometry.attributes.position.array;
    const ember = th.embers, snow = th.snow;
    this.amb.material.opacity = bonus ? 0 : Math.max(ember, snow) * .9;
    this.amb.material.color.set(ember > snow ? '#ffb000' : '#ffffff');
    for (let i = 0; i < this.ambN; i++) {
      if (ember > snow) { ap[i * 3 + 1] += dt * rnd(.5, 2); ap[i * 3] += dt * -.4; }
      else { ap[i * 3 + 1] -= dt * rnd(.5, 1.6); ap[i * 3] -= dt * 1.5; }
      if (ap[i * 3 + 1] > 13 || ap[i * 3 + 1] < -2 || ap[i * 3] < camX - 12) this.respawnAmb(ap, i, camX, false);
      if (ember <= snow && ap[i * 3 + 1] < -1) ap[i * 3 + 1] = 12;
    }
    this.amb.geometry.attributes.position.needsUpdate = true;

    this.bonus.visible = bonus;
    if (bonus) {
      this.bonus.position.x = camX;
      for (const c of this.clouds) { c.position.x -= dt * 3; if (c.position.x < -50) c.position.x = 80; }
    }
  }

  groundMesh(i) {
    if (this.groundPool[i]) return this.groundPool[i];
    const grid = this.gridTex.clone(); grid.needsUpdate = true;
    const rainbow = this.rainbowTex.clone(); rainbow.needsUpdate = true;
    const top = new THREE.MeshStandardMaterial({ color: 0x0c0718, map: grid, emissiveMap: grid, emissive: 0xff2bd6, emissiveIntensity: .9, roughness: .8, metalness: .15 });
    const side = new THREE.MeshStandardMaterial({ color: 0x07040e, roughness: .6, metalness: .4 });
    const m = new THREE.Group();
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), [side, side, top, side, side, side]);
    box.geometry.translate(0, -.5, 0);
    box.receiveShadow = true; m.add(box);
    const edgeMat = glow(0x33f0ff, 1.3);
    const edge = new THREE.Mesh(new THREE.BoxGeometry(1, .04, .04), edgeMat); edge.position.set(0, -.02, 1.6); m.add(edge);
    const edge2 = new THREE.Mesh(new THREE.BoxGeometry(1, .05, .05), edgeMat); edge2.position.set(0, -.02, -1.6); m.add(edge2);
    const hazMat = new THREE.MeshBasicMaterial({ map: this.hazTex });
    const haz = [0, 1].map(() => { const h = new THREE.Mesh(new THREE.BoxGeometry(.16, 1.2, 3.25), hazMat); h.position.y = -.58; m.add(h); return h; });
    m.userData = { box, top, edge, edge2, edgeMat, haz, grid, rainbow };
    this.scene.add(m);
    this.groundPool[i] = m;
    return m;
  }
}
