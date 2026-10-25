// 棋盘特效：大字招式名、粒子爆裂、震屏、闪光和合成音效（不依赖任何素材文件）。
(function (root) {
  let muted = false;
  try { muted = localStorage.getItem('fx-muted') === '1'; } catch {}

  let audio = null;
  function ac() {
    if (!audio) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      audio = new AC();
    }
    if (audio.state === 'suspended') audio.resume();
    return audio;
  }

  function tone(freq, dur, { type = 'sine', vol = .25, delay = 0, slide = 0 } = {}) {
    const a = ac();
    if (!a) return;
    const t = a.currentTime + delay;
    const o = a.createOscillator(), g = a.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(30, freq * slide), t + dur);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + .01);
    g.gain.exponentialRampToValueAtTime(.0001, t + dur);
    o.connect(g).connect(a.destination);
    o.start(t); o.stop(t + dur + .02);
  }

  function noise(dur, { vol = .3, delay = 0, freq = 1200 } = {}) {
    const a = ac();
    if (!a) return;
    const t = a.currentTime + delay;
    const buf = a.createBuffer(1, Math.floor(a.sampleRate * dur), a.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 2);
    const src = a.createBufferSource(), f = a.createBiquadFilter(), g = a.createGain();
    src.buffer = buf; f.type = 'lowpass'; f.frequency.value = freq; g.gain.value = vol;
    src.connect(f).connect(g).connect(a.destination);
    src.start(t);
  }

  const SOUNDS = {
    move() { tone(520, .07, { type: 'triangle', vol: .18 }); noise(.05, { vol: .15, freq: 2500 }); },
    capture() { noise(.18, { vol: .45, freq: 900 }); tone(180, .22, { type: 'square', vol: .12, slide: .5 }); },
    check() { tone(660, .12, { type: 'sawtooth', vol: .14 }); tone(880, .25, { type: 'sawtooth', vol: .14, delay: .12 }); },
    big() { tone(440, .12, { type: 'square', vol: .12 }); tone(660, .12, { type: 'square', vol: .12, delay: .1 }); tone(990, .3, { type: 'square', vol: .12, delay: .2 }); },
    mate() {
      noise(.6, { vol: .5, freq: 500 });
      tone(110, 1.2, { type: 'sawtooth', vol: .2, slide: .5 });
      [523, 659, 784, 1047].forEach((f, i) => tone(f, .5, { type: 'triangle', vol: .16, delay: .25 + i * .12 }));
    },
    coin() { tone(988, .06, { type: 'square', vol: .1 }); tone(1319, .14, { type: 'square', vol: .1, delay: .06 }); },
    jump() { tone(300, .15, { type: 'triangle', vol: .15, slide: 2.2 }); },
    hit() { noise(.35, { vol: .5, freq: 600 }); tone(140, .4, { type: 'sawtooth', vol: .18, slide: .4 }); },
    power() { [523, 659, 784, 1047, 1319].forEach((f, i) => tone(f, .12, { type: 'square', vol: .09, delay: i * .05 })); },
  };

  function sound(kind) {
    if (muted || !SOUNDS[kind]) return;
    try { SOUNDS[kind](); } catch {}
  }

  // 大字招式名。kind: check | capture | mate | big
  function banner(layer, text, { kind = 'big', sub = '', duration = 1300 } = {}) {
    const old = layer.querySelector('.fx-banner');
    if (old) old.remove();
    const el = document.createElement('div');
    el.className = 'fx-banner fx-' + kind;
    el.innerHTML = `<div class="fx-main">${text}</div>` + (sub ? `<div class="fx-sub">${sub}</div>` : '');
    el.style.animationDuration = duration + 'ms';
    layer.appendChild(el);
    setTimeout(() => el.remove(), duration);
  }

  // 在 (xPct, yPct) 处爆开粒子，坐标为棋盘层宽高的百分比
  function burst(layer, xPct, yPct, colors = ['#ffd36b', '#ff7b3a', '#e0322a'], count = 22) {
    const ring = document.createElement('div');
    ring.className = 'fx-ring';
    ring.style.left = xPct + '%'; ring.style.top = yPct + '%';
    ring.style.borderColor = colors[0];
    layer.appendChild(ring);
    setTimeout(() => ring.remove(), 600);
    for (let i = 0; i < count; i++) {
      const p = document.createElement('span');
      p.className = 'fx-particle';
      const ang = Math.random() * Math.PI * 2, dist = 30 + Math.random() * 70, size = 4 + Math.random() * 6;
      p.style.left = xPct + '%'; p.style.top = yPct + '%';
      p.style.width = p.style.height = size + 'px';
      p.style.background = colors[i % colors.length];
      p.style.setProperty('--dx', Math.cos(ang) * dist + 'px');
      p.style.setProperty('--dy', Math.sin(ang) * dist + 'px');
      p.style.animationDuration = 450 + Math.random() * 350 + 'ms';
      layer.appendChild(p);
      setTimeout(() => p.remove(), 900);
    }
  }

  function shake(el, strong) {
    el.classList.remove('fx-shake', 'fx-shake-strong');
    void el.offsetWidth; // 重新触发动画
    el.classList.add(strong ? 'fx-shake-strong' : 'fx-shake');
  }

  function flash(layer, color = 'rgba(255,255,255,.8)') {
    const el = document.createElement('div');
    el.className = 'fx-flash';
    el.style.background = color;
    layer.appendChild(el);
    setTimeout(() => el.remove(), 700);
  }

  // 音效开关按钮
  function muteButton(btn) {
    const paint = () => { btn.textContent = muted ? '🔇 音效' : '🔊 音效'; btn.title = muted ? '打开音效' : '关闭音效'; };
    paint();
    btn.onclick = () => {
      muted = !muted;
      try { localStorage.setItem('fx-muted', muted ? '1' : '0'); } catch {}
      paint();
      if (!muted) sound('move');
    };
  }

  root.FX = { sound, banner, burst, shake, flash, muteButton, isMuted: () => muted };
})(this);
