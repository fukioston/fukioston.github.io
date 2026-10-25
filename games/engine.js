// 中国象棋规则引擎。棋盘 board[r][c]：r=0 为黑方底线，r=9 为红方底线。
// 大写为红方（K帅 A仕 B相 N马 R车 C炮 P兵），小写为黑方。
// 着法坐标用 ICCS：列 a-i（红方从左到右），行 0-9（红方底线为 0），如 h2e2 = 炮二平五。
(function (root) {
  const INIT = [
    'rnbakabnr', '.........', '.c.....c.', 'p.p.p.p.p', '.........',
    '.........', 'P.P.P.P.P', '.C.....C.', '.........', 'RNBAKABNR',
  ];
  const FILES = 'abcdefghi';
  const ORTH = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const DIAG = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const KNIGHT = [[2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2]];
  const NAMES = { K: '帅', A: '仕', B: '相', N: '马', R: '车', C: '炮', P: '兵',
                  k: '将', a: '士', b: '象', n: '马', r: '车', c: '炮', p: '卒' };
  const RED_NUM = '一二三四五六七八九';

  const initialBoard = () => INIT.map(s => s.split('').map(ch => (ch === '.' ? null : ch)));
  const cloneBoard = b => b.map(row => row.slice());
  const isRed = p => !!p && p === p.toUpperCase();
  const inBoard = (r, c) => r >= 0 && r < 10 && c >= 0 && c < 9;
  const inPalace = (r, c, red) => c >= 3 && c <= 5 && (red ? r >= 7 && r <= 9 : r >= 0 && r <= 2);

  function pseudoMoves(b, r, c) {
    const p = b[r][c];
    const red = isRed(p);
    const out = [];
    const add = (r2, c2) => {
      if (!inBoard(r2, c2)) return;
      const q = b[r2][c2];
      if (q && isRed(q) === red) return;
      out.push([r, c, r2, c2]);
    };
    switch (p.toUpperCase()) {
      case 'K':
        for (const [dr, dc] of ORTH) if (inPalace(r + dr, c + dc, red)) add(r + dr, c + dc);
        break;
      case 'A':
        for (const [dr, dc] of DIAG) if (inPalace(r + dr, c + dc, red)) add(r + dr, c + dc);
        break;
      case 'B':
        for (const [dr, dc] of DIAG) {
          const r2 = r + 2 * dr, c2 = c + 2 * dc;
          if (!inBoard(r2, c2) || (red ? r2 < 5 : r2 > 4)) continue;
          if (b[r + dr][c + dc]) continue; // 塞象眼
          add(r2, c2);
        }
        break;
      case 'N':
        for (const [dr, dc] of KNIGHT) {
          if (!inBoard(r + dr, c + dc)) continue;
          const lr = Math.abs(dr) === 2 ? r + dr / 2 : r;
          const lc = Math.abs(dc) === 2 ? c + dc / 2 : c;
          if (b[lr][lc]) continue; // 蹩马腿
          add(r + dr, c + dc);
        }
        break;
      case 'R':
        for (const [dr, dc] of ORTH) {
          let r2 = r + dr, c2 = c + dc;
          while (inBoard(r2, c2)) {
            if (!b[r2][c2]) out.push([r, c, r2, c2]);
            else { add(r2, c2); break; }
            r2 += dr; c2 += dc;
          }
        }
        break;
      case 'C':
        for (const [dr, dc] of ORTH) {
          let r2 = r + dr, c2 = c + dc, jumped = false;
          while (inBoard(r2, c2)) {
            if (!jumped) {
              if (!b[r2][c2]) out.push([r, c, r2, c2]);
              else jumped = true;
            } else if (b[r2][c2]) {
              add(r2, c2);
              break;
            }
            r2 += dr; c2 += dc;
          }
        }
        break;
      case 'P': {
        add(r + (red ? -1 : 1), c);
        if (red ? r <= 4 : r >= 5) { add(r, c - 1); add(r, c + 1); }
        break;
      }
    }
    return out;
  }

  function findKing(b, red) {
    const k = red ? 'K' : 'k';
    for (let r = 0; r < 10; r++) for (let c = 3; c <= 5; c++) if (b[r][c] === k) return [r, c];
    return null;
  }

  function isInCheck(b, red) {
    const king = findKing(b, red);
    if (!king) return true;
    const other = findKing(b, !red);
    if (other && other[1] === king[1]) { // 白脸将
      let blocked = false;
      const [lo, hi] = [Math.min(king[0], other[0]), Math.max(king[0], other[0])];
      for (let r = lo + 1; r < hi; r++) if (b[r][king[1]]) { blocked = true; break; }
      if (!blocked) return true;
    }
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 9; c++) {
        const p = b[r][c];
        if (!p || isRed(p) === red) continue;
        for (const m of pseudoMoves(b, r, c)) if (m[2] === king[0] && m[3] === king[1]) return true;
      }
    }
    return false;
  }

  // 正在将军的对方棋子数（用于识别双将）
  function checkerCount(b, red) {
    const king = findKing(b, red);
    if (!king) return 0;
    let n = 0;
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 9; c++) {
        const p = b[r][c];
        if (!p || isRed(p) === red) continue;
        if (pseudoMoves(b, r, c).some(m => m[2] === king[0] && m[3] === king[1])) n++;
      }
    }
    return n;
  }

  function legalMoves(b, side) {
    const red = side === 'red';
    const out = [];
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 9; c++) {
        const p = b[r][c];
        if (!p || isRed(p) !== red) continue;
        for (const m of pseudoMoves(b, r, c)) {
          const captured = b[m[2]][m[3]];
          b[m[2]][m[3]] = p; b[r][c] = null;
          if (!isInCheck(b, red)) out.push(m);
          b[r][c] = p; b[m[2]][m[3]] = captured;
        }
      }
    }
    return out;
  }

  function applyMove(b, m) {
    const captured = b[m[2]][m[3]];
    b[m[2]][m[3]] = b[m[0]][m[1]];
    b[m[0]][m[1]] = null;
    return captured;
  }

  const toIccs = m => FILES[m[1]] + (9 - m[0]) + FILES[m[3]] + (9 - m[2]);
  function fromIccs(s) {
    const t = String(s).trim().toLowerCase();
    if (!/^[a-i][0-9][a-i][0-9]$/.test(t)) return null;
    return [9 - +t[1], FILES.indexOf(t[0]), 9 - +t[3], FILES.indexOf(t[2])];
  }

  // 中文记谱（如 炮二平五 / 马8进7），b 为走子前的局面
  function toChinese(b, m) {
    const [r, c, r2, c2] = m;
    const p = b[r][c];
    const red = isRed(p);
    const t = p.toUpperCase();
    const fileOf = col => (red ? 9 - col : col + 1);
    const num = n => (red ? RED_NUM[n - 1] : String(n));
    const same = [];
    for (let rr = 0; rr < 10; rr++) if (b[rr][c] === p) same.push(rr);
    let head;
    if (same.length > 1) {
      same.sort((x, y) => (red ? x - y : y - x)); // 前者在前
      const idx = same.indexOf(r);
      const labels = same.length === 2 ? ['前', '后'] : same.length === 3 ? ['前', '中', '后'] : null;
      head = (labels ? labels[idx] : num(idx + 1)) + NAMES[p];
    } else {
      head = NAMES[p] + num(fileOf(c));
    }
    if (r2 === r) return head + '平' + num(fileOf(c2));
    const forward = red ? r2 < r : r2 > r;
    const diagonal = t === 'N' || t === 'B' || t === 'A';
    return head + (forward ? '进' : '退') + num(diagonal ? fileOf(c2) : Math.abs(r2 - r));
  }

  const boardKey = (b, side) => b.map(row => row.map(p => p || '.').join('')).join('/') + ' ' + side;

  function boardText(b) {
    const lines = ['    a b c d e f g h i'];
    for (let r = 0; r < 10; r++) {
      lines.push(' ' + (9 - r) + '  ' + b[r].map(p => p || '.').join(' '));
      if (r === 4) lines.push('    ~~~~ 楚河 汉界 ~~~~');
    }
    return lines.join('\n');
  }

  function perft(b, side, depth) {
    if (depth === 0) return 1;
    let n = 0;
    for (const m of legalMoves(b, side)) {
      const cap = applyMove(b, m);
      n += perft(b, side === 'red' ? 'black' : 'red', depth - 1);
      b[m[0]][m[1]] = b[m[2]][m[3]]; b[m[2]][m[3]] = cap;
    }
    return n;
  }

  // ---------- 给 AI 的战术提示 ----------
  // 子力价值；兵卒过河后升值
  function pieceValue(p, r) {
    const t = p.toUpperCase();
    if (t === 'P') return (isRed(p) ? r <= 4 : r >= 5) ? 2 : 1;
    return { K: 100, R: 9, C: 4.5, N: 4, B: 2, A: 2 }[t];
  }
  // byRed 一方能吃到 (r, c) 的棋子价值列表（目标格须有对方棋子）
  function attackers(b, r, c, byRed) {
    const out = [];
    for (let rr = 0; rr < 10; rr++) for (let cc = 0; cc < 9; cc++) {
      const p = b[rr][cc];
      if (!p || isRed(p) !== byRed) continue;
      if (pseudoMoves(b, rr, cc).some(m => m[2] === r && m[3] === c)) out.push(pieceValue(p, rr));
    }
    return out;
  }
  // (r, c) 上的子是否有己方棋子保护：临时把它换成对方颜色再看己方能否吃到
  function defended(b, r, c) {
    const p = b[r][c];
    b[r][c] = isRed(p) ? p.toLowerCase() : p.toUpperCase();
    const n = attackers(b, r, c, isRed(p)).length;
    b[r][c] = p;
    return n > 0;
  }
  // red 方处于危险中的子：被攻击且无根，或可被更低价值的子吃掉
  function threatened(b, red) {
    const out = [];
    for (let r = 0; r < 10; r++) for (let c = 0; c < 9; c++) {
      const p = b[r][c];
      if (!p || isRed(p) !== red || p.toUpperCase() === 'K') continue;
      const atk = attackers(b, r, c, !red);
      if (!atk.length) continue;
      const v = pieceValue(p, r), guard = defended(b, r, c);
      if (!guard || Math.min(...atk) < v) out.push({ r, c, p, guard });
    }
    return out;
  }
  const sqName = (r, c) => FILES[c] + (9 - r);
  const pieceName = (p, r, c) => NAMES[p] + '@' + sqName(r, c);
  function material(b, red) {
    let s = 0;
    for (let r = 0; r < 10; r++) for (let c = 0; c < 9; c++) {
      const p = b[r][c];
      if (p && isRed(p) === red && p.toUpperCase() !== 'K') s += pieceValue(p, r);
    }
    return s;
  }
  // 每步合法着法的标签：吃子、将军、绝杀、走后会丢的子
  function analyzeMoves(b, side) {
    const red = side === 'red', opp = red ? 'black' : 'red';
    return legalMoves(b, side).map(m => {
      const tags = [];
      let score = 0;
      const cap = applyMove(b, m);
      const moved = b[m[2]][m[3]];
      if (cap) { tags.push('吃' + NAMES[cap]); score += pieceValue(cap, m[2]); }
      if (isInCheck(b, !red)) {
        if (!legalMoves(b, opp).length) { tags.push('绝杀!'); score += 1000; }
        else { tags.push('将军'); score += .5; }
      } else if (!legalMoves(b, opp).length) { tags.push('困毙!'); score += 1000; }
      let worst = 0;
      for (const t of threatened(b, red)) {
        const v = pieceValue(t.p, t.r);
        if (v > worst) worst = v;
        tags.push(`⚠${t.r === m[2] && t.c === m[3] ? '走后此子' : NAMES[t.p] + '@' + sqName(t.r, t.c)}${t.guard ? '可被低价子兑' : '无根受攻'}`);
      }
      score -= worst;
      b[m[0]][m[1]] = moved; b[m[2]][m[3]] = cap;
      return { m, tags, score };
    }).sort((a, b2) => b2.score - a.score);
  }

  const api = { NAMES, initialBoard, cloneBoard, isRed, legalMoves, applyMove, isInCheck, checkerCount,
                toIccs, fromIccs, toChinese, boardKey, boardText, perft,
                pieceValue, threatened, material, analyzeMoves, pieceName };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Xiangqi = api;
})(this);
