// 五子棋规则（15×15，黑先，无禁手，连成五子或以上即胜）。
// 棋盘 board[r][c]：r=0 在最上方；坐标写法为列 A-O + 行 1-15（最下方为第 1 行），如 H8 为天元。
(function (root) {
  const N = 15;
  const COLS = 'ABCDEFGHIJKLMNO';
  const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];

  const emptyBoard = () => Array.from({ length: N }, () => Array(N).fill(null));
  const inBoard = (r, c) => r >= 0 && r < N && c >= 0 && c < N;
  const toPos = (r, c) => COLS[c] + (N - r);
  function fromPos(s) {
    const m = /^([A-Oa-o])\s*(1[0-5]|[1-9])$/.exec(String(s).trim());
    return m ? [N - +m[2], COLS.indexOf(m[1].toUpperCase())] : null;
  }

  // (r, c) 处棋子所在的五连（或更长），没有则返回 null
  function winLine(b, r, c) {
    const p = b[r][c];
    if (!p) return null;
    for (const [dr, dc] of DIRS) {
      const line = [[r, c]];
      for (const s of [1, -1]) {
        let rr = r + dr * s, cc = c + dc * s;
        while (inBoard(rr, cc) && b[rr][cc] === p) { line.push([rr, cc]); rr += dr * s; cc += dc * s; }
      }
      if (line.length >= 5) return line;
    }
    return null;
  }

  // 落下后能立刻连成五子的空点
  function fivePoints(b, side) {
    const out = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      if (b[r][c]) continue;
      b[r][c] = side;
      if (winLine(b, r, c)) out.push([r, c]);
      b[r][c] = null;
    }
    return out;
  }

  // 在 (qr, qc) 落子后，是否形成经过 (pr, pc) 的五连
  function fiveThrough(b, side, qr, qc, pr, pc) {
    b[qr][qc] = side;
    let hit = false;
    for (const [dr, dc] of DIRS) {
      const line = [[qr, qc]];
      for (const s of [1, -1]) {
        let rr = qr + dr * s, cc = qc + dc * s;
        while (inBoard(rr, cc) && b[rr][cc] === side) { line.push([rr, cc]); rr += dr * s; cc += dc * s; }
      }
      if (line.length >= 5 && line.some(([r, c]) => r === pr && c === pc)) { hit = true; break; }
    }
    b[qr][qc] = null;
    return hit;
  }

  // 落下后形成活四或双冲四（之后有两个以上经过这颗子的成五点）的空点
  function openFourPoints(b, side) {
    const out = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      if (b[r][c]) continue;
      b[r][c] = side;
      if (!winLine(b, r, c)) {
        let n = 0;
        for (let qr = 0; qr < N && n < 2; qr++) for (let qc = 0; qc < N && n < 2; qc++) {
          if (!b[qr][qc] && Math.max(Math.abs(qr - r), Math.abs(qc - c)) <= 4 && fiveThrough(b, side, qr, qc, r, c)) n++;
        }
        if (n >= 2) out.push([r, c]);
      }
      b[r][c] = null;
    }
    return out;
  }

  // 在 (qr, qc) 落子后，沿 (dr, dc) 方向是否形成经过 (pr, pc) 的五连
  function fiveAlong(b, side, qr, qc, pr, pc, dr, dc) {
    b[qr][qc] = side;
    let lo = 0, hi = 0;
    while (inBoard(qr + dr * (lo - 1), qc + dc * (lo - 1)) && b[qr + dr * (lo - 1)][qc + dc * (lo - 1)] === side) lo--;
    while (inBoard(qr + dr * (hi + 1), qc + dc * (hi + 1)) && b[qr + dr * (hi + 1)][qc + dc * (hi + 1)] === side) hi++;
    b[qr][qc] = null;
    if (hi - lo + 1 < 5) return false;
    const k = dr ? (pr - qr) / dr : (pc - qc) / dc;
    return k >= lo && k <= hi && qr + dr * k === pr && qc + dc * k === pc;
  }
  // (r, c) 已落 side 的子后，沿一个方向的棋形：2=活四 1=冲四 0.5=活三 0=无
  function lineShape(b, side, r, c, dr, dc) {
    const empties = [];
    for (let k = -4; k <= 4; k++) {
      const rr = r + dr * k, cc = c + dc * k;
      if (k && inBoard(rr, cc) && !b[rr][cc]) empties.push([rr, cc]);
    }
    const fives = empties.filter(([qr, qc]) => fiveAlong(b, side, qr, qc, r, c, dr, dc)).length;
    if (fives >= 2) return 2;
    if (fives === 1) return 1;
    for (const [qr, qc] of empties) {
      b[qr][qc] = side;
      let n = 0;
      for (let k = -4; k <= 4 && n < 2; k++) {
        const rr = r + dr * k, cc = c + dc * k;
        if (inBoard(rr, cc) && !b[rr][cc] && fiveAlong(b, side, rr, cc, r, c, dr, dc)) n++;
      }
      b[qr][qc] = null;
      if (n >= 2) return 0.5;
    }
    return 0;
  }
  // side 方每个空点的威胁类型
  function threatPoints(b, side) {
    const res = { five: [], fourFour: [], fourThree: [], openFour: [], threeThree: [], four: [], three: [] };
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      if (b[r][c]) continue;
      b[r][c] = side;
      if (winLine(b, r, c)) { res.five.push([r, c]); b[r][c] = null; continue; }
      const shapes = DIRS.map(([dr, dc]) => lineShape(b, side, r, c, dr, dc));
      b[r][c] = null;
      const fours = shapes.filter(s => s >= 1).length, threes = shapes.filter(s => s === .5).length;
      if (shapes.includes(2)) res.openFour.push([r, c]);
      else if (fours >= 2) res.fourFour.push([r, c]);
      else if (fours && threes) res.fourThree.push([r, c]);
      else if (threes >= 2) res.threeThree.push([r, c]);
      else if (fours) res.four.push([r, c]);
      else if (threes) res.three.push([r, c]);
    }
    return res;
  }

  function boardText(b) {
    const lines = ['    ' + COLS.split('').join(' ')];
    for (let r = 0; r < N; r++) {
      lines.push(String(N - r).padStart(2) + '  ' + b[r].map(p => (p === 'black' ? 'X' : p === 'white' ? 'O' : '.')).join(' '));
    }
    return lines.join('\n');
  }

  const api = { N, COLS, emptyBoard, inBoard, toPos, fromPos, winLine, fivePoints, fiveThrough, openFourPoints, threatPoints, boardText };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Gomoku = api;
})(this);
